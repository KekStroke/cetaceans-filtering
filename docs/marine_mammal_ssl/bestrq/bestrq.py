"""
BEST-RQ self-supervised audio encoder for marine bioacoustics.
Variant: CONV-SUBSAMPLE (BEST-RQ-faithful).

Pipeline:
  audio(16k) -> log-mel(80) -> per-utt normalize
             -> Random-Projection Quantizer (FROZEN) -> per-frame target codes (from CLEAN mel)
  masked-mel -> conv-subsample (2x stride-2 = 4x time downsample)
             -> 6-layer transformer (dim 384, 6 heads)
             -> linear head (dim -> codebook_size)
             -> masked cross-entropy vs subsampled target codes (every 4th frame)

Key correctness points:
  * P (mel_dim->proj_dim) and C (codebook_size x proj_dim) are FIXED random buffers,
    seeded once, requires_grad=False, never trained.
  * Codebook rows and projected vectors are L2-normalized; target = argmax cosine sim.
  * Targets come from the CLEAN mel. Encoder sees the MASKED mel.
  * Loss computed ONLY at masked (subsampled) positions.
  * At init, CE ~= ln(codebook_size) ~= 9.0 because the head is randomly initialized
    and the RPQ target distribution is (near) uniform over the codebook.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


# ----------------------------------------------------------------------------- #
# Config
# ----------------------------------------------------------------------------- #
@dataclass
class BestRQConfig:
    # front-end
    sample_rate: int = 16000
    n_mels: int = 80
    n_fft: int = 400
    hop_length: int = 160
    f_min: float = 0.0
    f_max: float | None = 8000.0  # nyquist for 16k

    # random-projection quantizer (frozen SSL target)
    proj_dim: int = 16
    codebook_size: int = 8192
    rpq_seed: int = 1234

    # conv subsample (4x time downsample): 2 conv layers, each stride 2
    # conv internal channels kept smaller than encoder_dim to control the size
    # of the (channels*freq -> encoder_dim) projection; the transformer itself
    # stays at dim 384 per the variant spec.
    conv_dim: int = 256

    # transformer encoder
    encoder_dim: int = 384
    encoder_layers: int = 6
    encoder_heads: int = 6
    ffn_dim: int = 1024
    dropout: float = 0.1
    max_positions: int = 4096

    # masking (applied at mel-frame rate, before conv)
    mask_prob: float = 0.08      # fraction of frames chosen as span starts
    mask_span: int = 10          # span length in frames
    min_masks: int = 1

    # downsample factor implied by the two stride-2 conv layers
    @property
    def subsample_factor(self) -> int:
        return 4


# ----------------------------------------------------------------------------- #
# Random-Projection Quantizer (FROZEN target generator)
# ----------------------------------------------------------------------------- #
class RandomProjectionQuantizer(nn.Module):
    """Fixed random projection + fixed random codebook. Never trained.

    target_code(frame) = argmax_k cos_sim( normalize(P @ frame), normalize(C[k]) )
    Computed on the CLEAN, per-utterance-normalized mel.
    """

    def __init__(self, cfg: BestRQConfig):
        super().__init__()
        g = torch.Generator().manual_seed(cfg.rpq_seed)
        # projection: mel_dim (80) -> proj_dim (16)
        proj = torch.randn(cfg.n_mels, cfg.proj_dim, generator=g)
        # codebook: codebook_size (8192) x proj_dim (16), L2-normalized rows
        codebook = torch.randn(cfg.codebook_size, cfg.proj_dim, generator=g)
        codebook = F.normalize(codebook, dim=-1)
        # register as buffers so they move with .to(device)/.cuda() but never update
        self.register_buffer("projection", proj, persistent=True)
        self.register_buffer("codebook", codebook, persistent=True)
        # hard-freeze (buffers already have no grad, but be explicit)
        self.projection.requires_grad_(False)
        self.codebook.requires_grad_(False)

    @torch.no_grad()
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: (B, T, n_mels) clean normalized mel -> codes: (B, T) long."""
        # project: (B, T, n_mels) @ (n_mels, proj_dim) -> (B, T, proj_dim)
        proj = mel @ self.projection
        proj = F.normalize(proj, dim=-1)
        # cosine sim to every codebook row (codebook already L2-normalized):
        # (B, T, proj_dim) @ (proj_dim, codebook_size) -> (B, T, codebook_size)
        sim = proj @ self.codebook.t()
        codes = sim.argmax(dim=-1)  # (B, T)
        return codes


# ----------------------------------------------------------------------------- #
# Front-end: log-mel + per-utterance normalization
# ----------------------------------------------------------------------------- #
class LogMelFrontend(nn.Module):
    def __init__(self, cfg: BestRQConfig):
        super().__init__()
        self.cfg = cfg
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            f_min=cfg.f_min,
            f_max=cfg.f_max,
            power=2.0,
            center=True,
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """audio: (B, num_samples) -> mel: (B, T, n_mels) per-utt normalized log-mel."""
        # melspec is a fixed (non-learned) transform; run in fp32 for stability
        with torch.no_grad():
            mel = self.melspec(audio)              # (B, n_mels, T)
            mel = torch.log(mel + 1e-6)            # log-mel
        mel = mel.transpose(1, 2)                  # (B, T, n_mels)
        # per-utterance mean/std normalize over (T, n_mels)
        mean = mel.mean(dim=(1, 2), keepdim=True)
        std = mel.std(dim=(1, 2), keepdim=True)
        mel = (mel - mean) / (std + 1e-5)
        return mel


# ----------------------------------------------------------------------------- #
# Conv subsample: 2x stride-2 conv => 4x time downsample
# ----------------------------------------------------------------------------- #
class ConvSubsample(nn.Module):
    """Operates on mel feature sequence (B, T, n_mels) treated as 1-channel-ish.

    Implemented as 2D convs over (freq, time) like classic Conformer subsampling,
    downsampling time by 4 and projecting to conv_dim.
    """

    def __init__(self, cfg: BestRQConfig):
        super().__init__()
        self.cfg = cfg
        c = cfg.conv_dim
        # 2D conv stack over (mel, time): in_channels=1
        self.conv = nn.Sequential(
            nn.Conv2d(1, c, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(c, c, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        # after two stride-2 convs over freq dim (80 -> 40 -> 20), flatten freq
        freq_after = self._freq_after(cfg.n_mels)
        self.out_proj = nn.Linear(c * freq_after, cfg.encoder_dim)

    @staticmethod
    def _conv_out(n: int) -> int:
        # kernel 3, stride 2, pad 1
        return (n + 2 * 1 - 3) // 2 + 1

    def _freq_after(self, n_mels: int) -> int:
        return self._conv_out(self._conv_out(n_mels))

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: (B, T, n_mels) -> (B, T//4, encoder_dim)."""
        x = mel.unsqueeze(1)               # (B, 1, T, n_mels)
        x = self.conv(x)                   # (B, c, T', F')
        b, c, t, f = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b, t, c * f)  # (B, T', c*F')
        x = self.out_proj(x)               # (B, T', encoder_dim)
        return x


# ----------------------------------------------------------------------------- #
# Sinusoidal positional encoding
# ----------------------------------------------------------------------------- #
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_positions: int):
        super().__init__()
        pe = torch.zeros(max_positions, dim)
        pos = torch.arange(0, max_positions, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, dim)
        return x + self.pe[: x.size(1)].unsqueeze(0)


# ----------------------------------------------------------------------------- #
# BEST-RQ model
# ----------------------------------------------------------------------------- #
class BestRQModel(nn.Module):
    def __init__(self, cfg: BestRQConfig | None = None):
        super().__init__()
        self.cfg = cfg or BestRQConfig()
        cfg = self.cfg

        self.frontend = LogMelFrontend(cfg)
        self.quantizer = RandomProjectionQuantizer(cfg)

        # learned mask vector applied at mel-frame level (before conv subsample)
        self.mask_embedding = nn.Parameter(torch.randn(cfg.n_mels) * 0.02)

        self.subsample = ConvSubsample(cfg)
        self.pos_enc = SinusoidalPositionalEncoding(cfg.encoder_dim, cfg.max_positions)
        self.input_dropout = nn.Dropout(cfg.dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.encoder_dim,
            nhead=cfg.encoder_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=cfg.encoder_layers, enable_nested_tensor=False
        )
        self.encoder_norm = nn.LayerNorm(cfg.encoder_dim)

        # prediction head: encoder_dim -> codebook_size
        self.head = nn.Linear(cfg.encoder_dim, cfg.codebook_size)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # --------------------------------------------------------------------- #
    # Masking (at mel-frame rate)
    # --------------------------------------------------------------------- #
    def compute_mask(self, B: int, T: int, device) -> torch.Tensor:
        """Span-based mask. Returns bool (B, T): True = masked frame."""
        cfg = self.cfg
        mask = torch.zeros(B, T, dtype=torch.bool, device=device)
        # number of span starts per utterance
        num_spans = max(cfg.min_masks, int(round(cfg.mask_prob * T)))
        if T <= cfg.mask_span:
            # tiny sequence: mask a single short span
            for b in range(B):
                mask[b, : max(1, T // 2)] = True
            return mask
        max_start = T - cfg.mask_span
        for b in range(B):
            starts = torch.randint(0, max_start + 1, (num_spans,), device=device)
            for s in starts.tolist():
                mask[b, s : s + cfg.mask_span] = True
        # guarantee at least one masked frame per utterance
        for b in range(B):
            if not mask[b].any():
                s = int(torch.randint(0, max_start + 1, (1,)).item())
                mask[b, s : s + cfg.mask_span] = True
        return mask

    def subsample_mask(self, frame_mask: torch.Tensor, T_sub: int) -> torch.Tensor:
        """Align frame-rate mask to subsampled rate by taking every 4th frame.

        Mirrors target alignment exactly: subsampled position i corresponds to
        mel frame i*factor. A subsampled position is 'masked' iff that aligned
        mel frame is masked.
        """
        factor = self.cfg.subsample_factor
        idx = torch.arange(T_sub, device=frame_mask.device) * factor
        idx = idx.clamp(max=frame_mask.size(1) - 1)
        return frame_mask[:, idx]  # (B, T_sub)

    def align_targets(self, codes: torch.Tensor, T_sub: int) -> torch.Tensor:
        """Take every 4th frame code to match the subsampled time axis."""
        factor = self.cfg.subsample_factor
        idx = torch.arange(T_sub, device=codes.device) * factor
        idx = idx.clamp(max=codes.size(1) - 1)
        return codes[:, idx]  # (B, T_sub)

    # --------------------------------------------------------------------- #
    # Forward
    # --------------------------------------------------------------------- #
    def forward(self, audio: torch.Tensor, mask: torch.Tensor | None = None):
        """audio: (B, num_samples).

        Returns dict with logits, target codes, masked-position mask, loss, and
        instrumentation stats.
        """
        cfg = self.cfg
        mel = self.frontend(audio)                 # (B, T, n_mels) CLEAN normalized
        B, T, _ = mel.shape

        # targets from CLEAN mel (frozen RPQ), then aligned to subsampled rate
        codes_full = self.quantizer(mel)           # (B, T)

        # build / use frame-level mask
        if mask is None:
            frame_mask = self.compute_mask(B, T, mel.device)  # (B, T)
        else:
            frame_mask = mask

        # apply mask: replace masked mel frames with the learned mask embedding
        masked_mel = mel.clone()
        mask_vec = self.mask_embedding.to(mel.dtype)
        masked_mel[frame_mask] = mask_vec

        # conv subsample (4x) -> (B, T_sub, enc_dim)
        x = self.subsample(masked_mel)
        T_sub = x.size(1)

        # align targets + mask to subsampled axis
        target_codes = self.align_targets(codes_full, T_sub)     # (B, T_sub)
        sub_mask = self.subsample_mask(frame_mask, T_sub)        # (B, T_sub) bool

        # transformer encoder
        x = self.pos_enc(x)
        x = self.input_dropout(x)
        x = self.encoder(x)
        x = self.encoder_norm(x)

        logits = self.head(x)                      # (B, T_sub, codebook_size)

        loss, stats = self._masked_ce(logits, target_codes, sub_mask)
        return {
            "loss": loss,
            "logits": logits,
            "target_codes": target_codes,
            "sub_mask": sub_mask,
            "frame_mask": frame_mask,
            "T": T,
            "T_sub": T_sub,
            **stats,
        }

    # --------------------------------------------------------------------- #
    # Masked cross-entropy + instrumentation
    # --------------------------------------------------------------------- #
    def _masked_ce(self, logits, target_codes, sub_mask):
        cfg = self.cfg
        C = cfg.codebook_size
        flat_logits = logits.reshape(-1, C)
        flat_targets = target_codes.reshape(-1)
        flat_mask = sub_mask.reshape(-1)

        if flat_mask.sum() == 0:
            # degenerate guard (should not happen): use all positions
            flat_mask = torch.ones_like(flat_mask)

        sel_logits = flat_logits[flat_mask]
        sel_targets = flat_targets[flat_mask]
        loss = F.cross_entropy(sel_logits, sel_targets)

        # ---- instrumentation ----
        with torch.no_grad():
            n_masked = int(flat_mask.sum().item())
            # predicted-code perplexity over masked positions (collapse check)
            preds = sel_logits.argmax(dim=-1)
            pred_ppl = self._perplexity(preds, C)
            # target-code perplexity over masked positions (RPQ usage)
            tgt_ppl = self._perplexity(sel_targets, C)
            # masked-position accuracy
            acc = (preds == sel_targets).float().mean().item()
            stats = {
                "n_masked": n_masked,
                "pred_perplexity": pred_ppl,
                "target_perplexity": tgt_ppl,
                "masked_acc": acc,
                "num_unique_targets": int(sel_targets.unique().numel()),
            }
        return loss, stats

    @staticmethod
    def _perplexity(codes: torch.Tensor, num_codes: int) -> float:
        """exp(entropy) of the code distribution (collapse check)."""
        counts = torch.bincount(codes, minlength=num_codes).float()
        probs = counts / counts.sum().clamp(min=1)
        nz = probs[probs > 0]
        entropy = -(nz * nz.log()).sum()
        return float(entropy.exp().item())


# ----------------------------------------------------------------------------- #
# Step logger (instrumentation)
# ----------------------------------------------------------------------------- #
class StepLogger:
    """Records loss, pre-clip grad-norm, and codebook-usage perplexity per step."""

    def __init__(self):
        self.history = []

    @staticmethod
    def grad_norm(parameters) -> float:
        total = 0.0
        for p in parameters:
            if p.grad is not None:
                total += float(p.grad.detach().norm(2).item()) ** 2
        return total ** 0.5

    def log(self, step: int, out: dict, grad_norm: float):
        rec = {
            "step": step,
            "loss": float(out["loss"].detach().item()),
            "grad_norm": float(grad_norm),
            "pred_perplexity": out["pred_perplexity"],
            "target_perplexity": out["target_perplexity"],
            "masked_acc": out["masked_acc"],
            "n_masked": out["n_masked"],
        }
        self.history.append(rec)
        return rec


def count_params(model: nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    cfg = BestRQConfig()
    model = BestRQModel(cfg)
    n = count_params(model)
    print(f"BestRQModel params: {n/1e6:.2f}M")
    audio = torch.randn(2, 32000)
    out = model(audio)
    print("loss", float(out["loss"].detach()), "T", out["T"], "T_sub", out["T_sub"],
          "n_masked", out["n_masked"], "target_ppl", out["target_perplexity"])
