#!/usr/bin/env python3
# BEST-RQ (Chung et al. 2021, "Self-supervised Learning with Random-projection Quantizer for
# Speech Recognition"; the USM objective). Opt-in alternative to the data2vec-2.0 latent-regression
# target in data2vec2.py — a *frozen* random-projection quantizer turns a fixed log-mel view of the
# input into discrete codebook labels, and the encoder is trained with masked cross-entropy to
# predict them. No EMA teacher, no learned codebook -> no representation/codebook collapse, much
# simpler than data2vec/wav2vec2.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LogMelTargets(nn.Module):
    """Fixed (non-learnable) log-mel feature used as the BEST-RQ quantizer input. Kept FROZEN on
    purpose: BEST-RQ's stability comes from the target view of the signal not drifting during
    training (unlike the learned SincNet frontend, which would let the model make targets trivial).
    forward() pools the log-mel to exactly `n_frames` so it aligns 1:1 with the encoder sequence."""

    def __init__(self, sample_rate=16000, n_fft=512, hop=160, n_mels=80, fmin=20.0, fmax=None):
        super().__init__()
        import librosa  # available via pyproject (declared); only imported at model build time
        fb = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels,
                                 fmin=fmin, fmax=fmax or sample_rate / 2)
        self.register_buffer("mel_fb", torch.tensor(fb, dtype=torch.float32))   # (n_mels, F)
        self.register_buffer("window", torch.hann_window(n_fft))
        self.n_fft, self.hop, self.n_mels = n_fft, hop, n_mels

    @torch.no_grad()
    def forward(self, wav, n_frames):
        # wav: (B, S) -> (B, n_frames, n_mels)
        if wav.dim() == 3:
            wav = wav.squeeze(1)
        # keep the frozen target view in float32 regardless of the model's bf16/amp cast of buffers
        spec = torch.stft(wav.float(), self.n_fft, self.hop,
                          window=self.window.float().to(wav.device),
                          return_complex=True, center=True)                     # (B, F, T')
        power = spec.abs().pow(2)
        mel = torch.einsum("mf,bft->bmt", self.mel_fb.float().to(wav.device), power)  # (B, n_mels, T')
        logmel = torch.log(mel + 1e-6)
        logmel = F.adaptive_avg_pool1d(logmel, n_frames)                         # (B, n_mels, n_frames)
        return logmel.transpose(1, 2).contiguous()                              # (B, n_frames, n_mels)


class RandomProjectionQuantizer(nn.Module):
    """Frozen random projection + frozen random codebook (both buffers, so they are saved with the
    checkpoint and reused on resume). Maps a feature vector to the index of the nearest codebook
    entry by cosine similarity. Never updated by the optimizer."""

    def __init__(self, input_dim, codebook_dim=16, codebook_size=8192, seed=42):
        super().__init__()
        g = torch.Generator().manual_seed(seed)                                  # deterministic init
        proj = torch.randn(input_dim, codebook_dim, generator=g) / math.sqrt(input_dim)
        codebook = F.normalize(torch.randn(codebook_size, codebook_dim, generator=g), dim=-1)
        self.register_buffer("proj", proj)                                       # (input_dim, codebook_dim)
        self.register_buffer("codebook", codebook)                               # (codebook_size, codebook_dim)
        self.codebook_size = codebook_size

    @torch.no_grad()
    def forward(self, x):
        # x: (B, T, input_dim) -> indices (B, T). Force float32: frozen + stable + bf16-cast-safe.
        x = F.normalize(x.float(), dim=-1)               # per-frame; no batch stats -> stable targets
        p = F.normalize(x @ self.proj.float(), dim=-1)   # (B, T, codebook_dim)
        sim = p @ self.codebook.float().t()              # (B, T, codebook_size)
        return sim.argmax(dim=-1)                         # (B, T) long


class BestRQModule(nn.Module):
    """Bundles the fixed log-mel + frozen quantizer + the (trainable) linear prediction head.
    targets(source, T) -> (B, T) codebook indices;  logits(h) -> (..., codebook_size)."""

    def __init__(self, encoder_dim, sample_rate=16000, codebook_size=8192, codebook_dim=16,
                 n_mels=80, n_fft=512, hop=160, seed=42):
        super().__init__()
        self.logmel = LogMelTargets(sample_rate, n_fft, hop, n_mels)
        self.quantizer = RandomProjectionQuantizer(n_mels, codebook_dim, codebook_size, seed)
        self.head = nn.Linear(encoder_dim, codebook_size)                        # the only trained part
        self.codebook_size = codebook_size

    @torch.no_grad()
    def targets(self, source, n_frames):
        return self.quantizer(self.logmel(source, n_frames))                     # (B, n_frames)

    def logits(self, h):
        return self.head(h)                                                      # (..., codebook_size)
