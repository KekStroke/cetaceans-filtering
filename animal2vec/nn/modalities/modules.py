#!/usr/bin/env python3 -u
# Copyright (c) Max Planck Institute of Animal Behavior
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from fairseq.modules import (
    LayerNorm,
    SamePad,
    TransposeLast,
)


class SamePad2d(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        assert len(x.size()) == 4
        if self.remove > 0:
            x = x[:, :, : -self.remove, : -self.remove]
        return x


@dataclass
class D2vDecoderConfig:
    decoder_dim: int = 384
    decoder_groups: int = 16
    decoder_kernel: int = 5
    decoder_layers: int = 5
    input_dropout: float = 0.1

    add_positions_masked: bool = False
    add_positions_all: bool = False

    decoder_residual: bool = True
    projection_layers: int = 1
    projection_ratio: float = 2.0


class FixedPositionalEncoder(nn.Module):
    def __init__(self, pos_embed):
        super().__init__()
        self.positions = pos_embed

    def forward(self, x, padding_mask):
        return self.positions


class TextFeatPositionalEncoder(nn.Module):
    """
    Original encoder expects (B, T) long input. This module wraps it to take
    local_encoder output which are (B, T, D) float tensors
    """

    def __init__(self, pos_encoder):
        super().__init__()
        self.pos_encoder = pos_encoder

    def forward(self, x, padding_mask):
        # assume padded token embeddings are 0s
        return self.pos_encoder(x[..., 0])


class BlockEncoder(nn.Module):
    def __init__(self, blocks, norm_layer, layer_norm_first, layerdrop, dropout):
        super().__init__()
        self.blocks = blocks
        self.norm = norm_layer
        self.layer_norm_first = layer_norm_first
        self.layerdrop = layerdrop
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, x, padding_mask, alibi_bias, alibi_scale):
        if self.norm is not None and not self.layer_norm_first:
            x = self.norm(x)

        x = self.dropout(x)

        for i, blk in enumerate(self.blocks):
            if (
                    not self.training
                    or self.layerdrop == 0
                    or (np.random.random() > self.layerdrop)
            ):
                ab = alibi_bias
                if ab is not None and alibi_scale is not None:
                    scale = (
                        alibi_scale[i]
                        if alibi_scale.size(0) > 1
                        else alibi_scale.squeeze(0)
                    )
                    ab = ab * scale.type_as(ab)
                x, _ = blk(x, padding_mask, ab)

        if self.norm is not None and self.layer_norm_first:
            x = self.norm(x)

        return x


class DecoderBase(nn.Module):
    decoder_cfg: D2vDecoderConfig

    def __init__(self, cfg: D2vDecoderConfig):
        super().__init__()

        self.decoder_cfg = cfg

    def reset_parameters(self):
        for mod in self.proj.modules():
            if isinstance(mod, nn.Linear):
                mod.reset_parameters()

    def add_residual(self, x, residual, i, mask_info):
        if (
                residual is None
                or not self.decoder_cfg.decoder_residual
                or residual.size(1) != x.size(1)
        ):
            return x

        ret = x + residual

        return ret


class Decoder1d(DecoderBase):
    def __init__(self, cfg: D2vDecoderConfig, input_dim):
        super().__init__(cfg)

        def make_block(in_dim):
            block = [
                nn.Conv1d(
                    in_dim,
                    cfg.decoder_dim,
                    kernel_size=cfg.decoder_kernel,
                    padding=cfg.decoder_kernel // 2,
                    groups=cfg.decoder_groups,
                ),
                SamePad(cfg.decoder_kernel),
                TransposeLast(),
                LayerNorm(cfg.decoder_dim, elementwise_affine=False),
                TransposeLast(),
                nn.GELU(),
            ]

            return nn.Sequential(*block)

        self.blocks = nn.Sequential(
            *[
                make_block(input_dim if i == 0 else cfg.decoder_dim)
                for i in range(cfg.decoder_layers)
            ]
        )

        projs = []
        curr_dim = cfg.decoder_dim
        for i in range(cfg.projection_layers - 1):
            next_dim = int(curr_dim * cfg.projection_ratio) if i == 0 else curr_dim
            projs.append(nn.Linear(curr_dim, next_dim))
            projs.append(nn.GELU())
            curr_dim = next_dim
        projs.append(nn.Linear(curr_dim, input_dim))
        if len(projs) == 1:
            self.proj = projs[0]
        else:
            self.proj = nn.Sequential(*projs)

    def forward(self, x, mask_info):

        x = x.transpose(1, 2)

        residual = x

        for i, layer in enumerate(self.blocks):
            x = layer(x)
            x = self.add_residual(x, residual, i, mask_info)
            residual = x

        x = x.transpose(1, 2)
        x = self.proj(x)
        return x


class Decoder2d(DecoderBase):
    def __init__(self, cfg: D2vDecoderConfig, input_dim, h_size, w_size):
        super().__init__(cfg)

        self.h_size = h_size
        self.w_size = w_size

        def make_block(in_dim):
            block = [
                nn.Conv2d(
                    in_dim,
                    cfg.decoder_dim,
                    kernel_size=cfg.decoder_kernel,
                    padding=cfg.decoder_kernel // 2,
                    groups=cfg.decoder_groups,
                ),
                SamePad2d(cfg.decoder_kernel),
                TransposeLast(tranpose_dim=-3),
                LayerNorm(cfg.decoder_dim, elementwise_affine=False),
                TransposeLast(tranpose_dim=-3),
                nn.GELU(),
            ]

            return nn.Sequential(*block)

        self.blocks = nn.Sequential(
            *[
                make_block(input_dim if i == 0 else cfg.decoder_dim)
                for i in range(cfg.decoder_layers)
            ]
        )

        self.proj = nn.Linear(cfg.decoder_dim, input_dim)

    def forward(self, x, mask_info):
        B, T, C = x.shape

        x = x.transpose(1, 2).reshape(B, C, self.h_size, self.w_size)

        residual = x

        for i, layer in enumerate(self.blocks):
            x = layer(x)
            x = self.add_residual(x, residual, i, mask_info)
            residual = x

        x = x.reshape(B, -1, T).transpose(1, 2)
        x = self.proj(x)
        return x


class TransformerDecoder(nn.Module):
    decoder_cfg: D2vDecoderConfig

    def __init__(self, cfg: D2vDecoderConfig, input_dim, encoder):
        super().__init__()

        self.decoder_cfg = cfg

        self.input_proj = nn.Linear(input_dim, cfg.decoder_dim)

        self.encoder = encoder

        self.proj = nn.Linear(cfg.decoder_dim, input_dim)

    def reset_parameters(self):
        from fairseq.modules.transformer_sentence_encoder import init_bert_params

        self.apply(init_bert_params)

    def forward(self, x, mask_info):
        x = self.input_proj(x)
        x = self.encoder(x, None, None, 1)
        x = self.proj(x)
        return x


class GatedMlp(nn.Module):
    """GLU-variant gated FFN (Shazeer 2020, "GLU Variants Improve Transformer").
    Drop-in alternative to timm's Mlp(in_features, hidden_features, act_layer, drop). To keep
    parameter/FLOP count at parity with a plain 2-matrix Mlp built at the SAME nominal
    hidden_features (= mlp_ratio * dim), this module applies the standard 2/3 rescale (3 weight
    matrices vs 2) and rounds up to a multiple of 8, matching the convention used by e.g.
    facebookresearch/dinov2's SwiGLUFFNFused.
    """

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            drop=0.0,
            variant="swiglu",  # one of {"swiglu", "geglu", "reglu"}
            multiple_of=8,
            bias=True,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # parity rescale: a 3-matrix gated FFN at hidden' has params 3*dim*hidden'; a plain
        # 2-matrix FFN at `hidden_features` has params 2*dim*hidden_features. Setting
        # hidden' = (2/3)*hidden_features equalizes both, then round to a hardware-friendly
        # multiple.
        hidden_features = int(hidden_features * 2 / 3)
        hidden_features = ((hidden_features + multiple_of - 1) // multiple_of) * multiple_of

        act_fns = {"swiglu": nn.SiLU(), "geglu": nn.GELU(), "reglu": nn.ReLU()}
        if variant not in act_fns:
            raise ValueError(f"unknown gated_mlp_variant={variant!r}, expected one of {list(act_fns)}")
        self.act = act_fns[variant]

        # fused gate+value projection (one matmul instead of two)
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.drop1 = nn.Dropout(drop)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        hidden = self.drop1(self.act(x1) * x2)
        return self.drop2(self.w3(hidden))


def _build_rope_cache(seq_len, head_dim, base, device, dtype):
    """Standard RoPE (Su et al. 2021, RoFormer) sin/cos cache, computed fresh per forward call
    (clip lengths vary; no cross-call caching to avoid device/seq-len bookkeeping complexity).
    Returns cos, sin each of shape (seq_len, head_dim // 2)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # (seq_len, head_dim // 2)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def _apply_rope(x, cos, sin):
    """x: (B, H, N, D). cos/sin: (N, D // 2). Rotates each adjacent (even, odd) pair of the head
    dim by position-dependent angles; preserves the L2 norm of x, so this composes cleanly with
    cosine_attention's F.normalize (rotate-then-normalize == normalize-then-rotate)."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[None, None, :, :].type_as(x)
    sin = sin[None, None, :, :].type_as(x)
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    return torch.stack([rx1, rx2], dim=-1).flatten(-2)


class AltBlock(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.0,
            qkv_bias=False,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            mlp_drop=0.0,
            post_mlp_drop=0.0,
            drop_path=0.0,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            layer_norm_first=True,
            ffn_targets=False,
            cosine_attention=False,
            attn_output_gate=None,  # NEW: None (default, no-op) | "headwise" | "elementwise"
            gated_mlp=False,        # NEW: default False -> identical to current Mlp behavior
            gated_mlp_variant="swiglu",  # NEW: only consulted when gated_mlp=True
            use_rope=False,         # NEW: default False -> identical to current behavior
            rope_base=10000.0,      # NEW: only consulted when use_rope=True
    ):
        super().__init__()

        self.layer_norm_first = layer_norm_first
        self.ffn_targets = ffn_targets

        from timm.models.vision_transformer import DropPath, Mlp

        self.norm1 = norm_layer(dim)
        self.attn = AltAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            cosine_attention=cosine_attention,
            attn_output_gate=attn_output_gate,
            use_rope=use_rope,
            rope_base=rope_base,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if gated_mlp:
            self.mlp = GatedMlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                drop=mlp_drop,
                variant=gated_mlp_variant,
            )
        else:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=mlp_drop,
            )
        self.post_mlp_dropout = nn.Dropout(post_mlp_drop, inplace=False)

    def forward(self, x, padding_mask=None, alibi_bias=None):
        if self.layer_norm_first:
            x = x + self.drop_path(self.attn(self.norm1(x), padding_mask, alibi_bias))
            r = x = self.mlp(self.norm2(x))
            t = x
            x = r + self.drop_path(self.post_mlp_dropout(x))
            if not self.ffn_targets:
                t = x
        else:
            x = x + self.drop_path(self.attn(x, padding_mask, alibi_bias))
            r = x = self.norm1(x)
            x = self.mlp(x)
            t = x
            x = self.norm2(r + self.drop_path(self.post_mlp_dropout(x)))
            if not self.ffn_targets:
                t = x

        return x, t


class AltAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            cosine_attention=False,
            attn_output_gate=None,  # NEW: None (default, no-op) | "headwise" | "elementwise"
            use_rope=False,         # NEW: default False -> identical to current behavior
            rope_base=10000.0,      # NEW: only consulted when use_rope=True
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.cosine_attention = cosine_attention

        if cosine_attention:
            self.logit_scale = nn.Parameter(
                torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True
            )

        # --- RoPE (RoFormer, Su et al. 2021) -------------------------------------------------
        # Applied to q/k right after the qkv split, ADDITIVE to whatever ALiBi/conv-pos-embed is
        # already configured at the modality level (set use_alibi_encoder=False separately for a
        # RoPE-only setup). KNOWN LIMITATION: position index here is simply arange(N) over
        # whatever sequence arrives at this forward call. Under masking with token removal, the
        # codebase's existing ALiBi path (nn/modalities/base.py::masked_alibi) correctly preserves
        # TRUE pre-masking distances via torch.gather on mask_info.ids_keep; this RoPE
        # implementation does NOT thread true positions through masking the same way (that would
        # require passing position ids through BlockEncoder/AltBlock.forward analogous to
        # alibi_bias, a materially larger and riskier change) -- so under masking, RoPE here sees
        # "compressed" positions (gaps from removed tokens collapsed), NOT the original spacing.
        # This is a deliberate, documented scope tradeoff, not an oversight.
        self.use_rope = use_rope
        self.rope_base = rope_base
        # ---------------------------------------------------------------------------------------

        # --- attention-output gate (Qwen3-Next / "Gated Attention", arXiv:2505.06708) -------
        # Sigmoid gate computed from the BLOCK INPUT (same x that feeds self.qkv), applied to
        # the post-softmax attention output before self.proj. None (default) is a strict no-op:
        # no new parameters, forward() reduces exactly to today's code path.
        assert attn_output_gate in (None, "headwise", "elementwise"), attn_output_gate
        self.attn_output_gate = attn_output_gate
        self.head_dim = head_dim
        if attn_output_gate == "headwise":
            self.gate_proj = nn.Linear(dim, num_heads)
        elif attn_output_gate == "elementwise":
            self.gate_proj = nn.Linear(dim, dim)
        else:
            self.gate_proj = None
        if self.gate_proj is not None:
            # Highway/GRU-style init: start the gate near "pass-through" (sigmoid(2.0) ~= 0.88)
            # so a freshly-started gated run begins close to the proven ungated function.
            nn.init.zeros_(self.gate_proj.weight)
            nn.init.constant_(self.gate_proj.bias, 2.0)
        # ---------------------------------------------------------------------------------------

    def forward(self, x, padding_mask=None, alibi_bias=None):
        B, N, C = x.shape
        gate_input = x  # block input, captured before `x` gets reassigned below

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)  # qkv x B x H x L x D
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        dtype = q.dtype

        if self.use_rope:
            cos, sin = _build_rope_cache(N, self.head_dim, self.rope_base, q.device, q.dtype)
            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)

        if self.cosine_attention:
            # cosine attention
            attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
            # NOTE: `max` must be a plain float, not a CPU tensor (torch.tensor(1.0/0.01)) --
            # this branch was never exercised before (cosine_attention was dead/unwired code),
            # and a tensor `max` here raises a CPU/CUDA device-mismatch error in torch.clamp.
            logit_scale = torch.clamp(
                self.logit_scale, max=math.log(1.0 / 0.01)
            ).exp()
            attn = attn * logit_scale
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)

        if alibi_bias is not None:
            attn = attn.type_as(alibi_bias)
            attn[:, : alibi_bias.size(1)] += alibi_bias

        if padding_mask is not None and padding_mask.any():
            attn = attn.masked_fill(
                padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )

        attn = attn.softmax(dim=-1, dtype=torch.float32).to(dtype=dtype)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2)  #
        x = x.reshape(B, N, C)

        if self.gate_proj is not None:
            gate_score = self.gate_proj(gate_input)  # (B, N, H) for headwise, (B, N, C) for elementwise
            if self.attn_output_gate == "headwise":
                gate_score = (
                    gate_score.unsqueeze(-1)
                    .expand(B, N, self.num_heads, self.head_dim)
                    .reshape(B, N, C)
                )
            x = x * torch.sigmoid(gate_score.type_as(x))

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class EncDecAttention(nn.Module):
    def __init__(
            self,
            q_dim,
            kv_dim,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            cosine_attention=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = q_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(q_dim, q_dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(kv_dim, 2 * q_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(q_dim, q_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.cosine_attention = cosine_attention

        if cosine_attention:
            self.logit_scale = nn.Parameter(
                torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True
            )

    def forward(self, q, kv, padding_mask=None, alibi_bias=None):
        B, N, C = q.shape

        q = (
            self.q_proj(q)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )  # B x H x L x D
        kv = (
            self.kv_proj(kv)
            .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )  # kv x B x H x L x D
        k, v = (
            kv[0],
            kv[1],
        )  # make torchscript happy (cannot use tensor as tuple)

        dtype = q.dtype

        if self.cosine_attention:
            # cosine attention
            attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
            # see AltAttention.forward for why `max` must be a plain float here
            logit_scale = torch.clamp(
                self.logit_scale, max=math.log(1.0 / 0.01)
            ).exp()
            attn = attn * logit_scale
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)

        if alibi_bias is not None:
            attn = attn.type_as(alibi_bias)
            attn[:, : alibi_bias.size(1)] += alibi_bias

        if padding_mask is not None and padding_mask.any():
            attn = attn.masked_fill(
                padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )

        attn = attn.softmax(dim=-1, dtype=torch.float32).to(dtype=dtype)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2)  #
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class EncDecBlock(nn.Module):
    def __init__(
            self,
            q_dim,
            kv_dim,
            num_heads,
            mlp_ratio=4.0,
            qkv_bias=False,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            mlp_drop=0.0,
            post_mlp_drop=0.0,
            drop_path=0.0,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            layer_norm_first=True,
            cosine_attention=False,
            first_residual=True,
    ):
        super().__init__()

        self.layer_norm_first = layer_norm_first

        from timm.models.vision_transformer import DropPath, Mlp

        self.norm1 = norm_layer(q_dim)
        self.attn = EncDecAttention(
            q_dim,
            kv_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            cosine_attention=cosine_attention,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(q_dim)
        mlp_hidden_dim = int(q_dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=q_dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=mlp_drop,
        )
        self.post_mlp_dropout = nn.Dropout(post_mlp_drop, inplace=False)
        self.first_residual = first_residual

    def forward(self, q, kv, padding_mask=None, alibi_bias=None):
        r = q if self.first_residual else 0
        if self.layer_norm_first:
            x = r + self.drop_path(
                self.attn(self.norm1(q), kv, padding_mask, alibi_bias)
            )
            r = x = self.mlp(self.norm2(x))
            x = r + self.drop_path(self.post_mlp_dropout(x))
        else:
            x = r + self.drop_path(self.attn(q, kv, padding_mask, alibi_bias))
            r = x = self.norm1(x)
            x = self.mlp(x)
            x = self.norm2(r + self.drop_path(self.post_mlp_dropout(x)))

        return x


class EncDecTransformerDecoder(nn.Module):
    def __init__(self, cfg: D2vDecoderConfig, input_dim):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, cfg.decoder_dim)

        self.blocks = nn.Sequential(
            *[
                EncDecBlock(
                    q_dim=cfg.decoder_dim,
                    kv_dim=input_dim,
                    num_heads=8,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    qk_scale=None,
                    drop=0.0,
                    attn_drop=0.0,
                    mlp_drop=0.0,
                    post_mlp_drop=0.0,
                    drop_path=0.0,
                    act_layer=nn.GELU,
                    norm_layer=nn.LayerNorm,
                    layer_norm_first=False,
                    cosine_attention=False,
                    first_residual=i > 0,
                )
                for i in range(cfg.decoder_layers)
            ]
        )

        self.proj = nn.Linear(cfg.decoder_dim, input_dim)

    def reset_parameters(self):
        from fairseq.modules.transformer_sentence_encoder import init_bert_params

        self.apply(init_bert_params)

    def forward(self, x, kv):
        x = self.input_proj(x)
        for i, layer in enumerate(self.blocks):
            x = layer(x, kv)

        x = self.proj(x)
        return x
