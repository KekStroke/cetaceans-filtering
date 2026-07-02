#!/usr/bin/env python3
"""Regression + correctness tests for the opt-in architecture flags added to data2vec_multi
(cosine_attention, attn_output_gate, use_gated_mlp, use_rope, use_frontend_glu_gate).

Self-contained: no checkpoint, no dataset, CPU-only friendly (GPU used if present). Run with:
    pytest animal2vec/tests/test_arch_flags.py
or standalone:
    python animal2vec/tests/test_arch_flags.py

Key regressions locked in here:
  * default-off flags do NOT add parameters and reproduce a plain scaled-dot-product attention
    exactly (backward-compat guarantee);
  * RoPE is norm-preserving, identity at position 0, and relative-position only;
  * FrontendGLUGate is an EXACT identity at init AND trainable (guards the "zero-init both convs
    -> permanently frozen, never learns" bug).
"""
import math
import torch

from animal2vec.torch2_compat import apply_torch2_fairseq_compat
apply_torch2_fairseq_compat()

import animal2vec.nn.modalities.modules as M
from animal2vec.nn.utils import FrontendGLUGate

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DIM, HEADS, B, N = 64, 4, 2, 20


def _ref_sdpa(attn_mod, x):
    """Plain scaled-dot-product self-attention using attn_mod's own qkv/proj weights — the exact
    computation AltAttention must reduce to when all flags are off."""
    B, N, C = x.shape
    h, d = attn_mod.num_heads, C // attn_mod.num_heads
    qkv = attn_mod.qkv(x).reshape(B, N, 3, h, d).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    attn = (q * (d ** -0.5)) @ k.transpose(-2, -1)
    attn = attn.softmax(dim=-1)
    out = (attn @ v).transpose(1, 2).reshape(B, N, C)
    return attn_mod.proj(out)


def test_default_off_is_plain_attention():
    torch.manual_seed(0)
    a = M.AltAttention(DIM, num_heads=HEADS, qkv_bias=True).to(DEV).eval()
    assert a.gate_proj is None and not a.use_rope and not a.cosine_attention
    x = torch.randn(B, N, DIM, device=DEV)
    with torch.no_grad():
        got, ref = a(x), _ref_sdpa(a, x)
    assert torch.allclose(got, ref, atol=1e-5), (got - ref).abs().max().item()


def test_gated_mlp_variants_and_parity():
    from timm.models.vision_transformer import Mlp
    plain = sum(p.numel() for p in Mlp(in_features=DIM, hidden_features=int(DIM * 4.0)).parameters())
    for var in ("swiglu", "geglu", "reglu"):
        g = M.GatedMlp(in_features=DIM, hidden_features=int(DIM * 4.0), variant=var).to(DEV)
        y = g(torch.randn(B, N, DIM, device=DEV))
        assert y.shape == (B, N, DIM) and torch.isfinite(y).all()
        assert 0.9 < sum(p.numel() for p in g.parameters()) / plain < 1.1
    try:
        M.GatedMlp(DIM, DIM, variant="nope"); assert False
    except ValueError:
        pass


def test_rope_properties():
    hd = DIM // HEADS
    cos, sin = M._rope_cos_sin(torch.arange(N, device=DEV), hd, 10000.0, torch.float32)
    x = torch.randn(B, HEADS, N, hd, device=DEV)
    rx = M._apply_rope(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), rx.norm(dim=-1), atol=1e-5)      # norm preserving
    assert torch.allclose(rx[:, :, 0], x[:, :, 0], atol=1e-6)             # identity at pos 0
    qv = torch.randn(hd, device=DEV); kv = torch.randn(hd, device=DEV)
    q = qv.view(1, 1, 1, hd).expand(1, 1, N, hd).contiguous()
    k = kv.view(1, 1, 1, hd).expand(1, 1, N, hd).contiguous()
    dots = (M._apply_rope(q, cos, sin) @ M._apply_rope(k, cos, sin).transpose(-2, -1))[0, 0]
    for off in range(-4, 5):                                              # relative-position only
        vals = [dots[m, m - off].item() for m in range(N) if 0 <= m - off < N]
        assert max(vals) - min(vals) < 1e-3


def test_rope_mask_invariance():
    """The masking fix: a kept token must rotate by its TRUE original index, not its compressed
    post-mask slot. So cos/sin for a gathered subsequence equals cos/sin of the full sequence
    indexed at those positions, and per-sample (B,N) positions apply correctly through _apply_rope."""
    hd = DIM // HEADS
    full = torch.arange(16, device=DEV)
    kept = torch.tensor([0, 3, 7, 9, 15], device=DEV)            # positions surviving a random mask
    cf, sf = M._rope_cos_sin(full, hd, 10000.0, torch.float32)
    ck, sk = M._rope_cos_sin(kept, hd, 10000.0, torch.float32)
    assert torch.allclose(ck, cf[kept], atol=1e-6) and torch.allclose(sk, sf[kept], atol=1e-6)
    x = torch.randn(2, HEADS, kept.numel(), hd, device=DEV)
    pos_b = kept.unsqueeze(0).expand(2, -1)                      # (B, N) per-sample positions
    cb, sb = M._rope_cos_sin(pos_b, hd, 10000.0, torch.float32)
    rx = M._apply_rope(x, cb, sb)
    assert rx.shape == x.shape and torch.allclose(x.norm(dim=-1), rx.norm(dim=-1), atol=1e-5)


def test_altattention_position_ids_wiring():
    torch.manual_seed(0)
    x = torch.randn(B, N, DIM, device=DEV)
    # rope on: explicit contiguous positions == None (both are the true positions of a full seq)
    a = M.AltAttention(DIM, num_heads=HEADS, qkv_bias=True, use_rope=True).to(DEV).eval()
    with torch.no_grad():
        assert torch.allclose(a(x, position_ids=None),
                              a(x, position_ids=torch.arange(N, device=DEV)), atol=1e-6)
    # default-off: position_ids must be completely ignored (bit-identical)
    b = M.AltAttention(DIM, num_heads=HEADS, qkv_bias=True).to(DEV).eval()
    with torch.no_grad():
        assert torch.allclose(b(x), b(x, position_ids=torch.arange(N, device=DEV)), atol=0)


def test_attn_output_gate_forward_and_init():
    for mode in ("headwise", "elementwise"):
        a = M.AltAttention(DIM, num_heads=HEADS, qkv_bias=True, attn_output_gate=mode).to(DEV).eval()
        y = a(torch.randn(B, N, DIM, device=DEV))
        assert y.shape == (B, N, DIM) and torch.isfinite(y).all()
        assert abs(torch.sigmoid(a.gate_proj.bias).mean().item() - 0.8808) < 1e-3  # pass-through init


def test_cosine_attention_no_device_mismatch():
    a = M.AltAttention(DIM, num_heads=HEADS, qkv_bias=True, cosine_attention=True).to(DEV).eval()
    y = a(torch.randn(B, N, DIM, device=DEV))   # must not raise a CPU/CUDA clamp mismatch
    assert torch.isfinite(y).all()


def test_frontend_glu_gate_identity_and_trainable():
    fg = FrontendGLUGate(DIM, kernel_size=5).to(DEV).train()
    x = torch.randn(B, DIM, N, device=DEV)
    with torch.no_grad():
        assert torch.allclose(fg(x), x, atol=1e-6)              # exact identity at init
    # ... but NOT frozen: a few steps must move every parameter off its init.
    before = {n: p.detach().clone() for n, p in fg.named_parameters()}
    opt = torch.optim.AdamW(fg.parameters(), lr=1e-2)
    for _ in range(4):
        opt.zero_grad()
        loss = (fg(torch.randn(B, DIM, N, device=DEV)) ** 2).mean()
        loss.backward()
        assert fg.depthwise.weight.grad.abs().sum() > 0        # depthwise wakes immediately
        opt.step()
    for n, p in fg.named_parameters():
        assert not torch.allclose(p.detach(), before[n]), f"{n} never moved (frozen)"


def test_all_flags_together_forward_backward():
    blk = M.AltBlock(DIM, HEADS, mlp_ratio=4.0, qkv_bias=True, cosine_attention=True,
                     attn_output_gate="elementwise", gated_mlp=True, gated_mlp_variant="swiglu",
                     use_rope=True, rope_base=10000.0).to(DEV).train()
    y, _ = blk(torch.randn(B, N, DIM, device=DEV))
    assert y.shape == (B, N, DIM) and torch.isfinite(y).all()
    (y ** 2).mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in blk.parameters() if p.requires_grad)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    n_fail = 0
    for fn in fns:
        try:
            fn(); print(f"[PASS] {fn.__name__}")
        except Exception as e:
            n_fail += 1; print(f"[FAIL] {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - n_fail}/{len(fns)} passed")
    raise SystemExit(1 if n_fail else 0)
