#!/usr/bin/env python3
"""Unit tests for the BEST-RQ objective module (animal2vec/nn/bestrq.py).

Self-contained, CPU-only. Covers the properties that make BEST-RQ correct: the quantizer is
FROZEN and DETERMINISTIC, the log-mel target view aligns to the encoder frame rate, only the
prediction head trains, and the masked cross-entropy is calibrated (starts at ~ln(codebook_size)).

    pytest animal2vec/tests/test_bestrq.py    # or: python animal2vec/tests/test_bestrq.py
"""
import math
import torch
import torch.nn.functional as F

from animal2vec.nn.bestrq import BestRQModule, RandomProjectionQuantizer, LogMelTargets

B, S, T, ED, K = 3, 16000, 100, 256, 8192


def test_logmel_aligns_to_frame_count():
    lm = LogMelTargets(sample_rate=16000, n_mels=80)
    out = lm(torch.randn(B, S), T)
    assert out.shape == (B, T, 80) and torch.isfinite(out).all()
    # works for a different target frame count too (adaptive pooling)
    assert lm(torch.randn(B, S), 37).shape == (B, 37, 80)


def test_quantizer_frozen_deterministic_in_range():
    q = RandomProjectionQuantizer(80, codebook_dim=16, codebook_size=K, seed=42)
    assert list(q.parameters()) == []                      # no trainable params (buffers only)
    x = torch.randn(B, T, 80)
    idx = q(x)
    assert idx.shape == (B, T) and idx.dtype == torch.long and 0 <= idx.min() and idx.max() < K
    # deterministic across fresh instances with the same seed; different across seeds
    assert torch.equal(RandomProjectionQuantizer(80, codebook_size=K, seed=42)(x), idx)
    assert not torch.equal(RandomProjectionQuantizer(80, codebook_size=K, seed=7)(x), idx)


def test_quantizer_targets_are_bf16_cast_safe():
    # model.to(bf16) casts the frozen buffers; forward must still work (upcasts internally)
    q = RandomProjectionQuantizer(80, codebook_size=K).to(torch.bfloat16)
    idx = q(torch.randn(B, T, 80))
    assert idx.shape == (B, T) and idx.max() < K


def test_module_only_head_trains_and_ce_is_calibrated():
    m = BestRQModule(encoder_dim=ED, sample_rate=16000, codebook_size=K, n_mels=80)
    trainable = [n for n, p in m.named_parameters() if p.requires_grad]
    assert trainable == ["head.weight", "head.bias"], trainable   # quantizer + log-mel are frozen
    src = torch.randn(B, S)
    tgt = m.targets(src, T)                                 # (B, T) long, no grad
    assert not tgt.requires_grad and tgt.shape == (B, T)
    h = torch.randn(B, T, ED, requires_grad=True)
    mask = torch.rand(B, T) < 0.5
    loss = F.cross_entropy(m.logits(h)[mask], tgt[mask])
    # random head -> near-uniform -> CE ~= ln(codebook_size); NOT the huge data2vec MSE
    assert abs(loss.item() - math.log(K)) < 0.5, (loss.item(), math.log(K))
    loss.backward()
    assert m.head.weight.grad is not None and torch.isfinite(m.head.weight.grad).all()


def test_targets_stable_under_head_change():
    # BEST-RQ's stability: targets depend only on the (frozen) quantizer + input, not the model
    m = BestRQModule(encoder_dim=ED, codebook_size=K, n_mels=80)
    src = torch.randn(B, S)
    t1 = m.targets(src, T)
    with torch.no_grad():
        m.head.weight.mul_(5.0)                            # perturb the trained part
    assert torch.equal(m.targets(src, T), t1)              # targets unchanged


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    nfail = 0
    for fn in fns:
        try:
            fn(); print(f"[PASS] {fn.__name__}")
        except Exception as e:
            nfail += 1; print(f"[FAIL] {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - nfail}/{len(fns)} passed")
    raise SystemExit(1 if nfail else 0)
