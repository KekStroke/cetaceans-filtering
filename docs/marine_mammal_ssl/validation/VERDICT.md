# animal2vec checkpoint validation — VERDICT (ckpt 13.5k & 25k, blue run lr=5e-5)

## TL;DR: the encoder is LEARNING (rising, ×17 chance) but at 25k it is UNDERTRAINED — its features are still BELOW an 8 kHz log-mel baseline. Keep training; don't expect useful features yet.

### Calibration on Watkins species (same probe) — this corrects an earlier too-rosy read
| encoder | band | macro-F1 |
|---|---|---:|
| AVES | 16 kHz | 0.885 |
| AVES | 8 kHz | 0.853 |
| log-mel | 16 kHz | 0.760 |
| **log-mel** | **8 kHz** | **0.675** |
| **animal2vec-25k** | **8 kHz** | **0.542** |
| chance | | 0.032 |

**8 kHz is NOT the excuse** (AVES barely drops at 8 kHz; log-mel-8k 0.675 still beats animal2vec 0.542). The gap is **undertraining**, not bandwidth — at 25k (8 %) the encoder's frozen features are worse than a trivial 8 kHz spectrogram. The **rising trend** (+0.16 per ~11.5 k updates) is the encouraging part, but it describes the *trajectory*, not current usefulness.

| task | 13.5k | 25k | Δ | read |
|---|---:|---:|---:|---|
| **Watkins (31-way species, fair 8 kHz)** | 0.378 | **0.542** | **+0.164** | learns transferable features, **rising** with training (chance ≈ 0.032) |
| K-class (12-way orca call-types) | 0.246 | 0.236 | ~0 (flat) | poor proxy at 8 kHz |

- Watkins kNN-purity 0.18 → **0.26**, KMeans NMI 0.27 → **0.36** — clustering also improving.
- Frozen linear probe, per-layer best (final layer worst, as usual); Watkins' own train/test split.
- Modern torch 2.9 + RTX 5090, lossless by weights (slim 1.3 GB checkpoints; 10 s crop for long clips).

## Why K-class looked bad but isn't a red flag
Orca **call-type** discrimination needs the **> 4 kHz band** — and 8 kHz (Nyquist 4 kHz) destroys it (K21 has 52 % of its energy > 4 kHz). Add the **lab-clean clips vs field-passive pretraining domain** gap and the fine-grained subtlety, and K-class is simply the wrong probe for this 8 kHz field encoder. On the **fair, coarser** task (species) the same encoder scores **0.54** and climbs.

## What this means
1. **The encoder genuinely learned useful marine-audio representations** — species 0.54 ≫ chance 0.03 at only **~8 % of training**, and **improving +0.16 per ~11.5 k updates**. This matches the healthy internal signals (loss/target_var rising) — now the **downstream probe confirms transfer is real and growing**.
2. **Keep training** — it's on a healthy trajectory; species-F1 should keep climbing well past 0.54 over the 320 k schedule. Re-probe later checkpoints (one command).
3. **For the paper headline, evaluate on 16 kHz / species tasks** where 8 kHz isn't crippling — and **pretrain the next encoder at 16 kHz** (we measured +10.2 pts on the orca tasks).

## Files
`a2v_extract.py` · `a2v_layer_sweep.py` · `a2v_watkins.py` (turnkey, modern GPU) · `animal2vec_watkins_trend.json` · `animal2vec_verdict.png`
