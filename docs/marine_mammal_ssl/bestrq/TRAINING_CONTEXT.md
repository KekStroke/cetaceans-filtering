# BEST-RQ — training context & logs (for Anvar)

A from-scratch BEST-RQ SSL encoder, built as a **fast, interpretable-loss** alternative to the
animal2vec / data2vec-2.0 run. Files here: `bestrq.py` (model), `bestrq_train.py` (training),
`train.log` (raw run log), `train_metrics.json` (parsed), `SMOKE.md` (smoke test). This doc is the
"training context" to reproduce or adapt it.

## Method (why the loss is readable)
- Target = **fixed Random-Projection Quantizer** (NO EMA teacher): project the clean log-mel →
  argmax-cosine into a frozen 8192-codebook → per-frame code.
- Encoder sees the **masked** mel and predicts the code at masked positions; loss = **masked
  cross-entropy over 8192** → **starts at ln(8192)=9.01** and is directly interpretable (unlike the
  data2vec latent-MSE that makes "the loss is huge" meaningless).
- No teacher forward → cheaper per step than animal2vec.

## Data
- **Train (unlabeled):** 10,000 clips from `new_training_data/` (orca calls + noise, **labels
  ignored**), 1.0 s @ 16 kHz, center-crop/pad. *Swap this for your corpus.*
- **Health-probe set (labeled):** 130/class K-class clips, recording-disjoint (GroupKFold-5) — used
  only to MEASURE transfer every 250 steps, never for training.

## Config
| group | values |
|---|---|
| front-end | log-mel 80, 16 kHz, n_fft 400, hop 160, per-utt norm |
| RPQ (frozen) | proj_dim 16, codebook 8192, seed 1234 |
| conv-subsample | 2× stride-2 (4× time downsample), conv_dim 256 |
| transformer | dim 384, 6 layers, 6 heads, ffn 1024, pre-norm — **14.0 M params** |
| masking | 8% span-starts × span 10 frames (mel-rate), one learned mask vector |
| optim | AdamW lr 5e-4 (warmup 120 + cosine), wd 0.01, betas (0.9, 0.98), grad-clip 1.0 |
| precision | **bf16 autocast (NOT fp16)** |
| schedule | 2000 steps target, batch 64, probe every 250 |

## Run log (cut at ~step 750 by a system reboot — trend fully established)
| step | loss | masked-acc | target-ppl | frozen-probe mF1 |
|---:|---:|---:|---:|---:|
| init (smoke) | ~9.19 | — | — | 0.642 (random ≈ log-mel) |
| 100 | 4.41 | 0.30 | 99 | — |
| 250 | — | — | — | 0.652 |
| 500 | 3.62 | 0.35 | 105 | 0.690 |
| 700 / 750 | 3.88 | 0.31 | 133 | 0.696 |

- **Loss interpretable & falling** (9.19 → 3.88).
- **target perplexity ~100–157** (codebook 8192) → no target collapse.
- **frozen probe 0.642 → 0.696** — the encoder learns beyond the fixed log-mel front-end (random-init
  already ≈ log-mel because the front-end is a real mel transform).
- **Throughput ~1455 clips/s** (1 s audio each, bf16, batch 64) on a *laptop* RTX 5090.

## Reproduce / point at YOUR corpus
1. `bestrq.py` + `bestrq_train.py` are self-contained (torch + torchaudio + sklearn + soundfile/librosa).
2. In `bestrq_train.py`, replace the `new_training_data` glob + `load_audio` with your corpus loader
   (any 16 kHz wavs; labels ignored). Keep the rest.
3. `python bestrq_train.py`. Bump `STEPS` for a real run (this demo was cut at 750).

## The point for the roadmap
The **interpretable loss (from ln(codebook)) + a frozen-probe-every-N-steps** is the "is it training?"
answer *by construction*. Even if you keep animal2vec, **drop the same 3-signal logger + frozen probe
onto your run** — that ends the "loss is huge / did it train?" uncertainty without changing your method.
