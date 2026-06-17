# animal2vec — signal/noise FILTRATION probe (Anvar's "проверь фильтрацию")

**TL;DR:** the frozen encoder **can** separate call-present clips from ambient noise and is **improving
with training** (13k→25k: F1 0.68→0.73, AUC 0.74→0.81), but even on this easy binary task it is **still
below a trivial log-mel-8k filter** (F1 0.90, AUC 0.96). Same verdict as species: learning, undertrained.
As a filter *right now* a plain log-mel would do better; the encoder's value is in its rising trajectory.

## Task
Binary **signal (any K-call) vs noise** on 500+500 balanced clips from `new_training_data`,
**recording-disjoint GroupKFold-5**, frozen linear probe, per-layer best. Calibrated against AVES-8k and
log-mel-8k on the **exact same clips/split** (`filter_manifest.json`). 8k = audio bandlimited to 8 kHz to
match animal2vec's Nyquist. Memory-safe (1 s clips, one model per process, watchdog).

## Results
| encoder | macro-F1 | bal-acc | ROC-AUC | best layer |
|---|---:|---:|---:|---|
| animal2vec 13k | 0.681 | 0.681 | 0.735 | L1 |
| **animal2vec 25k** | **0.734** | **0.734** | **0.807** | L13 |
| log-mel-8k | 0.903 | 0.903 | 0.958 | — |
| AVES-8k | 0.971 | 0.971 | 0.994 | — |
| chance | 0.500 | 0.500 | 0.500 | — |

- **Above chance and rising** (+0.05 F1 / +0.07 AUC over 13k→25k) — the encoder is learning a usable
  signal/noise boundary.
- **Below log-mel-8k by ~0.17 F1 / ~0.15 AUC** — undertrained; a handcrafted spectrogram filter beats it.
- **Best layer moves late (L13)** for filtration, vs early (L3) for species/call-type — coarse
  presence/absence lives in the deeper, more abstract layers (expected).

## Read
This is the *filter-fitness* view of the same undertraining story. It does **not** contradict "the encoder
is learning" — it quantifies *how far along*: at 25k (~8 % of schedule) it filters at AUC 0.81, a log-mel
does 0.96, AVES 0.99. Re-run on later checkpoints; the gap to log-mel should close, then to AVES.

**Caveat:** "noise" here is the K-project's ambient-noise class (same recordings), a proxy for the
filtering use case — not Anvar's dedicated ~15 h sound/noise filter set. Re-run on that set when available
(swap the manifest source; harness unchanged).

## Files
`a2v_filter.py` · `filter_baselines.py` · `filter_plot.py` · `a2v_filter.json` · `filter_baselines.json`
· `a2v_filter.png` · `filter_manifest.json`
