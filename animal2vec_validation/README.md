# animal2vec validation + SHAP

Frozen-probe validation harness and frequency-band attribution ("SHAP-style") analysis for
animal2vec checkpoints, ported from the original research environment. Companion to
`docs/marine_mammal_ssl/validation/validate.py` (checkpoint slimming/retention) — this directory
focuses on **probing what a checkpoint's embeddings are actually good for**, calibrated against
AVES (frozen, off-the-shelf, no marine pretraining) and a trivial log-mel-spectrogram baseline.

## TL;DR findings

- animal2vec genuinely learns transferable marine-audio representations early in training
  (Watkins 31-way species macro-F1 rising 0.38 → 0.54 over ~11.5k updates), but at ~8% of the
  planned schedule its frozen features are still **below** a trivial 8kHz log-mel + linear-probe
  baseline on both species classification and signal-vs-noise filtration. AVES (zero marine
  pretraining) currently beats animal2vec by a wide margin on the same tasks. Read this as
  **undertrained, not architecturally broken** — see the `feature/architecture-gated-rope`
  branch/PR for follow-up architecture experiments motivated partly by this read.
- Frequency-band occlusion attribution (`a2v_shap.py`) shows the encoder currently relies almost
  entirely on the lowest 0–0.5kHz band and largely ignores the 1–4kHz band where most K-call
  energy actually sits — consistent with "undertrained", since a stronger encoder should track the
  call bands more closely (correlate with `attr_vs_energy_pearson` in `a2v_shap.json`).

## Real-checkpoint results (added after porting, run against 2 production checkpoints)

Ran the full suite (filtration, Watkins species, layer sweep, SHAP attribution) against two real
checkpoints supplied by the team: Anvar's `resume13581_lr1e4_checkpoint4` (8kHz, the "gold"
checkpoint) and `a2v16khz5s_lr1e4_epoch1_dataset_r2_checkpoint_1_27000` (16kHz, step 27000).

| checkpoint | rate | filt macro-F1 | filt AUC | Watkins macro-F1 | knn-pur | NMI |
|---|---:|---:|---:|---:|---:|---:|
| `resume13581_checkpoint4` | 8kHz | 0.788 (L12) | 0.864 | 0.700 (L4) | 0.408 | 0.447 |
| `a2v16khz5s_checkpoint_1_27000` | 16kHz | 0.801 (L8) | 0.880 | 0.682 (L2) | 0.421 | 0.476 |

Both are well above the original `ckpt13k`/`ckpt25k` reference points (filt-F1 0.681/0.734,
Watkins-F1 0.378/0.542) and, notably, the 16kHz checkpoint now reaches **101% of the log-mel-8k
Watkins baseline** — the first checkpoint in this series to match/beat that bar rather than fall
short of it.

K-class (12-way call-type) layer sweep on `resume13581_checkpoint4`: best L12 = 0.318 vs final =
0.308 — still well below the log-mel/AVES bars (0.654/0.874-0.894) on this harder, fine-grained
task, consistent with the "good species ID, weaker on full call-type discrimination" pattern.

SHAP attribution on `resume13581_checkpoint4` (K-class, L1, 12-way probe F1=0.299): mean band
importance peaks at 0–0.5kHz (+0.132) and 1–1.5kHz (+0.102); attribution-vs-call-energy Pearson
r=0.150 (still low — the encoder isn't strongly tracking where the call energy actually sits, even
on this much-more-trained checkpoint).

**Bug found + fixed while running this:** `a2v_extract.py` (and `a2v_watkins.py`'s independent
audio-prep path) hardcoded `SR = 8000` for *all* checkpoints. The 16kHz checkpoint's SincNet
frontend kernels are rate-specific — feeding it 8kHz-resampled audio silently halves its usable
bandwidth and *understates* its real performance (confirmed by re-running before/after: filt-F1
0.762→0.801, Watkins-F1 0.582→0.682, both jumping ~10pt once fed the correct rate). Fixed by
porting the same auto-detect-from-checkpoint-cfg approach already used in
`docs/marine_mammal_ssl/validation/validate.py` (`hotfix/validate-16khz`, PR #6) — `a2v_extract.py`
now sets the global `SR` from the checkpoint's own `task.sample_rate` / `model.sample_rate` /
`model.modalities.audio.sample_rate` before any audio is loaded, and `a2v_watkins.py`'s `prep()`
now follows the same detected rate instead of its own separate hardcoded 8000. All numbers in the
table above are post-fix.

**Rate-awareness (fixed):** `a2v_shap.py` now re-derives its `SR`/`BANDS` from the rate
`a2v_extract._detect_set_sr()` reads off the checkpoint cfg, so a 16kHz checkpoint gets the full
0..8kHz grid (16 bands) with correctly labeled axes instead of a mislabeled 0..4kHz half.
`a2v_layer_sweep.py` and `a2v_filter.py` were never rate-bound — they only use `E.load_audio()`,
which already follows the detected rate, and carry no band logic of their own.

**Deliberately-still-8kHz:** `filter_baselines.py`'s AVES/log-mel baselines bandlimit to 8kHz *by
design* — the point is a like-for-like comparison against an 8kHz-Nyquist animal2vec, hence the
`AVES-8k` / `log-mel-8k` names. When calibrating against a 16kHz checkpoint, either widen those
baselines to the checkpoint's rate (and rename) or read the `-8k` numbers as a conservative
lower-Nyquist reference.

## Setup

These scripts assume the repo's `animal2vec` package is importable (run from the repo root, or
with the repo root on `PYTHONPATH` — `a2v_extract.py` adds it automatically via `__file__`).
Extra deps beyond the base `pyproject.toml`: `scipy`, `scikit-learn`, `librosa`, `soundfile`,
`matplotlib`, `datasets` (HuggingFace, for the Watkins species task), `torchaudio` (for the AVES
baseline). A few scripts read **environment-specific paths** from env vars with the original
research-environment defaults as fallback — set these for your own checkout:

| env var | used by | default (original environment) |
|---|---|---|
| `A2V_LAB_DIR` | `a2v_filter.py`, `a2v_shap.py`, `a2v_layer_sweep.py` | K-class labeled clips dir |
| `A2V_WATKINS_ARROW_DIR` | `a2v_watkins.py`, `watkins_baselines.py` | HF `beans_watkins` arrow cache dir |
| `A2V_AVES_WEIGHTS` | `filter_baselines.py`, `watkins_baselines.py` | path prefix to AVES `.torchaudio.pt` + `.model_config.json` |

## Scripts

**Core extraction (everything else builds on this):**
- `a2v_extract.py` — loads + sanitizes a raw animal2vec checkpoint (strips fork-specific config
  keys like `multi_corpus_keys` so the model still loads losslessly), extracts mean-pooled
  per-layer embeddings. `sanitize_and_save()` / `load_model()` are reused by every script below.

**Frozen-probe validation:**
- `a2v_validate.py` — standalone, decoupled CLI: takes a precomputed `.npz` of embeddings
  (`X[, y, groups, classes]`) and runs a recording-disjoint GroupKFold LogReg probe (macro-F1 +
  bootstrap 95% CI + per-class F1) plus clustering diagnostics (k-NN purity, silhouette, KMeans
  NMI/ARI) and a t-SNE figure. No GPU / animal2vec dependency — works on embeddings from anything.
- `a2v_watkins.py` — 31-way Watkins marine-mammal species classification (the fair, 8kHz-native
  task), with clustering diagnostics. Accumulates results across runs into
  `animal2vec_watkins.json` for the dynamics view.
- `a2v_filter.py` — binary signal-vs-noise filtration probe (any K-call vs the noise class),
  recording-disjoint GroupKFold. Dumps `filter_manifest.json` (the exact clip/group split) so
  `filter_baselines.py` can score AVES/log-mel on the **same** clips for a fair comparison.
  Accumulates into `a2v_filter.json`.
- `a2v_layer_sweep.py` — per-layer frozen-probe sweep on the K-class task (the final transformer
  layer is usually the *worst* one for SSL probes; finds which layer is actually best).
- `watkins_baselines.py` / `filter_baselines.py` — AVES + log-mel calibration baselines for the
  two tasks above, scored on the exact same data/splits, standalone (no `animal2vec.nn` import).

**Attribution / interpretability:**
- `a2v_shap.py` — frequency-band **occlusion** attribution: bandstop-removes each 0.5kHz band
  (rate auto-detected from the checkpoint), measures the drop in the frozen probe's predicted prob
  for the true class. Fast proxy; single-band removal under-states importance on a redundant
  encoder — prefer `a2v_shap_tb.py` for the faithful version.
- `a2v_shap_tb.py` — **proper Shapley values** over the frequency bands (each 0.5kHz band is a
  coalition player; a coalition is rendered by summing its band-pass components). Exact enumeration
  when #bands ≤ 10, permutation sampling otherwise. Logs the figure + per-band scalars to
  **TensorBoard** (`tb_shap/<ckpt>/`) and writes one PNG/JSON per checkpoint. Run one checkpoint at
  a time.
- `a2v_shap_replot.py` — re-renders a saved SHAP figure with corrected frequency-band axis labels
  (pure plotting, reads the SHAP JSON, no GPU).

**Trend / dynamics across training (compare parallel runs):**
- `a2v_dynamics.py` — pure plotter; reads the accumulated probe JSONs + `dynamics_registry.json`
  (checkpoint tag → {run, step}) and plots both probes' trajectories per run, plus a
  "% of log-mel-8k baseline reached" comparable view.
- `run_dynamics.sh` — turnkey driver: runs both probes on a batch of checkpoints (one process at a
  time, RAM-watchdog'd), registers them, regenerates the dynamics plot.
- `filter_plot.py` — bar-chart plotter for the filtration calibration (animal2vec vs log-mel vs
  AVES at different checkpoint steps), reads `a2v_filter.json` + `filter_baselines.json`.

## Outputs

This PR ships the **tool only** — the scripts write their own results (`*.json`, `*.png`,
TensorBoard logs under `tb_shap/`) into this directory when run; those are not committed. The
headline numbers are in the paper / PR discussion, and every figure and metric here is
reproducible by running the scripts against a checkpoint.
