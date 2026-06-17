# animal2vec checkpoint validation (turnkey)

Drop in an animal2vec (`data2vec_multi`) pretraining checkpoint → get a frozen-probe validation on
**modern torch 2.9 + GPU** (the legacy torch-1.13 fairseq stack is CPU-only on the RTX 5090). Loads
**losslessly by weights**: sanitizes Anvar-fork config keys (`multi_corpus_keys`, …), sets `skip_ema`,
drops the EMA teacher, and patches version-mismatched fairseq fns. Input = 10 s @ 8 kHz (80000) or native.
Memory-safe: slim 1.3 GB checkpoint, 10 s input cap, **one checkpoint per process**, RAM watchdog.

## 👉 Start here: `SUMMARY.md`
One-page overview of all three checks with the cross-task table (encoder vs log-mel/AVES) and the verdict.
`VERDICT.md` has the full calibration; `SHAP_A2V.md` / `FILTER_A2V.md` / `DYNAMICS.md` are the per-task detail.

## Scripts
**Core loader / probe**
- `a2v_extract.py` — checkpoint + audio → mean-pooled encoder embeddings (`.npz`). GPU. (all others import it)
- `a2v_layer_sweep.py` — per-transformer-layer frozen-probe sweep (final layer is usually worst).
- `a2v_validate.py` — embeddings → frozen LogReg probe (recording-disjoint) + clustering (k-NN purity,
  silhouette, KMeans NMI/ARI, t-SNE).

**Validation tasks** (each is one command; calibration scripts score AVES-8k + log-mel-8k on the same data)
- `a2v_watkins.py` — Watkins 31-way species + clustering (the fair 8 kHz test). Calib: `watkins_baselines.py`.
- `a2v_filter.py` — binary signal/noise **filtration** probe. Calib: `filter_baselines.py`, plot: `filter_plot.py`.
- `a2v_shap.py` — frequency-band **SHAP** (occlusion) attribution; `a2v_shap_replot.py` re-renders from JSON.
- `a2v_dynamics.py` + `run_dynamics.sh` — **dynamics across runs/steps** (compare blue/orange, decide which
  to keep). See `DYNAMICS.md`.

## Usage
```bash
PY=~/a2v_env/bin/python          # torch 2.9 + cu128 + fairseq 0.12.2 + shims
cd ~/a2v                         # the animal2vec repo (registers data2vec_multi)
$PY a2v_watkins.py  <ckpt>       # species + clustering   (+ $PY watkins_baselines.py for AVES/log-mel)
$PY a2v_filter.py   <ckpt>       # signal/noise           (+ $PY filter_baselines.py)
$PY a2v_shap.py     <ckpt>       # frequency-band attribution
# compare runs across steps (memory-safe, one process per checkpoint, RAM watchdog):
bash run_dynamics.sh "blue:30000:/path/blue_30k.pt" "orange:30000:/path/orange_30k.pt"
```

## ✅ Verdict (current checkpoint 25k)
Encoder **works and is learning** but **undertrained at ~8 %**: Watkins species **13.5k 0.378 → 25k 0.542
(+0.164, rising)**; filtration **0.68 → 0.73 (rising)**; both still below an 8 kHz log-mel baseline. K-class
(0.24, flat) is a poor proxy — 8 kHz kills the >4 kHz orca call detail. → keep training; headline on
16 kHz / species. SHAP shows attention still in the lowest band (0–0.5 kHz), the fingerprint of undertraining.
