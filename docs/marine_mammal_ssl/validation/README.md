# animal2vec checkpoint validation (turnkey)

Drop in an animal2vec (`data2vec_multi`) pretraining checkpoint → frozen-probe validation on **modern
torch 2.9 + GPU** (the legacy torch-1.13 fairseq stack is CPU-only on the RTX 5090). Loads **losslessly by
weights**: sanitizes fork config keys (`multi_corpus_keys`, …), sets `skip_ema`, drops the EMA teacher, and
patches version-mismatched fairseq fns. Input = 10 s @ 8 kHz (80000) or native. Memory-safe: slim 1.3 GB
checkpoint, 10 s input cap, **one checkpoint per process**, RAM watchdog.

## Scripts
Core: `a2v_extract.py` (checkpoint + audio → mean-pooled embeddings; imported by the rest) ·
`a2v_layer_sweep.py` (per-layer probe sweep) · `a2v_validate.py` (probe + clustering).

Tasks — one command each; the `*_baselines.py` score AVES-8k + log-mel-8k on the same data for calibration:
- `a2v_watkins.py` — Watkins 31-way species + clustering (the fair 8 kHz test). Calib `watkins_baselines.py`.
- `a2v_filter.py` — binary signal/noise filtration. Calib `filter_baselines.py`, plot `filter_plot.py`.
- `a2v_shap.py` — frequency-band SHAP (occlusion) attribution; `a2v_shap_replot.py` re-renders from JSON.
- `a2v_dynamics.py` + `run_dynamics.sh` — compare runs/checkpoints across training steps.

## Usage
```bash
PY=~/a2v_env/bin/python          # torch 2.9 + cu128 + fairseq 0.12.2 + shims
cd ~/a2v                         # the animal2vec repo (registers data2vec_multi)
$PY a2v_watkins.py <ckpt>        # species + clustering   (+ $PY watkins_baselines.py)
$PY a2v_filter.py  <ckpt>        # signal/noise           (+ $PY filter_baselines.py)
$PY a2v_shap.py    <ckpt>        # frequency-band attribution
bash run_dynamics.sh "blue:30000:/path/blue_30k.pt" "orange:30000:/path/orange_30k.pt"   # compare runs
```
Each script writes its own JSON/PNG next to itself — regenerated per run, not checked in.

## Verdict (checkpoint 25k)
Encoder works and is **learning** but **undertrained at ~8 %**: Watkins species 13.5k 0.378 → 25k 0.542
(+0.164, rising); filtration 0.68 → 0.73 (rising); both still below an 8 kHz log-mel baseline. K-class
(0.24, flat) is a poor proxy — 8 kHz removes the >4 kHz orca call detail. → keep training; headline on
16 kHz / species. SHAP: attention is still in the lowest band (0–0.5 kHz), the fingerprint of undertraining.
