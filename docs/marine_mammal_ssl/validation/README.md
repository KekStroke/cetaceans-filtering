# animal2vec checkpoint validation (turnkey)

A single `validate.py` — drop in a `data2vec_multi` pretraining checkpoint, get a frozen-probe validation on
modern torch 2.x + GPU. Loads **losslessly by weights** (sanitizes fork cfg keys like `multi_corpus_keys`,
sets `skip_ema`, drops the EMA teacher, patches version-mismatched fairseq fns). Memory-safe: slim 1.3 GB
checkpoint, 10 s/8 kHz input cap, **one model per process**.

## Usage
```bash
cd "$A2V_REPO"                      # the animal2vec repo (registers data2vec_multi); default ~/a2v
PY=~/a2v_env/bin/python             # torch 2.x + cu12 + fairseq 0.12.2 + the shims
V=/path/to/validate.py

$PY $V watkins <ckpt> [--run blue --step 30000] [--no-baselines] [--limit N]   # 31-way species + clustering
$PY $V filter  <ckpt> [--run blue --step 30000] [--no-baselines] [--limit N]   # binary signal/noise
$PY $V shap    <ckpt> [--limit N]                                              # frequency-band attribution → PNG
$PY $V kclass  <ckpt> [--n-per-class N]                                        # per-layer probe on K-class
$PY $V dynamics                                                                # plot accumulated runs/steps
```
- `watkins`/`filter` accumulate results and, with `--run/--step`, register the checkpoint — then `dynamics`
  plots the comparison across runs/steps (decide which parallel run to keep). One model per process, so to
  compare runs just call `watkins`/`filter` once per checkpoint, then `dynamics`.
- `--no-baselines` skips the AVES-8k / log-mel-8k calibration; `--limit N` caps clips for a fast smoke run.
- Outputs (JSON + PNG) go to `$A2V_OUT` (default `./a2v_val_results`) — regenerated per run, **not** checked in.
- **Memory:** run **one checkpoint at a time** (each loads ~1.3 GB). To sweep several, call `watkins`/`filter`
  per checkpoint sequentially, then `dynamics` — don't run them concurrently. On a tight box wrap the loop
  with a `free -m`/`pkill` watchdog.

## Config (env vars — point these at your data/weights)
| var | default | meaning |
|---|---|---|
| `A2V_REPO` | `~/a2v` | animal2vec repo (for `import nn`) |
| `A2V_OUT` | `./a2v_val_results` | where JSON/PNG outputs go |
| `A2V_KCLASS` | `data/kclass_wavs` | dir of labelled K-class `.wav` clips |
| `A2V_WATKINS` | `data/beans_watkins` | BEANS Watkins arrow dir (train/test) |
| `A2V_AVES` | `weights/aves-base-bio.torchaudio` | AVES torchaudio weights prefix (`.pt`/`.json`) |

## Verdict (checkpoint 25k)
Encoder works and is **learning** but **undertrained at ~8 %**: Watkins species 13.5k 0.378 → 25k 0.542
(+0.164, rising); filtration 0.68 → 0.73 (rising); both still below an 8 kHz log-mel baseline. K-class
(0.24, flat) is a poor proxy — 8 kHz removes the >4 kHz orca call detail. SHAP: attention is still in the
lowest band (0–0.5 kHz), the fingerprint of undertraining. → keep training; headline on 16 kHz / species.
