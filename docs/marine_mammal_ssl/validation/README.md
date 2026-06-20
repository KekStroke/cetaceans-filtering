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

$PY $V slim    <raw.pt> [--out S.pt]                                           # raw ~5GB ckpt → ~1.3GB inference ckpt
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

## ⚠️ Extract features with `mask=False`
data2vec masks ~93 % of the input by default (`mask=True`); extracting probe features that way collapses
them. `validate.py` calls the encoder with `mask=False` (clean, unmasked, **deterministic**) — the
canonical frozen-probe path. An earlier run that left masking on produced a spuriously "undertrained" read;
the numbers below are the corrected, mask-off ones.

## Verdict (checkpoints 13.5k & 25k, clean features)
The encoder is **strong already at ~8 % of training**: Watkins species (31-way) **13.5k 0.795 → 25k 0.839**
(rising) — far above the 8 kHz log-mel baseline (0.675) and **matching AVES-8k (0.853)**, a domain-trained
SSL reference. Signal/noise filtration **0.81** (near-ceiling for every encoder; just under log-mel 0.90).
SHAP attribution **tracks the call energy** (attr-vs-energy r ≈ 0.73). → healthy run; keep training and
re-probe later checkpoints; for the headline evaluate on 16 kHz / species where 8 kHz isn't the limiter.
