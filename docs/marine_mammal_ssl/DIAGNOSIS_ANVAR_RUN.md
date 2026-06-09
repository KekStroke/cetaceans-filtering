# Diagnosis — Anvar's animal2vec pretraining run (`data2vec_multi`)

*From `hydra.yaml` / `config.yaml` / `animal2vec_train.log` / TensorBoard, run 2026-05-13→15, ~4,224 updates. Config = **stock default** (`overrides.yaml` is empty `[]`). Figure: `anvar_run_diagnosis.png`.*

## ✅ Verdict: THE MODEL IS TRAINING — the "huge loss" is a non-problem
- **loss 11.8 → 4.92** (min 3.26), updates 343→4271 — monotone down.
- **target_var 0.352 → 0.505** (rising, healthy) → **no representation collapse** (collapse would drive it → 0).
- **pred_var ~0.30** (one brief dip to 0.10, recovered).
- The data2vec2 latent-MSE is uninterpretable by design; judged by **trend + target_var**, this is a healthy run. **Stop worrying about the magnitude.**

## 🔴 Fix #1 (CRITICAL) — fp16 loss-scale collapse
- `common.fp16: true` + `fp16_init_scale: 1`. The log shows a cascade: gradient overflow → loss scale `1 → 0.5 → … → 0.0001`, and it stays **0.0001–0.0156** the whole run (**49 overflow events**). Normal is ~128. Gradients are barely representable; many steps drop their gradient entirely.
- **Fix: bf16** — `common.bf16: true`, `common.fp16: false`. GPU has ~47 GB free (looks like A100-80) → bf16 supported, no loss-scaling, overflow gone, usually faster too.
- (If bf16 is genuinely unavailable: at minimum `fp16_init_scale: 128`.)

## 🟠 Fix #2 (SPEED — the run is ~2 months at this rate) — drop `update_freq`
- `ups = 0.02` → **1 update / 50 s**; at `max_update: 100000` that is **~58 days**. In ~52 h it reached only **4,224** updates.
- Effective batch = `max_tokens 408000 × update_freq 80 × clone_batch 12 × 2 GPU` = MeerKAT-scale — far larger than this marine corpus needs.
- **Fix: cut `update_freq` 80 → 8–16** → ~5–10× more updates/hour, and the 10,000-step warmup finishes in sane wall-clock (at update 4224 the LR is still only 1.27e-5 of the 3e-5 target — 42 % through warmup).
- Add GPUs if available (data2vec normally uses 8–16).

## 🟠 Fix #3 — 1-GPU OOM
- Cause: `clone_batch: 12` (12 masked copies/sample) × `max_tokens: 408000`. To fit one GPU: `clone_batch` 12 → 6–8, or `max_tokens` 408000 → ~200000 (compensate via `update_freq`). The 2-GPU workaround is fine.

## 🟡 Fix #4 — sample rate 8 kHz → 16 kHz (next corpus)
- `sample_rate: 8000`, data `cetaceans_8khz_10s`. We measured **+10.2 pts** downstream at 16 kHz, and SHAP (r=0.867) shows 8 kHz removes the exact bands the encoder uses. Don't kill the current run for this — but **build the next corpus at 16 kHz** (best ROI).

## 🟡 Fix #5 — EMA anneal mismatch
- `ema_anneal_end_step: 300000` but `max_update: 100000` → the EMA decay never reaches its end value within the run. Set `ema_anneal_end_step ≈ max_update` (e.g. 80,000).

## Minor
- `masked_pct = 0.928` (mask_prob 1.5, length 2) — 92.8 % masked, aggressive (MeerKAT default for sparse events). OK for sparse marine calls; lower it if you want faster convergence.
- When a checkpoint exists, run our frozen-probe on it for an independent "did it learn useful features?" read beyond the loss.

## TL;DR priority
**1. bf16** (stability) · **2. `update_freq` 80→~12** (speed) · 3. `clone_batch`/`max_tokens` for 1-GPU · 4. 16 kHz next corpus · 5. `ema_anneal_end_step`.
