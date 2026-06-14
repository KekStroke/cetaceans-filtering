# animal2vec pretraining: BLUE vs ORANGE — health check & next-run plan

**TL;DR:** Both runs are **healthy**. The rising loss is **not divergence** — it is *driven by* rising `target_var` (the EMA teacher's targets get richer, so there is literally more variance to regress). Verified: after the loss minimum, `corr(loss, target_var) = 0.99` in both runs (Pearson 0.991, Spearman 0.997–0.99996). **Do not pick the min-loss checkpoint.** **ORANGE (lr = 1e-4, the paper's LR) is the run to keep and base the next one on.** Both are only ~3–4% into the 320k schedule and barely out of warmup.

---

## 1. Verdict — are both runs healthy? Is the rising loss a problem?

**Yes, both are healthy. No, the rising loss is not a problem.** I tried to refute the "healthy" hypothesis and could not — every divergence signature is absent and every health signal holds.

| Evidence | BLUE (lr 5e-5) | ORANGE (lr 1e-4) | Reading |
|---|---|---|---|
| Pearson(loss, target_var), post-loss-min | **0.9906** | **0.9908** | loss ≈ target_var × const |
| Spearman(loss, target_var), u>2000 | 0.997 | 0.99996 | monotone lockstep |
| OLS slope / R² (loss ~ tv) | +11.7 / 0.97 | +14.8 / 0.98 | loss is a linear readout of tv |
| loss/target_var ratio (mean, CV) | 7.68, 10.3% | 8.60, 15.5% | ratio nearly constant across the whole climb |
| max single-step \|Δloss\| after min | **0.066** | **0.094** | smooth, no jumps/discontinuities |
| target_var min (collapse guard) | 0.246 | 0.246 | never near 0 — **no collapse** |
| pred_var recovered from its min | 0.10 → 0.30 | 0.10 → 0.33 | predictor tracks teacher, rising |
| gnorm | stable 5–10, no spikes | stable 5–10, **declining** in tail | no instability |

The loss is essentially a constant multiple of `target_var`. A genuine divergence does **not** keep the loss locked to a slowly-rising teacher-variance signal, and does **not** rise *smoothly and proportionally*. This is exactly the data2vec-2.0 latent-MSE signature, and it is the opposite of the malign case in fairseq issue #4177 ("loss abruptly changes and stays high") — ours has no abrupt change (max step ≈ 0.07–0.09; median ≈ 0.02).

**Two non-fatal caveats to flag (real, neither is a reason to stop):**
- **`clip` pinned ~100%** from ~u600 onward. With `clip_norm=1.0` and gnorm 5–10, every gradient is rescaled to (near) unit norm, so the applied step ≈ `lr × (g/‖g‖)`. This is the **paper default**, but it means **lr literally equals the effective step size** — see §3/§4.
- **BLUE's gnorm creeps up** ~+1.05/1k updates (≈5.6→9.9, max 10.9). Mild, no spikes. **ORANGE's gnorm is flat/declining** (last-10 mean 5.94 < tail mean 8.8) — the cleaner run. The real instability tripwire is gnorm, *not* loss or clip% (clip masks gnorm growth): alert only on a sustained gnorm climb >~20.

---

## 2. The key insight — why loss rises, and why min-loss ≠ best checkpoint

**The loss target is a moving target that grows on purpose.** In data2vec-2.0 / animal2vec the loss is latent-MSE of the student against an **EMA teacher**. Early on the teacher is near-collapsed (low `target_var` ≈ 0.25, features nearly constant → easy to predict → low loss). As training de-collapses the teacher, `target_var` rises (BLUE 0.25→0.47, ORANGE 0.25→0.52, approaching the paper's ~0.5 end-regime). **Richer targets = more variance to regress = higher regression loss — even as the features get better.**

**The inflection ORDER proves the causation direction** (not just correlation):

| | pred_var min | target_var min | **loss min** |
|---|---|---|---|
| BLUE | u1650 | u1550 | **u2600 (1.796)** |
| ORANGE | u1200 | u1250 | **u2350 (1.686)** |

target_var turns up first; the loss keeps *falling* for ~1000 more updates, then turns and tracks target_var upward. That lag is the fingerprint of "teacher enriches → loss follows," i.e. the healthy mechanism — not instability (which would hit loss first/abruptly).

**Consequence for checkpoint selection:** the loss minimum (~u2350–2600) is the point of the *poorest, most-collapsed teacher*, not the best features. **Never select by `argmin(loss)`.** Select instead by, in priority order:
1. **Frozen-probe score (PRIMARY, decisive)** — freeze the student encoder, mean-pool last-layer features, train only a linear / shallow-MLP head on a labeled probe set, report balanced-acc / mAP. Pick `argmax(probe)`, tie-break to the later update. This is the only metric that directly answers "did it learn useful features?" and it stays roughly monotone while the loss rises.
2. **target_var trend** — must stay ≫0 (collapse guard) and ideally have reached the paper regime (~0.5) with its slope flattening (teacher maturing).
3. **pred_var tracking target_var** — healthy end-state is pred_var following tv up (ratio toward ~0.6–0.8), which both show (BLUE 0.64, ORANGE 0.62).
4. **gnorm sanity (veto only)** — reject checkpoints inside a gnorm-spike window. None here.

---

## 3. Blue vs Orange — which trajectory is better

**ORANGE is the better run to keep and to base the next one on** — but for a sharper reason than "higher target_var," and with one important nuance that resolves a disagreement between the analyses.

**At matched UPDATES, ORANGE leads** on target_var by a steady **+0.031 → +0.046**, and reaches BLUE's *final* target_var (0.469) by **update 8250 vs BLUE's 10800**:

| update | BLUE tv | ORANGE tv | Δ |
|---|---|---|---|
| 4000 | 0.304 | 0.335 | +0.031 |
| 6000 | 0.368 | 0.409 | +0.041 |
| 8000 | 0.418 | 0.464 | +0.045 |
| 8250 | 0.423 | **0.469** (= BLUE's final) | +0.046 |
| 10800 | 0.469 | 0.508 | +0.039 |

**Nuance (the decisive control): at matched LR, BLUE is actually AHEAD** — lr=5e-5: BLUE tv **0.457** vs ORANGE **0.374** (+0.083); lr=2.5e-5: BLUE 0.338 vs ORANGE 0.269 (+0.070). So **ORANGE is *not* more sample-efficient per gradient** — it is the *same trajectory traversed ~2× faster*, because clip=100% makes each step ≈ `lr × unit-vector`, and ORANGE's lr is 2× BLUE's, so its applied step is literally 2×. This refutes any "lr=1e-4 is intrinsically better per-step" claim, but it does **not** change the recommendation: in wall-clock / update terms ORANGE covers the same ground twice as fast with **no instability cost** (gnorm calm and *declining*, pred_var recovered, no NaN), and it is the one already **in the paper's target_var regime**.

**Maturity:** ORANGE's target_var slope has flattened to **+0.0045/1k** near **0.52** (≈ paper's ~0.5 end-regime) — decelerating, teacher maturing. BLUE is still climbing at **+0.017/1k** at only 0.47 — ~3–4k updates behind ORANGE at every milestone. **BLUE is fine, just slow.**

**Net:** keep ORANGE (lr=1e-4). It is in-regime, the first run plausibly worth probing, and the cleaner gnorm trajectory. Stopping it loses the better run. (Anvar stopped ORANGE — worth resuming it, or at minimum carrying its lr forward.)

---

## 4. Next-run config

Grounded in the verified facts: ORANGE beats BLUE purely via a larger effective step; clip=100% means lr ≈ step size; effective batch is still **below** the paper's ~1020 s/step; and there is ~57 GB VRAM free.

```yaml
# NEXT RUN — base on ORANGE
optimizer:
  lr: 1.0e-4                 # = the paper. ORANGE proved it's stable; 5e-5 just halves the step.
  lr_scheduler: cosine
  warmup_updates: 10000      # keep — both runs warmed cleanly over 10k
  max_update: 320000         # run the FULL schedule — we're only ~3-4% in and barely out of warmup
  clip_norm: 1.0             # = paper default. KEEP. (this is WHY lr == effective step; see below)
  weight_decay: 0.01         # paper

precision: bf16              # KEEP — fp16 overflow is fixed; bf16 has the range, no loss_scale bookkeeping

# --- grow effective batch toward the paper's ~1020 s/step using the free VRAM ---
clone_batch: 6               # 3 -> 6 (paper uses 12). Cheap: shares ONE encoder fwd, adds masked passes.
                             #   Go to 6, validate one step; if VRAM comfortable, push toward 12.
update_freq: 12              # 10 -> 12 (FREE — pure grad-accum, no VRAM cost)
# (optionally raise per-GPU batch if VRAM is still slack after the above)

ema:
  ema_decay: <paper schedule>  # keep paper's update-indexed EMA schedule (lr-independent)

# === STOP / SELECT on these, NOT on loss ===
checkpoint_policy:
  save_every: 2500           # frequent early to catch the probe-curve shape; can widen to 5000 once climbing
  select_by: frozen_probe    # argmax(probe), tie-break to later update — NEVER argmin(loss)
  probe_every: 2500          # early; then ~5000 once clearly climbing (probing ≪ pretraining cost)
monitoring:
  - target_var: must keep rising then plateau (~0.5-0.55 expected); always > 0
  - pred_var:   must stay > 0.05 and track target_var (ratio ~0.62 now — healthy)
  - gnorm:      stable; ALERT only if sustained > ~20 (clip hides it, so gnorm is the real tripwire)
  - frozen_probe: the ONLY "is it learning" truth — beat log-mel 0.654, approach/beat AVES 0.894
```

**Why these numbers**
- **lr 1e-4, not 5e-5** — under clip=100%, lr *is* the step size; 1e-4 is simply 2× the useful step with no observed downside (gnorm calm/declining, pred_var recovered, no NaN). It is the paper's value and already in the paper's tv regime.
- **clip_norm = 1.0, kept** — it's the paper default and the reason lr maps directly to step size. Treat **gnorm** (not loss, not clip%) as the instability alarm; alert on sustained gnorm > ~20 if lr is ever pushed higher.
- **clone_batch 3→6 (toward paper's 12)** — the biggest under-spec vs the paper. More multimask views per item → lower-variance teacher targets → cleaner target_var growth. Reuses the single encoder forward, so it's the cheap way to spend the 57 GB. Validate one step at 6, then push toward 12 if it fits.
- **update_freq 10→12** — free effective-batch growth via grad-accum, no VRAM cost. With cb=6 this roughly doubles the effective batch toward the paper regime.
- **bf16, kept** — fp16 overflow is gone; bf16 has the dynamic range, no loss_scale/overflow handling.
- **max_update 320k, unchanged** — we're ~3–4% in and barely out of the 10k warmup; nothing has plateaued. The paper ran 20 days / 100 epochs to settle tv ~0.5; we've only just hit that *value*, not the converged *plateau*. Judge "done" by **target_var plateau + frozen-probe plateau**, never by loss.

**Evaluation plan (start NOW, don't wait for 320k):** run the existing BenchSuite frozen-probe (`/mnt/c/Users/Iaroslav/CETACEANS/bench_suite.py`, the same harness that produced the AVES 0.894 / log-mel 0.654 bars) on ORANGE's checkpoints at ~5k / 7.5k / 10k / 13.5k to locate the probe plateau (cast bf16→fp32 for the probe forward; fixed train/val/test split across all checkpoints so deltas are pure encoder quality). Probe BLUE too, to confirm lr=1e-4 ≥ lr=5e-5 downstream, but don't expect its peak yet. **Beat log-mel 0.654** = learned beyond raw spectrogram (minimum bar); **approach/beat AVES 0.894** = competitive with strong published animal SSL (headline target). The early-checkpoint curve shape tells you the harvest update — confirm the probe does **not** peak at the loss-min (~u2400).