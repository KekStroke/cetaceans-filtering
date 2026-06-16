# Speeding up the current animal2vec training on A100 — grounded in Anvar's config + log

**Measured now** (his `animal2vec_train.log`, run1): `wps≈180 000`, `ups=0.02` (≈49 s/optimizer-update),
`wpb≈8.89M` tokens/update (`update_freq=80 × clone_batch=12 × max_tokens=408000`), `bsz≈399`,
`max_update=100000` → **≈2 months**. Stack: **torch 1.13.1 + fairseq 0.12.2**, single A100 per run
(`distributed_world_size=1`, two parallel runs blue/orange).

**Diagnosis — it is COMPUTE-bound, not I/O-bound.** `wps≈180k` is high and steady; if the loader were
starving the GPU, wps would sag and jitter. So the big A100 wins are **faster math kernels**, not the data
pipeline. That also means: **you can cut wall-clock without touching the optimization** — keep
`wpb≈8.89M` tokens/update (same effective batch ⇒ same convergence ⇒ safe on the "edущий" run), just make
each token cheaper.

---

## Tier 0 — free & LOSSLESS, safe on the live run (do today)
Same math, faster kernels — won't move the trajectory.

1. **bf16, not fp16** — A100 has native bf16; no loss-scaler, no `fp16_init_scale:1` collapse. (Run1 was
   `fp16:true`; runs-2 already switched — just confirm BOTH runs are bf16.)
   ```yaml
   common: { fp16: false, bf16: true, amp: false }
   ```
2. **Enable TF32** (fairseq never sets it; torch 1.13 has matmul-TF32 OFF by default) +
   cudnn.benchmark (shapes are fixed 10 s/8 kHz). Paste `a100_speedups.py::enable()` at the top of the
   training entrypoint. Free ~1.1–1.3× on the fp32 ops, autotuned conv frontend.
3. **Loader headroom** (cheap insurance against tail stalls): `num_workers 8→16` (A100 boxes have the
   cores), `persistent_workers`, `prefetch_factor: 4`, pin_memory. Won't be the big win here (compute-bound)
   but removes the occasional stall.

Expected Tier 0: **~1.2–1.4×**, zero risk to convergence.

## Tier 1 — LOSSLESS but a structural change (apply on the run you keep)
4. **Fill the A100 and hold the effective batch.** `max_tokens=408000` is modest for 80 GB. Raise it until
   ~85 % memory, and **cut `update_freq` by the same factor** so `max_tokens × update_freq` (⇒ `wpb`) stays
   ≈8.89M. Bigger micro-batches = better tensor-core occupancy + fewer accumulation iterations (less
   Python/loader overhead per update) → higher wps, **identical effective batch → identical dynamics.**
   ```
   e.g. max_tokens 408k→816k  AND  update_freq 80→40   (wpb unchanged ≈8.89M)
   ```
   This is the **cautious-friendly throughput win** — recommend it even on the live run. Expected **~1.3–1.7×**.
5. **Port training to torch 2.x + cu12 → `torch.compile`** (highest pure-compute lever). We already proved
   animal2vec *loads & runs* on torch 2.9+cu128 on sm_120 via the shims in `a2v_extract.py` (torch._six
   shim, `compute_mask_indices` monkeypatch) — that de-risks the import side. With torch 2.x:
   - `model = torch.compile(model)` on the encoder — typically **1.2–1.7×** (fuses the sinc/conv frontend +
     transformer MLPs), lossless.
   - fused AdamW (`fused=True`), better SDPA. (SDPA-flash gain is *small* here — sequences are short
     ~250 frames after the frontend, so attention isn't the bottleneck; MLP/conv dominate.)
   Cost: ~a day of porting (fairseq 0.12.2 has rough edges on torch 2.x — same shims + check the AMP/optimizer
   path). Highest ceiling of everything here.
6. **After you pick the winner run, go DDP across both A100** (`distributed_world_size=2`, halve
   `update_freq` to hold `wpb`) → **~2×**, the canonical near-lossless multi-GPU speedup. (You keep 2 separate
   runs only while comparing blue vs orange; once one is killed, the survivor should use both cards.)

## Tier 2 — DYNAMICS-CHANGING (only on the run you're willing to throw away)
These make it faster by making each update *cheaper in data*, i.e. they change the optimization — exactly
the "scary" ones. Don't touch the run that's converging nicely.
- **Lower the effective batch** (drop `update_freq` or `clone_batch` *without* compensating). More
  updates/sec, but smaller/weaker SSL signal per step → different (often worse) convergence; would need LR
  re-tuning. `clone_batch` also controls the number of masked views (the multimask objective) — reducing it
  weakens the data2vec target. Treat as an *experiment*, not a speedup.

---

## Recommended order
1. Tier 0 (today, both runs) — bf16 + `a100_speedups.enable()` + loader knobs.
2. Tier 1 #4 (fill GPU, hold `wpb`) — safe throughput, even on the live run.
3. After blue-vs-orange is decided: Tier 1 #6 (DDP both A100) + #5 (torch 2.x + compile) on the survivor.
4. Leave Tier 2 for the disposable run only.

Net realistic, lossless: **Tier 0×1 ≈1.3×**, **+ fill-GPU ≈1.5×**, **+ torch.compile ≈1.5×**,
**+ DDP×2** → compounded ≈ **4–6×** on the kept run (≈2 months → ~1.5–3 weeks) **without changing the
result**. The compute-bound profile is *good news*: the speed is in kernels you can swap losslessly, not in
a data rewrite.

## Files
`a100_speedups.py` (drop-in TF32/cudnn/alloc toggles). Earlier: `TRAINING_SPEED.md`,
`DIAGNOSIS_ANVAR_RUN.md` (the fp16→bf16 / update_freq context).
