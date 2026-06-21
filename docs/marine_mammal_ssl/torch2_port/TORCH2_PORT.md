# animal2vec training on torch 2.x (+ torch.compile) — port + speed benchmark

**Result: animal2vec (fairseq 0.12.2 / data2vec_multi) pretraining runs end-to-end on torch 2.9 AND the
latest torch 2.12.1, in bf16, on an RTX 5090 (sm_120).** `torch.compile` on the transformer blocks gives a
steady **~1.10–1.11× (≈10%)** training speedup; the latest torch ports cleanly and matches 2.9 in eager.

This is the "rewrite to torch 2.x" lever (the one Anvar's speedup benchmark flagged as *potentially best but
hard*). The hard part is done: the fairseq-1.13-era → torch-2.x gaps are bridged by **monkeypatches in a
`sitecustomize.py`** (zero edits to the fairseq install), plus **bf16** (which nulls the GradScaler/AMP path)
and a couple of vanilla-fairseq config substitutions.

## Why it matters (beyond the 10%)
- **torch 1.13 can't use sm_120 at all** — on a 5090 (or any new GPU) the legacy stack is CPU-only. torch
  2.x is the *only* way to train on current hardware.
- **Native bf16** sidesteps the fp16 `loss_scale` collapse and the AMPOptimizer breakage Anvar hit (the bf16
  branch of `FP16Optimizer` sets `scaler=None`). No loss scaler needed.
- **Unlocks `torch.compile`** (the ~10% here) and, later, SDPA/flash + fused optimizers.

## Benchmark (RTX 5090 Laptop 24GB, bf16, depth12/embed768, steady-state ms/optimizer-update, warmup discarded)
| config | torch 2.9.0+cu128 | torch 2.12.1+cu130 |
|---|---:|---:|
| eager (compile OFF) | 319.2 ms | 317.8 ms |
| **compile (blocks, default)** | **286.4 ms (1.11×)** | **286.5 ms (1.11×)** |
| clone_batch=8 eager → compile (2.12.1) | — | 401.0 → 366.3 ms (**1.10×**) |
| compile `reduce-overhead` (2.9) | 351.5 ms (**slower** ✗) | — |

- **`torch.compile(block, mode="default", dynamic=True, fullgraph=False)` on the 12 transformer blocks ≈ +10%**,
  consistent across torch 2.9 / 2.12.1 and clone_batch 4 / 8.
- **`reduce-overhead` (CUDA graphs) is WORSE** — the masking + clone_batch give variable sequence lengths, so
  cudagraphs re-capture/fallback and add overhead. Use `default`.
- **Latest torch (2.12.1) ≈ 2.9 in eager** (no free lunch from the version bump itself); its value is cleaner
  compile + future kernels.
- The ~10% is the *transformer-block* share; the sinc frontend, boolean masking (`x[masked_b]`), `.item()`
  variance guards, EMA and dataloader stay eager and cap the gain. Compiling them needs refactoring those
  graph-breaking ops.

## The port (what's needed)
1. **`sitecustomize.py`** on `PYTHONPATH` (auto-loads before fairseq). Six monkeypatches, all against
   fairseq/a2v (not torch internals → version-agnostic across 2.9–2.12):
   - `torch._six` shim (defensive no-op — unused by this fairseq build);
   - `compute_mask_indices` kwarg-swallow (drops fork-only `add_masks`);
   - `EMAModuleConfig.__init__` swallow + preserve fork kwargs (`log_norms`, `add_missing_params`);
   - `EMAModule.__init__` reimpl honoring `copy_model=False` (vanilla's unconditional `deepcopy` recurses on
     the task circular ref) + `self.logs={}` (read unconditionally at data2vec2.py:966);
   - `EMAModule.set_decay` swallow fork `weight_decay`;
   - `ModelCriterion.__init__` accept fork `can_sum`.
2. **bf16** (`common.bf16=true, fp16=false, amp=false`) — the key escape from the AMP wall.
3. **Vanilla-fairseq config substitutions** (the real run uses fork-only fairseq features):
   - optimizer `composite`+`dynamic_groups` → plain `adam` + `cosine` (differential LR is irrelevant to throughput);
   - criterion `expanded_model` kept (the model/task `II`-interpolate many keys from it) with focal/segmentation off;
   - `task.min_label_size=-1` (so 0-byte labels aren't filtered in SSL); audio under a dir literally named `wav`
     (the label-path regex requires it); clips exactly 80000 samples; `max_tokens ≥ 80000`.
4. **`torch.compile` toggle** added to `data2vec2.py` (env `A2V_COMPILE_BLOCKS=1`, `A2V_COMPILE_MODE`) — see
   `data2vec2_compile_blocks.patch`. Compiles blocks only; frontend/masking/EMA stay eager.

## Files
`sitecustomize.py` · `smoke_pretrain.yaml` (tiny 24GB pretrain config) · `data2vec2_compile_blocks.patch` ·
`run_smoke.sh` (port smoke) · `run_bench.sh` (compile OFF/ON benchmark).

## Launch
```bash
cd /path/to/animal2vec
export PYTHONPATH=$PWD:$PYTHONPATH                  # auto-loads sitecustomize.py
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
python animal2vec_train.py --config-name smoke_pretrain                       # eager
A2V_COMPILE_BLOCKS=1 python animal2vec_train.py --config-name smoke_pretrain  # +torch.compile (~10%)
```

## Caveats
Measured on a single 5090 laptop at small batch — on the A100 at clone_batch=12 the compute/overhead balance
(and thus the exact compile %) will differ. The config uses vanilla-fairseq optimizer/criterion, so it's a
*throughput* port, not a bit-for-bit reproduction of the fork's training recipe (differential LR etc.). For a
real run, fold the `sitecustomize` shims into the actual config + the fork's composite optimizer.
