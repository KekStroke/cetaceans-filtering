# animal2vec on torch 2.x (+ torch.compile) — port + speed benchmark

**The "port" is NOT a code rewrite.** animal2vec pretraining (data2vec_multi, the *real* config with the
`composite` differential-LR optimizer) runs **end-to-end on the latest torch 2.12.1 (cu130), bf16, RTX 5090
(sm_120), with ZERO source changes and ZERO monkeypatches** — *provided you install the fairseq commit
animal2vec actually requires.* `torch.compile` on the transformer blocks adds **~1.12–1.13× (≈12%)**.

## The real finding (why it looked hard, and isn't)
animal2vec's README pins fairseq to a **specific commit**, not the PyPI release:
```
pip install git+https://github.com/facebookresearch/fairseq.git@920a548ca770fb1a951f7f4289b4d3a0c1bc226f
```
The PyPI wheel `fairseq==0.12.2` is a **later, feature-stripped** build that **lacks** the data2vec-era APIs
animal2vec uses (`EMAModule(copy_model=…)`, `EMAModuleConfig.log_norms/add_missing_params`, composite
`dynamic_groups`, `ModelCriterion(can_sum=…)`, `compute_mask_indices(add_masks=…)`). Verified that commit
920a548 **has all of them natively**. So the apparent "torch-2.x incompatibilities" were almost entirely a
**fairseq-version mismatch**, not torch. Install the right commit and they vanish — `torch._six` is only
referenced in `model_parallel/megatron` (unused by animal2vec), so **no shim is needed at all**.

**Verified (RTX 5090, torch 2.12.1+cu130, fairseq@920a548, EMPTY sitecustomize):** `real_pretrain.yaml`
(composite optimizer, two LR groups `lr_default` + decoder) trains end-to-end, bf16, healthy
(loss↓, target_var 0.35, gnorm ~27). 16 kHz @ 5 s (`real_16khz.yaml`) also runs.

## Recipe (the whole "port")
```bash
uv venv a2v --python 3.10 && source a2v/bin/activate            # or python -m venv + pip==24.0
uv pip install numpy==1.23.5 cython==3.2.5 setuptools wheel
uv pip install torch==2.12.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
uv pip install --no-build-isolation \
    "git+https://github.com/facebookresearch/fairseq.git@920a548ca770fb1a951f7f4289b4d3a0c1bc226f"   # torch must precede (setup.py imports it)
uv pip install scipy scikit-learn librosa soundfile matplotlib pandas \
    tensorflow==2.11.0 timm==0.6.12 scikit-image intervaltree iopath          # animal2vec nn deps
git clone https://github.com/livingingroups/animal2vec ~/a2v
# then: bf16 in the config (common.bf16=true, fp16=false, amp=false) and train. No shims, no source edits.
```
The only config change vs the legacy run is **bf16** (sidesteps the fp16 loss-scale collapse / AMP path —
this is config, not code). torchvision/torchaudio must match the torch ABI (else `torchvision::nms` missing).

## torch.compile benchmark (RTX 5090 24GB, bf16, depth12/embed768, steady-state ms/optimizer-update)
| stack | eager | compile (blocks, default, dynamic) | speedup |
|---|---:|---:|---:|
| **fairseq@920a548 + torch 2.12.1** (the real stack) | 317.1 ms | **279.9 ms** | **1.13×** |
| vanilla 0.12.2 + shims, torch 2.9 | 319.2 ms | 286.4 ms | 1.11× |
| vanilla 0.12.2 + shims, torch 2.12.1 | 317.8 ms | 286.5 ms | 1.11× |

- `torch.compile(block, mode="default", dynamic=True, fullgraph=False)` on the 12 transformer blocks ≈ **+12%**,
  stable across torch/clone_batch. Toggle: env `A2V_COMPILE_BLOCKS=1` (one ~10-line edit in `data2vec2.py`,
  see `data2vec2_compile_blocks.patch`). Frontend/masking/EMA stay eager.
- `reduce-overhead` (CUDA graphs) is **worse** — variable seq-len (masking) defeats cudagraphs. Use `default`.
- Latest torch ≈ 2.9 in eager (the version bump alone gives nothing; compile + future kernels are the point).
- The ~12% is the transformer-block share; the eager sinc frontend, masking, EMA, dataloader cap the gain.

## Why bother (beyond +12%)
- **torch 1.13 cannot use sm_120 at all** — on a 5090/newer the legacy stack is CPU-only. torch 2.x is the
  *only* way to train on current hardware.
- **Native bf16** removes the fp16 `loss_scale` collapse / Inf-NaN you hit at large batch.
- Unlocks `torch.compile` (the +12%) and future SDPA/flash + fused optimizers.

## Files
- `data2vec2_compile_blocks.patch` — the **only** source edit (torch.compile toggle in data2vec2.py).
- `real_pretrain.yaml` — real config (composite optimizer) for the smoke. `real_16khz.yaml` — 16 kHz @ 5 s.
  `smoke_pretrain.yaml` — simplified (plain adam) variant. `run_smoke.sh` / `run_bench.sh` — portable runners.
- `sitecustomize.py` — **OPTIONAL FALLBACK ONLY** — recreates the 6 missing APIs *if you're forced onto the
  PyPI `fairseq==0.12.2` wheel* (no git build). **Not needed** with fairseq@920a548.

## Caveats
Measured on one 5090 laptop at small batch — on the A100 at clone_batch=12 the compute/overhead balance (and
exact compile %) will differ. The smoke uses tiny data/steps to prove the stack; for real training keep the
full schedule.
