#!/usr/bin/env python3
"""Drop-in A100 speedups for the animal2vec (fairseq data2vec_multi) training.
Paste this import at the VERY TOP of the training entrypoint (before the model/trainer is built),
or `from a100_speedups import enable; enable()`. All toggles here are LOSSLESS w.r.t. convergence
(same effective batch, same optimization) — they only make the GPU kernels faster on A100."""
import os


def enable(verbose=True):
    import torch
    # --- A100 tensor-core matmul in TF32 (OFF by default since torch 1.12 for matmul) ---
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # --- fixed 10s/8kHz input shape -> let cuDNN autotune the conv frontend once ---
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")     # torch 2.x: TF32 path for fp32 matmuls
    except Exception:
        pass
    # --- allocator: less fragmentation on the big EMA-teacher + multimask clones ---
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if verbose:
        dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        print(f"[a100_speedups] TF32+cudnn.benchmark on, alloc=expandable_segments | {dev}", flush=True)


if __name__ == "__main__":
    enable()
