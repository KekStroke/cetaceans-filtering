#!/usr/bin/env python3
"""Drop-in A100 speedups for the animal2vec (fairseq data2vec_multi) training.
Paste at the VERY TOP of the training entrypoint (before the model/trainer is built):
    from a100_speedups import enable; enable()
Safe on torch 1.13 AND 2.x (version-gated lines are guarded). The only behavioural toggle is
cudnn.benchmark, which is OFF by default here on purpose — see the note below."""
import os


def enable(fixed_shapes=False, verbose=True):
    import torch
    # --- TF32: the real A100 win. Only affects fp32-typed matmuls/convs; ops already in bf16/fp16
    #     keep their kernels. matmul-TF32 is OFF by default since torch 1.12; cudnn-TF32 usually on.
    #     ~bit-level precision drop, universally used for training — does NOT change convergence. ---
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # --- cudnn.benchmark helps ONLY when input shapes are STABLE. animal2vec uses token-batching
    #     (max_tokens, required_batch_size_multiple=1, min_sample_size=1) => batch shape varies every
    #     step => benchmark would re-autotune constantly (SLOWER + extra workspace memory, OOM risk on
    #     a full A100). So OFF by default. Turn on ONLY if you pad to one fixed shape. ---
    torch.backends.cudnn.benchmark = bool(fixed_shapes)

    # --- set_float32_matmul_precision exists since torch 1.12 (so it RUNS on his 1.13.1); it just
    #     reaffirms the TF32 matmul path set above. Guarded only for very old torch. ---
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    if verbose:
        dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        print(f"[a100_speedups] TF32 on | cudnn.benchmark={torch.backends.cudnn.benchmark} | {dev}", flush=True)


# NOTE on the allocator: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` helps fragmentation, but
# (1) it is torch >= 2.1 only (no-op on his 1.13), and (2) it must be a REAL launch env var set BEFORE
# CUDA initialises — setting os.environ mid-script does nothing once CUDA is up. So put it on the
# command line, not here:   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train.py
if __name__ == "__main__":
    enable()
