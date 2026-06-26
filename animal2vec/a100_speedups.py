"""A100-oriented torch runtime switches for animal2vec training."""

from __future__ import annotations


def enable(fixed_shapes: bool = False, verbose: bool = True) -> None:
    """Enable lossless A100 math fast paths.

    TF32 only affects float32 matmul/conv operations. bf16/fp16 kernels keep their
    native tensor-core paths. cuDNN benchmarking stays off by default because the
    fairseq token batches can vary in shape.
    """

    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = bool(fixed_shapes)

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    if verbose:
        device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        print(
            "[animal2vec.a100_speedups] "
            f"TF32 on | cudnn.benchmark={torch.backends.cudnn.benchmark} | {device}",
            flush=True,
        )
