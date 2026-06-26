"""Compatibility shims for running the legacy fairseq animal2vec code on torch 2.x."""

from __future__ import annotations

import collections.abc
import inspect
import sys
import types
from types import ModuleType


def _install_torch_six_shim() -> None:
    if "torch._six" in sys.modules:
        return

    module = types.ModuleType("torch._six")
    module.string_classes = (str, bytes)
    module.int_classes = (int,)
    module.container_abcs = collections.abc
    sys.modules["torch._six"] = module


def _drop_unknown_kwargs(module: ModuleType, function_name: str) -> None:
    if not hasattr(module, function_name):
        return

    original = getattr(module, function_name)
    try:
        allowed = set(inspect.signature(original).parameters)
    except (TypeError, ValueError):
        return

    def wrapper(*args, **kwargs):
        return original(*args, **{key: value for key, value in kwargs.items() if key in allowed})

    setattr(module, function_name, wrapper)


def patch_animal2vec_modules() -> None:
    """Patch already-imported animal2vec modules that capture fairseq helpers."""

    try:
        import animal2vec.nn.modalities.base as base

        _drop_unknown_kwargs(base, "compute_mask_indices")
    except Exception:
        return


def apply_torch2_fairseq_compat() -> None:
    """Apply import-time shims before fairseq and animal2vec modules are imported."""

    _install_torch_six_shim()
