"""Single inference path for publication-facing animal2vec validators.

The input contract comes from the checkpoint config and encoder masking is
always disabled. Keeping those decisions here prevents the sample-rate and
masked-feature bugs that affected earlier validation scripts.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import os
import re
import sys
from contextlib import nullcontext
from typing import Any, Mapping


# Must be set before torch initializes CUDA for deterministic cuBLAS kernels.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


class ConfigError(ValueError):
    """Raised when a checkpoint has no unambiguous input configuration."""


_INFERENCE_ONLY_CONFIG_KEYS = {
    "task.multi_corpus_keys",
    "model.multi_corpus_keys",
    "model.use_bestrq",
    "model.bestrq_codebook_size",
    "model.bestrq_codebook_dim",
    "model.bestrq_n_mels",
    "model.bestrq_n_fft",
    "model.bestrq_hop",
    "model.bestrq_seed",
}


@dataclasses.dataclass(frozen=True)
class InputSpec:
    sample_rate: int
    max_sample_size: int
    normalize: bool
    sample_rate_key: str
    max_sample_size_key: str

    @property
    def window_seconds(self) -> float:
        return self.max_sample_size / self.sample_rate


@dataclasses.dataclass
class LoadedEncoder:
    model: Any
    input_spec: InputSpec
    checkpoint_metadata: dict[str, Any]
    config_sha256: str
    stripped_config_keys: list[str]
    ignored_state_keys: list[str]


def _node_get(node: Any, key: str, default: Any = None) -> Any:
    if node is None:
        return default
    if isinstance(node, Mapping):
        return node.get(key, default)
    try:
        return node.get(key, default)
    except Exception:
        return getattr(node, key, default)


def _nested_get(node: Any, path: tuple[str, ...]) -> Any:
    current = node
    for key in path:
        current = _node_get(current, key)
        if current is None:
            return None
    return current


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"{field} must be a positive integer, got {value!r}")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field} must be a positive integer, got {value!r}") from exc
    if result <= 0:
        raise ConfigError(f"{field} must be positive, got {result}")
    return result


def _first_config_value(
    cfg: Any, paths: tuple[tuple[str, ...], ...], field: str
) -> tuple[int, str]:
    found: list[tuple[Any, str]] = []
    for path in paths:
        value = _nested_get(cfg, path)
        if value is not None:
            found.append((value, ".".join(path)))
    if not found:
        names = ", ".join(".".join(path) for path in paths)
        raise ConfigError(f"checkpoint config is missing {field}; checked {names}")
    normalized = {_positive_int(value, key) for value, key in found}
    if len(normalized) != 1:
        detail = ", ".join(f"{key}={value!r}" for value, key in found)
        raise ConfigError(f"checkpoint config has conflicting {field} values: {detail}")
    return normalized.pop(), found[0][1]


def input_spec_from_config(cfg: Any) -> InputSpec:
    """Read SR/window from cfg; never infer them from paths or defaults."""

    sample_rate, sample_rate_key = _first_config_value(
        cfg,
        (
            ("task", "sample_rate"),
            ("model", "sample_rate"),
            ("model", "modalities", "audio", "sample_rate"),
        ),
        "sample_rate",
    )
    max_sample_size, max_sample_size_key = _first_config_value(
        cfg,
        (
            ("task", "max_sample_size"),
            ("model", "max_sample_size"),
            ("model", "modalities", "audio", "max_sample_size"),
        ),
        "max_sample_size",
    )
    return InputSpec(
        sample_rate=sample_rate,
        max_sample_size=max_sample_size,
        normalize=bool(_nested_get(cfg, ("task", "normalize"))),
        sample_rate_key=sample_rate_key,
        max_sample_size_key=max_sample_size_key,
    )


def extract_features(
    model: Any, source: Any, *, torch_module: Any | None = None
) -> Mapping[str, Any]:
    """Run feature extraction with masking unconditionally disabled.

    ``torch_module`` is injectable so this invariant is testable without the
    heavyweight training environment.
    """

    if torch_module is None:
        import torch as torch_module  # type: ignore[no-redef]
    context = (
        torch_module.inference_mode()
        if hasattr(torch_module, "inference_mode")
        else nullcontext()
    )
    with context:
        output = model(source=source, features_only=True, mask=False)
    if not isinstance(output, Mapping) or "x" not in output:
        raise RuntimeError("animal2vec encoder must return a mapping containing 'x'")
    return output


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def _jsonable_config(cfg: Any) -> Any:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(cfg):
            return OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    except ImportError:
        pass
    if dataclasses.is_dataclass(cfg):
        return dataclasses.asdict(cfg)
    if isinstance(cfg, Mapping):
        return {str(key): _jsonable_config(value) for key, value in cfg.items()}
    if isinstance(cfg, (list, tuple)):
        return [_jsonable_config(value) for value in cfg]
    if isinstance(cfg, (str, int, float, bool)) or cfg is None:
        return cfg
    return str(cfg)


def config_sha256(cfg: Any) -> str:
    import hashlib

    payload = json.dumps(
        _jsonable_config(cfg), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _delete_nested(node: Any, full_key: str) -> bool:
    from omegaconf import open_dict

    parts = full_key.split(".")
    current = node
    for part in parts[:-1]:
        current = _node_get(current, part)
        if current is None:
            return False
    key = parts[-1]
    try:
        with open_dict(current):
            if key in current:
                del current[key]
                return True
    except Exception:
        pass
    if isinstance(current, dict) and key in current:
        del current[key]
        return True
    return False


def _strip_unknown_config_keys(cfg: Any) -> list[str]:
    """Adapt fork configs while returning a complete audit trail."""

    from omegaconf import OmegaConf
    from animal2vec.nn.audio_tasks import AudioConfigCCAS
    from animal2vec.nn.data2vec2 import Data2VecMultiConfig

    stripped: list[str] = []
    for section_name, schema_type in (
        ("task", AudioConfigCCAS),
        ("model", Data2VecMultiConfig),
    ):
        section = _node_get(cfg, section_name)
        if section is None:
            continue
        schema = OmegaConf.structured(schema_type)
        for _ in range(128):
            try:
                OmegaConf.merge(schema, section)
                break
            except Exception as exc:
                text = str(exc)
                match = re.search(r"full_key:\s*(\S+)", text) or re.search(
                    r"Key '([^']+)' not in", text
                )
                if not match:
                    raise ConfigError(
                        f"cannot reconcile checkpoint {section_name} config: {exc}"
                    ) from exc
                qualified_key = f"{section_name}.{match.group(1)}"
                if qualified_key not in _INFERENCE_ONLY_CONFIG_KEYS:
                    raise ConfigError(
                        f"checkpoint config key {qualified_key!r} is unknown to this "
                        "checkout and is not safe to ignore for inference"
                    ) from exc
                if not _delete_nested(section, match.group(1)):
                    raise ConfigError(
                        f"cannot remove allowed checkpoint key {qualified_key!r}"
                    ) from exc
                stripped.append(qualified_key)
        else:
            raise ConfigError(
                f"too many unknown keys in checkpoint {section_name} config"
            )
    optimizer = _node_get(cfg, "optimizer")
    if optimizer is not None and _delete_nested(optimizer, "dynamic_groups"):
        stripped.append("optimizer.dynamic_groups")
    return stripped


def _patch_python311_dataclasses_for_fairseq() -> None:
    """Allow the pinned legacy fairseq dataclasses to import on Python 3.11."""

    import dataclasses as dc

    if sys.version_info < (3, 11) or getattr(dc, "_fairseq_py311_patched", False):
        return
    source = inspect.getsource(dc._get_field)
    source = source.replace("def _get_field", "def _get_field_fairseq_py311", 1)
    source = source.replace(
        "    # For real fields, disallow mutable defaults.  Use unhashable as a proxy\n"
        "    # indicator for mutability.  Read the __hash__ attribute from the class,\n"
        "    # not the instance.\n"
        "    if f._field_type is _FIELD and f.default.__class__.__hash__ is None:\n"
        "        raise ValueError(f'mutable default {type(f.default)} for field '\n"
        "                         f'{f.name} is not allowed: use default_factory')\n\n",
        "",
    )
    namespace = dc.__dict__.copy()
    exec(source, namespace)
    dc._get_field = namespace["_get_field_fairseq_py311"]
    dc._fairseq_py311_patched = True


def _checkpoint_metadata(
    checkpoint: Mapping[str, Any], checkpoint_name: str, spec: InputSpec
) -> dict[str, Any]:
    extra = _node_get(checkpoint, "extra_state", {})
    iterator = _node_get(extra, "train_iterator", {})
    num_updates = _node_get(extra, "num_updates")
    if num_updates is None:
        history = _node_get(checkpoint, "optimizer_history", []) or []
        if history:
            num_updates = _node_get(history[-1], "num_updates")
    return {
        "file": checkpoint_name,
        "num_updates": num_updates,
        "epoch": _node_get(iterator, "epoch"),
        "sample_rate": spec.sample_rate,
        "max_sample_size": spec.max_sample_size,
        "normalize": spec.normalize,
    }


def load_encoder(checkpoint_path: Any, device: str, *, seed: int = 0) -> LoadedEncoder:
    """Load a frozen encoder and reject unexplained weight mismatches."""

    from pathlib import Path

    import torch
    from omegaconf import OmegaConf, open_dict

    from animal2vec.torch2_compat import (
        apply_torch2_fairseq_compat,
        patch_animal2vec_modules,
    )

    _patch_python311_dataclasses_for_fairseq()
    apply_torch2_fairseq_compat()
    import animal2vec.nn  # noqa: F401
    from fairseq import tasks
    from fairseq.dataclass.initialize import add_defaults

    patch_animal2vec_modules()
    path = Path(checkpoint_path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    raw_cfg = checkpoint.get("cfg")
    if raw_cfg is None:
        raise ConfigError(
            f"{path.name} has no embedded cfg; an explicit config loader is required"
        )
    spec = input_spec_from_config(raw_cfg)
    digest = config_sha256(raw_cfg)
    if OmegaConf.is_config(raw_cfg):
        cfg = OmegaConf.create(
            OmegaConf.to_container(raw_cfg, resolve=True, enum_to_str=True)
        )
    else:
        cfg = OmegaConf.create(raw_cfg)
    OmegaConf.set_struct(cfg, False)
    stripped = _strip_unknown_config_keys(cfg)
    add_defaults(cfg)

    cfg.common.cpu = device == "cpu"
    cfg.common.fp16 = False
    cfg.common.bf16 = False
    cfg.common.no_progress_bar = True
    cfg.task.data = "."
    cfg.task.sample_rate = spec.sample_rate
    cfg.task.max_sample_size = spec.max_sample_size
    cfg.dataset.max_tokens = spec.max_sample_size
    cfg.dataset.batch_size = 1
    with open_dict(cfg.model):
        cfg.model.skip_ema = True

    task = tasks.setup_task(cfg.task)
    model = task.build_model(cfg.model)
    state = checkpoint.get("model", {})
    ignored_state = sorted(
        key
        for key in state
        if key == "_ema" or key.startswith(("_ema", "ema."))
    )
    student_state = {
        key: value for key, value in state.items() if key not in ignored_state
    }
    missing, unexpected = model.load_state_dict(student_state, strict=False)
    allowed = ("_ema", "ema.", "bestrq.")
    disallowed_missing = sorted(
        key for key in missing if not key.startswith(allowed)
    )
    disallowed_unexpected = sorted(
        key for key in unexpected if not key.startswith(allowed)
    )
    if disallowed_missing or disallowed_unexpected:
        raise RuntimeError(
            "checkpoint/model mismatch outside the inference allowlist: "
            f"missing={disallowed_missing}, unexpected={disallowed_unexpected}"
        )
    ignored_state.extend(sorted(unexpected))
    metadata = _checkpoint_metadata(checkpoint, path.name, spec)
    del checkpoint, student_state
    seed_everything(seed)
    model = model.to(device).eval()
    return LoadedEncoder(
        model=model,
        input_spec=spec,
        checkpoint_metadata=metadata,
        config_sha256=digest,
        stripped_config_keys=stripped,
        ignored_state_keys=sorted(set(ignored_state)),
    )
