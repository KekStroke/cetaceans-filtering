#!/usr/bin/env python3
"""
Universal downstream validation for animal2vec checkpoints.

What this script checks
-----------------------
It freezes animal2vec, extracts per-layer embeddings, then trains a simple
linear probe (StandardScaler + LogisticRegression) for sound/noise classification.
The checkpoint weights are never updated.

Two dataset layouts are supported:

1) Folder labels: separate folders for target sound and noise.
   Use this for Dominica-like data where the whole file label is known from
   the folder name.

   Change these options for a new folder dataset:
     --data-dir    root containing the label folders
     --sound-dir   folder name for target sounds, e.g. Signal_parts
     --noise-dir   folder name for noise, e.g. Noise_parts

   Example:
     python scripts/validate_animal2vec_unified.py folder CHECKPOINT1.pt CHECKPOINT2.pt ^
       --data-dir data/Dominica_dataset ^
       --sound-dir Signal_parts ^
       --noise-dir Noise_parts ^
       --out-dir outputs/animal2vec/unified/dominica

2) JSON annotations: labels live inside each audio timeline.
   Use this for Label Studio / annotations.json style data where one wav can
   contain both target sound and noise/artifacts/unmarked audio.

   Change these options for a new annotation dataset:
     --data-dir         folder with wav files
     --annotation       JSON annotation file
     --positive-label   label treated as target sound, default: sound
     --include-unmarked include unannotated gaps as noise, default: on

   Example:
     python scripts/validate_animal2vec_unified.py json CHECKPOINT1.pt CHECKPOINT2.pt ^
       --data-dir data/drive_whale_validation_zips ^
       --annotation data/drive_whale_validation_zips/annotations.json ^
       --out-dir outputs/animal2vec/unified/annotations

Input window size
-----------------
The script keeps the animal2vec training-compatible input length:
  80000 samples from the checkpoint config.
So a 16 kHz checkpoint receives 5 seconds; an 8 kHz checkpoint receives 10 seconds.
For JSON annotations, the event centers are selected once and each checkpoint
reads its own 5s/10s window around the same center.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import inspect
import json
import math
import random
import re
import sys
import tempfile
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from animal2vec.torch2_compat import apply_torch2_fairseq_compat, patch_animal2vec_modules


DEFAULT_CONFIG = REPO_ROOT / "animal2vec/configs/cetaceans/pretrain_16khz_5s_torch2.yaml"
DEFAULT_OUT = REPO_ROOT / "outputs/animal2vec/unified_validation"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def patch_python311_dataclasses_for_fairseq() -> None:
    """Fairseq 0.12 has mutable dataclass defaults that Python 3.11 rejects."""
    if sys.version_info < (3, 11) or getattr(dataclasses, "_fairseq_py311_patched", False):
        return

    source = inspect.getsource(dataclasses._get_field)
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
    namespace = dataclasses.__dict__.copy()
    exec(source, namespace)
    dataclasses._get_field = namespace["_get_field_fairseq_py311"]
    dataclasses._fairseq_py311_patched = True


def setup_animal2vec_runtime() -> None:
    patch_python311_dataclasses_for_fairseq()
    apply_torch2_fairseq_compat()
    import animal2vec.nn  # noqa: F401 - registers fairseq models/tasks

    patch_animal2vec_modules()


def short_tag(tag: str) -> str:
    low = tag.lower()
    sr = "16k" if "16khz" in low or "16k" in low else "8k" if "8khz" in low or "8k" in low else ""
    layer = ""
    match = re.search(r"(?:^|_)L(\d+)(?:_|$)", tag)
    if match:
        layer = f"L{match.group(1)}"
    return " ".join(part for part in (sr, layer) if part) or tag[:28]


def display_tag(result: dict[str, Any]) -> str:
    label = short_tag(result["tag"])
    if not re.match(r"^\d+k\b", label):
        sample_rate = result.get("checkpoint_meta", {}).get("sample_rate")
        if sample_rate:
            label = f"{int(sample_rate) // 1000}k {label}"
    return label


@contextmanager
def checkpoint_path(path: Path) -> Iterator[Path]:
    if path.suffix.lower() != ".zip":
        yield path
        return

    with zipfile.ZipFile(path) as archive:
        members = [m for m in archive.namelist() if m.lower().endswith((".pt", ".pth", ".ckpt"))]
        if not members:
            raise ValueError(f"{path} does not contain a .pt/.pth/.ckpt checkpoint")
        member = sorted(members, key=lambda name: archive.getinfo(name).file_size, reverse=True)[0]
        with tempfile.TemporaryDirectory(prefix="a2v_ckpt_") as tmp_dir:
            tmp_path = Path(tmp_dir) / Path(member).name
            log(f"extracting {member} from {path.name}")
            with archive.open(member) as src, tmp_path.open("wb") as dst:
                while True:
                    chunk = src.read(1024 * 1024 * 64)
                    if not chunk:
                        break
                    dst.write(chunk)
            yield tmp_path


def cfg_get(node: Any, key: str, default: Any = None) -> Any:
    try:
        return node.get(key, default)
    except Exception:
        try:
            return node[key]
        except Exception:
            return default


def cfg_set(node: Any, key: str, value: Any) -> None:
    try:
        node[key] = value
    except Exception:
        with open_dict(node):
            node[key] = value


def cfg_del_nested(node: Any, full_key: str) -> bool:
    parts = full_key.split(".")
    current = node
    for part in parts[:-1]:
        try:
            current = current[part]
        except Exception:
            return False
    key = parts[-1]
    try:
        if key in current:
            del current[key]
            return True
    except Exception:
        pass
    try:
        with open_dict(current):
            if key in current:
                del current[key]
                return True
    except Exception:
        pass
    return False


def checkpoint_cfg(ckpt: dict[str, Any]) -> Any:
    cfg = ckpt.get("cfg")
    if cfg is None:
        cfg = OmegaConf.load(DEFAULT_CONFIG)
    elif OmegaConf.is_config(cfg):
        cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True, enum_to_str=True))
    else:
        cfg = OmegaConf.create(cfg)
    OmegaConf.set_struct(cfg, False)
    return cfg


def strip_unknown_cfg_keys(cfg: Any) -> None:
    from animal2vec.nn.audio_tasks import AudioConfigCCAS
    from animal2vec.nn.data2vec2 import Data2VecMultiConfig

    sections = (("task", AudioConfigCCAS), ("model", Data2VecMultiConfig))
    for section, dataclass_type in sections:
        node = cfg_get(cfg, section)
        if node is None:
            continue
        schema = OmegaConf.structured(dataclass_type)
        for _ in range(128):
            try:
                OmegaConf.merge(schema, node)
                break
            except Exception as exc:
                text = str(exc)
                match = re.search(r"full_key:\s*(\S+)", text) or re.search(r"Key '([^']+)' not in", text)
                if not match:
                    raise
                full_key = match.group(1)
                if not cfg_del_nested(node, full_key):
                    raise
                log(f"stripped unknown checkpoint cfg key: {section}.{full_key}")
    optimizer_cfg = cfg_get(cfg, "optimizer")
    if optimizer_cfg is not None and cfg_del_nested(optimizer_cfg, "dynamic_groups"):
        log("stripped unknown checkpoint cfg key: optimizer.dynamic_groups")


def checkpoint_meta(ckpt: dict[str, Any], ckpt_file: Path) -> dict[str, Any]:
    cfg = ckpt.get("cfg")
    task_cfg = cfg_get(cfg_get(cfg, "task", {}), "sample_rate")
    model_cfg = cfg_get(cfg_get(cfg, "model", {}), "sample_rate")
    audio_cfg = cfg_get(cfg_get(cfg_get(cfg, "model", {}), "modalities", {}), "audio", {})
    max_sample_size = cfg_get(cfg_get(cfg, "task", {}), "max_sample_size") or 80000
    sample_rate = task_cfg or model_cfg or cfg_get(audio_cfg, "sample_rate") or 8000
    return {
        "checkpoint": str(ckpt_file),
        "checkpoint_name": ckpt_file.name,
        "num_updates": cfg_get(ckpt.get("extra_state", {}), "num_updates"),
        "epoch": cfg_get(cfg_get(ckpt.get("extra_state", {}), "train_iterator", {}), "epoch"),
        "sample_rate": int(sample_rate),
        "max_sample_size": int(max_sample_size),
    }


def load_feature_model(ckpt_file: Path, device: str) -> tuple[torch.nn.Module, dict[str, Any]]:
    setup_animal2vec_runtime()
    from fairseq import tasks
    from fairseq.dataclass.initialize import add_defaults

    log(f"loading checkpoint: {ckpt_file}")
    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    meta = checkpoint_meta(ckpt, ckpt_file)
    cfg = checkpoint_cfg(ckpt)
    strip_unknown_cfg_keys(cfg)
    add_defaults(cfg)

    cfg_set(cfg.common, "cpu", device == "cpu")
    cfg_set(cfg.common, "fp16", False)
    cfg_set(cfg.common, "bf16", False)
    cfg_set(cfg.common, "no_progress_bar", True)
    cfg_set(cfg.task, "data", str(REPO_ROOT))
    cfg_set(cfg.task, "sample_rate", meta["sample_rate"])
    cfg_set(cfg.task, "max_sample_size", meta["max_sample_size"])
    cfg_set(cfg.task, "normalize", False)
    cfg_set(cfg.dataset, "max_tokens", meta["max_sample_size"])
    cfg_set(cfg.dataset, "batch_size", 1)
    cfg_set(cfg.model, "skip_ema", True)

    task = tasks.setup_task(cfg.task)
    model = task.build_model(cfg.model)
    state = ckpt.get("model", {})
    state = {k: v for k, v in state.items() if not (k == "_ema" or k.startswith("_ema") or k.startswith("ema."))}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        log(f"model load: {len(missing)} missing keys")
    if unexpected:
        log(f"model load: {len(unexpected)} unexpected keys")
    del ckpt, state

    model = model.to(device).eval()
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    log(f"model ready on {device}: {params_m:.0f}M params, sr={meta['sample_rate']}, window={meta['max_sample_size']}")
    return model, meta


def read_audio_window(path: Path, start_s: float, window_s: float, target_sr: int, target_samples: int) -> np.ndarray:
    import soundfile as sf
    from scipy.signal import resample_poly

    with sf.SoundFile(str(path)) as handle:
        source_sr = int(handle.samplerate)
        start_frame = max(0, int(round(start_s * source_sr)))
        frames = int(math.ceil(window_s * source_sr))
        handle.seek(min(start_frame, max(0, len(handle) - 1)))
        audio = handle.read(frames, dtype="float32", always_2d=True)
    if audio.size == 0:
        raise ValueError(f"empty audio window from {path}")
    y = audio.mean(axis=1).astype(np.float32)
    if source_sr != target_sr:
        gcd = math.gcd(source_sr, target_sr)
        y = resample_poly(y, target_sr // gcd, source_sr // gcd).astype(np.float32)
    if len(y) > target_samples:
        y = y[:target_samples]
    elif len(y) < target_samples:
        y = np.pad(y, (0, target_samples - len(y))).astype(np.float32)
    y = y - float(y.mean())
    std = float(y.std())
    if std > 1e-8:
        y = y / std
    return y.astype(np.float32)


def embed_layers(model: torch.nn.Module, wav: np.ndarray, device: str) -> list[np.ndarray]:
    with torch.inference_mode():
        x = torch.from_numpy(wav).view(1, -1).to(device)
        out = model(source=x, features_only=True, mask=False)
        feats: list[np.ndarray] = []
        for layer_result in out.get("layer_results") or []:
            tensor = layer_result[0] if isinstance(layer_result, (tuple, list)) else layer_result
            feats.append(tensor.reshape(-1, tensor.shape[-1]).float().mean(0).cpu().numpy())
        feats.append(out["x"][0].reshape(-1, out["x"].shape[-1]).float().mean(0).cpu().numpy())
        return feats


def extract_all_layers(model: torch.nn.Module, wavs: list[np.ndarray], device: str) -> list[np.ndarray]:
    per_layer: list[list[np.ndarray]] | None = None
    for i, wav in enumerate(wavs, start=1):
        feats = embed_layers(model, wav, device)
        if per_layer is None:
            per_layer = [[] for _ in feats]
        for layer_i, feat in enumerate(feats):
            per_layer[layer_i].append(feat)
        if i % 20 == 0 or i == len(wavs):
            log(f"  embedded {i}/{len(wavs)} windows")
    if per_layer is None:
        raise RuntimeError("no embeddings were produced")
    return [np.stack(values) for values in per_layer]


def normalize_audio_name(name: str) -> str:
    """Label Studio exports can prepend an 8-char upload id: abc123ef-file.wav."""
    return re.sub(r"^[0-9a-fA-F]{8}-", "", Path(name).name)


def clipped_window_start(center_s: float, window_s: float, duration_s: float) -> float:
    if duration_s <= window_s:
        return 0.0
    return min(max(0.0, center_s - window_s / 2.0), duration_s - window_s)


def evenly_spaced_centers(duration_s: float, window_s: float, count: int) -> list[float]:
    if duration_s <= window_s:
        return [duration_s / 2.0]
    count = max(1, count)
    if count == 1:
        return [duration_s / 2.0]
    margin = min(1.0, max(0.0, (duration_s - window_s) / 10.0))
    lo = margin + window_s / 2.0
    hi = max(lo, duration_s - margin - window_s / 2.0)
    return np.linspace(lo, hi, count).tolist()


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(float(start), float(end)) for start, end in merged]


def gap_centers(gaps: list[tuple[float, float]], selection_window_s: float, max_per_file: int) -> list[float]:
    centers: list[float] = []
    for start, end in gaps:
        if end - start < selection_window_s:
            continue
        n = max(1, int((end - start) // max(selection_window_s * 3.0, 1.0)))
        for center in np.linspace(start + selection_window_s / 2.0, end - selection_window_s / 2.0, min(n, 8)):
            centers.append(float(center))
    return centers[:max_per_file]


def audio_name_from_task(item: dict[str, Any]) -> str | None:
    if isinstance(item.get("audio"), str):
        return Path(item["audio"]).name
    if isinstance(item.get("file_upload"), str):
        return Path(item["file_upload"]).name
    data = item.get("data")
    if isinstance(data, dict) and isinstance(data.get("audio"), str):
        return Path(data["audio"]).name
    return None


def regions_from_task(item: dict[str, Any]) -> list[dict[str, Any]]:
    # Simple export shape: {"audio": "...", "label": [{"start": ..., "labels": [...]}]}
    if isinstance(item.get("label"), list):
        return item["label"]

    # Label Studio export shape: {"annotations": [{"result": [{"value": {...}}]}]}
    regions: list[dict[str, Any]] = []
    for annotation in item.get("annotations") or []:
        for result in annotation.get("result") or []:
            value = result.get("value")
            if isinstance(value, dict):
                regions.append(value)
    return regions


def load_annotation_tasks(annotation_path: Path) -> list[dict[str, Any]]:
    obj = json.loads(annotation_path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise ValueError(f"{annotation_path} must contain a JSON list")
    tasks = []
    for item in obj:
        if not isinstance(item, dict):
            continue
        audio = audio_name_from_task(item)
        if not audio:
            continue
        intervals = []
        for region in regions_from_task(item):
            try:
                start = float(region["start"])
                end = float(region["end"])
            except Exception:
                continue
            labels = [str(label).lower() for label in region.get("labels", [])]
            if end > start:
                intervals.append({"start": start, "end": end, "labels": labels})
        tasks.append({"audio": audio, "intervals": intervals})
    return tasks


def folder_candidates(args: argparse.Namespace, selection_window_s: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import soundfile as sf

    sound_root = args.data_dir / args.sound_dir
    noise_root = args.data_dir / args.noise_dir
    if not sound_root.exists() or not noise_root.exists():
        raise SystemExit(f"Expected {sound_root} and {noise_root}. Change --sound-dir/--noise-dir for this dataset.")

    candidates: list[dict[str, Any]] = []
    counts = {"sound_files": 0, "noise_files": 0}
    for label_name, label, folder in (("sound", 1, sound_root), ("noise", 0, noise_root)):
        files = sorted(folder.glob("*.wav"))
        if args.max_files_per_class > 0:
            files = files[: args.max_files_per_class]
        counts[f"{label_name}_files"] = len(files)
        for path in files:
            duration = float(sf.info(str(path)).duration)
            for center in evenly_spaced_centers(duration, selection_window_s, args.windows_per_file):
                candidates.append({"path": path, "center": center, "label": label, "group": path.stem, "source": label_name})

    return candidates, {"layout": "folder", **counts}


def json_candidates(args: argparse.Namespace, selection_window_s: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import soundfile as sf

    tasks = load_annotation_tasks(args.annotation)
    wavs_by_name = {p.name: p for p in args.data_dir.glob("*.wav")}
    wavs_by_name.update({normalize_audio_name(p.name): p for p in args.data_dir.glob("*.wav")})
    positive_label = args.positive_label.lower()

    missing = sorted({task["audio"] for task in tasks if normalize_audio_name(task["audio"]) not in wavs_by_name})
    matched = [task for task in tasks if normalize_audio_name(task["audio"]) in wavs_by_name]
    if not matched:
        raise SystemExit(
            f"No annotation audio names matched wav files under {args.data_dir}. "
            "Check --data-dir or strip/mapping rules."
        )

    candidates: list[dict[str, Any]] = []
    counts = {"matched_audio_files": len(matched), "missing_audio_files": len(missing), "sound_regions": 0,
              "non_sound_regions": 0, "unmarked_regions": 0, "missing_audio_examples": missing[:20]}
    for task in matched:
        path = wavs_by_name[normalize_audio_name(task["audio"])]
        duration = float(sf.info(str(path)).duration)
        all_intervals: list[tuple[float, float]] = []
        for interval in task["intervals"]:
            start = max(0.0, min(float(interval["start"]), duration))
            end = max(0.0, min(float(interval["end"]), duration))
            if end <= start:
                continue
            labels = set(interval["labels"])
            all_intervals.append((start, end))
            is_sound = positive_label in labels
            counts["sound_regions" if is_sound else "non_sound_regions"] += 1
            candidates.append({
                "path": path,
                "center": (start + end) / 2.0,
                "label": 1 if is_sound else 0,
                "group": path.stem,
                "source": "sound" if is_sound else "labeled_noise",
            })

        if args.include_unmarked:
            cursor = 0.0
            gaps = []
            for start, end in merge_intervals(all_intervals):
                if start > cursor:
                    gaps.append((cursor, start))
                cursor = max(cursor, end)
            if cursor < duration:
                gaps.append((cursor, duration))
            for center in gap_centers(gaps, selection_window_s, args.max_unmarked_per_file):
                candidates.append({"path": path, "center": center, "label": 0, "group": path.stem, "source": "unmarked_noise"})
                counts["unmarked_regions"] += 1

    return candidates, {"layout": "json", "annotation": str(args.annotation), **counts}


def balance_candidates(candidates: list[dict[str, Any]], max_examples_per_class: int) -> list[dict[str, Any]]:
    rng = random.Random(42)
    sound = [item for item in candidates if item["label"] == 1]
    noise = [item for item in candidates if item["label"] == 0]
    rng.shuffle(sound)
    rng.shuffle(noise)
    limit = min(max_examples_per_class, len(sound), len(noise))
    if limit < 2:
        raise SystemExit(f"Not enough balanced examples: sound={len(sound)}, noise={len(noise)}")
    selected = sound[:limit] + noise[:limit]
    rng.shuffle(selected)
    return selected


def materialize_windows(
    candidates: list[dict[str, Any]],
    model_sr: int,
    target_samples: int,
    log_name: str,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, dict[str, Any]]:
    import soundfile as sf

    window_s = target_samples / model_sr
    durations = {item["path"]: float(sf.info(str(item["path"])).duration) for item in candidates}
    wavs: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []
    for i, item in enumerate(candidates, start=1):
        start_s = clipped_window_start(float(item["center"]), window_s, durations[item["path"]])
        wavs.append(read_audio_window(item["path"], start_s, window_s, model_sr, target_samples))
        labels.append(int(item["label"]))
        groups.append(str(item["group"]))
        if i % 100 == 0 or i == len(candidates):
            log(f"  prepared {i}/{len(candidates)} windows from {log_name}")

    return (
        wavs,
        np.asarray(labels, dtype=np.int64),
        np.asarray(groups),
        {
            "window_seconds": window_s,
            "n_examples": len(candidates),
            "n_sound": int(sum(labels)),
            "n_noise": int(len(labels) - sum(labels)),
        },
    )


def probe_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict[str, float | None]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    n_splits = min(5, len(set(groups)), int(np.bincount(y).min()))
    if n_splits < 2:
        raise ValueError("Need at least two examples per class and two audio groups for grouped CV")

    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    y_score: list[np.ndarray] = []
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, test_idx in cv.split(X, y, groups):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"))
        clf.fit(X[train_idx], y[train_idx])
        y_true.append(y[test_idx])
        y_pred.append(clf.predict(X[test_idx]))
        y_score.append(clf.predict_proba(X[test_idx])[:, 1])

    truth = np.concatenate(y_true)
    pred = np.concatenate(y_pred)
    score = np.concatenate(y_score)
    result: dict[str, float | None] = {
        "macro_f1": float(f1_score(truth, pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
    }
    try:
        result["auc"] = float(roc_auc_score(truth, score))
    except ValueError:
        result["auc"] = None
    return result


def validate_checkpoint(
    ckpt_arg: Path,
    args: argparse.Namespace,
    selected_candidates: list[dict[str, Any]],
    dataset_meta: dict[str, Any],
    run_name: str,
) -> dict[str, Any]:
    import torch

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    with checkpoint_path(ckpt_arg.resolve()) as ckpt_file:
        model, ckpt_meta = load_feature_model(ckpt_file, device)
        wavs, labels, groups, materialized_meta = materialize_windows(
            selected_candidates,
            ckpt_meta["sample_rate"],
            ckpt_meta["max_sample_size"],
            run_name,
        )
        layers = extract_all_layers(model, wavs, device)
        per_layer = {}
        for layer_i, X in enumerate(layers):
            layer_name = "final" if layer_i == len(layers) - 1 else f"L{layer_i}"
            per_layer[layer_name] = probe_cv(X, labels, groups)
            metrics = per_layer[layer_name]
            log(
                f"  {run_name} {layer_name}: macro-F1={metrics['macro_f1']:.4f}, "
                f"bal-acc={metrics['balanced_accuracy']:.4f}, auc={metrics['auc']}"
            )
        best_layer = max(per_layer, key=lambda layer: float(per_layer[layer]["macro_f1"] or -1.0))
        result = {
            "tag": ckpt_arg.stem,
            "checkpoint": str(ckpt_arg),
            "checkpoint_meta": ckpt_meta,
            "task": f"{run_name}_sound_noise",
            "primary_metric": "macro_f1",
            "best_layer": best_layer,
            "best": per_layer[best_layer],
            "per_layer": per_layer,
            "data": {**dataset_meta, **materialized_meta},
        }
        log(f"{run_name} {ckpt_arg.name} best {best_layer}: {result['best']['macro_f1']:.4f}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def write_outputs(results: list[dict[str, Any]], out_dir: Path, run_name: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{run_name}_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    rows = []
    for result in results:
        for layer, metrics in result["per_layer"].items():
            rows.append({
                "run": run_name,
                "checkpoint": display_tag(result),
                "layer": layer,
                "is_best": layer == result["best_layer"],
                "model_sr": result["checkpoint_meta"]["sample_rate"],
                "window_seconds": result["data"]["window_seconds"],
                "n_examples": result["data"]["n_examples"],
                "n_sound": result["data"]["n_sound"],
                "n_noise": result["data"]["n_noise"],
                "macro_f1": metrics["macro_f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "auc": metrics["auc"],
            })
    with (out_dir / f"{run_name}_per_layer.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"# animal2vec {run_name} validation",
        "",
        "Task: binary downstream probe, `sound` vs `noise`.",
        "",
        "| checkpoint | sr | best layer | macro-F1 | bal. acc | AUC | examples | sound | noise |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        best = result["best"]
        auc = "" if best["auc"] is None else f"{best['auc']:.4f}"
        lines.append(
            f"| {display_tag(result)} | {result['checkpoint_meta']['sample_rate']} | {result['best_layer']} | "
            f"{best['macro_f1']:.4f} | {best['balanced_accuracy']:.4f} | {auc} | "
            f"{result['data']['n_examples']} | {result['data']['n_sound']} | {result['data']['n_noise']} |"
        )
    (out_dir / f"{run_name}_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    labels = [display_tag(result) for result in results]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].bar(labels, [result["best"]["macro_f1"] for result in results], color=["#4477aa", "#66aa55", "#cc6677"])
    axes[0].set_title("Best downstream macro-F1")
    axes[0].set_ylabel("macro-F1")
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis="y", alpha=0.25)
    for result in results:
        layer_names = list(result["per_layer"].keys())
        x = np.arange(len(layer_names))
        y = [result["per_layer"][layer]["macro_f1"] for layer in layer_names]
        axes[1].plot(x, y, marker="o", linewidth=1.8, label=display_tag(result))
    axes[1].set_title("Macro-F1 by animal2vec layer")
    axes[1].set_xlabel("layer")
    axes[1].set_ylabel("macro-F1")
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    if results:
        layer_names = list(results[0]["per_layer"].keys())
        step = max(1, len(layer_names) // 8)
        ticks = np.arange(0, len(layer_names), step)
        axes[1].set_xticks(ticks)
        axes[1].set_xticklabels([layer_names[i] for i in ticks], rotation=30, ha="right")
    fig.suptitle(f"animal2vec {run_name} validation")
    fig.tight_layout()
    fig.savefig(out_dir / f"{run_name}_validation.png", dpi=160)
    plt.close(fig)


def common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("checkpoints", nargs="+", type=Path, help="animal2vec checkpoint .pt/.pth/.ckpt/.zip")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--run-name", default=None, help="name used in output filenames")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--max-examples-per-class", type=int, default=400)
    parser.add_argument(
        "--selection-window-seconds",
        type=float,
        default=5.0,
        help="Used only to select comparable event centers before each checkpoint reads 80000 samples.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    folder = sub.add_parser("folder", help="whole-file labels from separate sound/noise folders")
    common_args(folder)
    folder.add_argument("--data-dir", type=Path, required=True)
    folder.add_argument("--sound-dir", default="Signal_parts", help="folder under --data-dir with target sounds")
    folder.add_argument("--noise-dir", default="Noise_parts", help="folder under --data-dir with noise/background")
    folder.add_argument("--windows-per-file", type=int, default=2)
    folder.add_argument("--max-files-per-class", type=int, default=0, help="0 means use all files")

    js = sub.add_parser("json", help="timeline labels from annotation JSON")
    common_args(js)
    js.add_argument("--data-dir", type=Path, required=True, help="folder with wav files")
    js.add_argument("--annotation", type=Path, required=True, help="annotations.json / Label Studio export")
    js.add_argument("--positive-label", default="sound", help="label treated as target whale sound")
    js.add_argument("--include-unmarked", action=argparse.BooleanOptionalAction, default=True)
    js.add_argument("--max-unmarked-per-file", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.out_dir = args.out_dir.resolve()
    if getattr(args, "annotation", None):
        args.annotation = args.annotation.resolve()

    run_name = args.run_name
    if not run_name:
        run_name = args.annotation.stem if args.mode == "json" else args.data_dir.name
        run_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_name).strip("_")

    # IMPORTANT:
    # Candidate centers are selected once, before loading checkpoints. This keeps
    # comparisons fair: 16k and 8k checkpoints see windows centered on the same
    # events, even though their seconds-per-window differ.
    if args.mode == "folder":
        candidates, dataset_meta = folder_candidates(args, args.selection_window_seconds)
    else:
        candidates, dataset_meta = json_candidates(args, args.selection_window_seconds)
    selected = balance_candidates(candidates, args.max_examples_per_class)

    results = [validate_checkpoint(ckpt, args, selected, dataset_meta, run_name) for ckpt in args.checkpoints]
    write_outputs(results, args.out_dir / run_name, run_name)


if __name__ == "__main__":
    main()
