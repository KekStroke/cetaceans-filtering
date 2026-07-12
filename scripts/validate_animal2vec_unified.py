#!/usr/bin/env python3
"""
Dominica click-ToA validation for animal2vec checkpoints.

What this script checks
-----------------------
It freezes animal2vec, extracts per-layer embeddings, then trains a simple
linear probe (StandardScaler + LogisticRegression). The checkpoint weights are
never updated.

This script keeps the full temporal sequence of animal2vec embeddings, creates
one click/no-click label per embedding frame from Dominica Click time of arrivals
(ToA), splits train/test by audio files, and reports frame-level plus event-level
tolerance metrics.

Important timing assumption checked for the current checkpoints:
  8 kHz, 80000 samples -> 10 s -> 2000 embeddings -> 200 Hz -> 5 ms/frame.
  16 kHz, 80000 samples -> 5 s -> 2000 embeddings -> 400 Hz -> 2.5 ms/frame.
The embeddings are emitted in chronological order.

Dominica CSV format:
  Recording name,Click time of arrivals (ToA) [sec]
  SW_10_filtered.wav,0.2840816

Example:
  python scripts/validate_animal2vec_unified.py CHECKPOINT1.pt CHECKPOINT2.pt ^
    --data-dir data/Dominica_dataset/Signal_parts ^
    --annotation path/to/Annotations_Dominica.csv ^
    --hop-seconds 5 ^
    --out-dir outputs/animal2vec/unified/dominica_toa

Input window size
-----------------
The script keeps the animal2vec training-compatible input length:
  80000 samples from the checkpoint config.
So a 16 kHz checkpoint receives 5 seconds; an 8 kHz checkpoint receives 10 seconds.
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
from typing import Any, Iterator, Sequence

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


def embed_layer_sequences(model: torch.nn.Module, wav: np.ndarray, device: str) -> list[np.ndarray]:
    """Return per-layer frame sequences as T x C arrays, preserving time order."""
    with torch.inference_mode():
        x = torch.from_numpy(wav).view(1, -1).to(device)
        out = model(source=x, features_only=True, mask=False)
        feats: list[np.ndarray] = []
        for layer_result in out.get("layer_results") or []:
            tensor = layer_result[0] if isinstance(layer_result, (tuple, list)) else layer_result
            if tensor.dim() == 3 and tensor.shape[0] == 1:
                seq = tensor[0]
            elif tensor.dim() == 3 and tensor.shape[1] == 1:
                seq = tensor[:, 0]
            else:
                raise RuntimeError(f"unexpected layer result shape: {tuple(tensor.shape)}")
            feats.append(seq.float().cpu().numpy())
        feats.append(out["x"][0].float().cpu().numpy())
        return feats


def check_embedding_chronology(
    model: torch.nn.Module,
    model_sr: int,
    target_samples: int,
    device: str,
    layer_name: str,
) -> dict[str, Any]:
    """Verify temporal ordering by moving one impulse through a silent window."""
    silence = np.zeros(target_samples, dtype=np.float32)
    baseline_layers = embed_layer_sequences(model, silence, device)
    layer_idx = layer_index_from_name(layer_name, len(baseline_layers))
    baseline = baseline_layers[layer_idx]
    n_frames = int(baseline.shape[0])
    impulse_samples: list[int] = []
    peak_frames: list[int] = []

    for fraction in (0.1, 0.3, 0.5, 0.7, 0.9):
        sample_idx = min(int(round(fraction * target_samples)), target_samples - 1)
        wav = silence.copy()
        wav[sample_idx] = 1.0
        sequence = embed_layer_sequences(model, wav, device)[layer_idx]
        response = np.linalg.norm(sequence - baseline, axis=1)
        impulse_samples.append(sample_idx)
        peak_frames.append(int(np.argmax(response)))

    monotonic = all(a < b for a, b in zip(peak_frames, peak_frames[1:]))
    correlation = float(np.corrcoef(impulse_samples, peak_frames)[0, 1])
    samples_per_embedding = float(target_samples / n_frames)
    result = {
        "chronological": bool(monotonic and correlation > 0.99),
        "ordering_correlation": correlation,
        "input_samples": int(target_samples),
        "output_embeddings": n_frames,
        "samples_per_embedding": samples_per_embedding,
        "embedding_seconds": float(samples_per_embedding / model_sr),
        "impulse_samples": impulse_samples,
        "peak_embedding_indices": peak_frames,
    }
    log(
        "chronology check: "
        f"chronological={result['chronological']}, input={target_samples}, output={n_frames}, "
        f"samples/embedding={samples_per_embedding:.3f}, impulses={impulse_samples}, peaks={peak_frames}"
    )
    return result


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


def _numeric_values(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, str):
        try:
            return [float(value)]
        except ValueError:
            return []
    if isinstance(value, list):
        out: list[float] = []
        for item in value:
            out.extend(_numeric_values(item))
        return out
    return []


def _toa_from_region(region: dict[str, Any], toa_label: str, toa_from: str) -> list[float]:
    """Extract point click times from either explicit ToA fields or labeled intervals."""
    explicit_keys = (
        "Click time of arrivals (ToA) [sec]",
        "Click time of arrivals",
        "ToA",
        "toa",
        "time_of_arrival",
        "time_of_arrivals",
        "click_time",
        "click_times",
        "time",
        "times",
    )
    for key in explicit_keys:
        values = _numeric_values(region.get(key))
        if values:
            return values

    labels = {str(label).lower() for label in region.get("labels", [])}
    if toa_label.lower() not in labels:
        return []
    try:
        start = float(region["start"])
        end = float(region["end"])
    except Exception:
        return []
    if end < start:
        start, end = end, start
    if toa_from == "start":
        return [start]
    if toa_from == "end":
        return [end]
    return [(start + end) / 2.0]


def load_toa_tasks(annotation_path: Path, toa_label: str, toa_from: str) -> list[dict[str, Any]]:
    if annotation_path.suffix.lower() == ".csv":
        grouped: dict[str, list[float]] = {}
        with annotation_path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            name_col = "Recording name"
            toa_col = "Click time of arrivals (ToA) [sec]"
            if name_col not in (reader.fieldnames or []) or toa_col not in (reader.fieldnames or []):
                raise ValueError(f"{annotation_path} must contain {name_col!r} and {toa_col!r}")
            for row in reader:
                try:
                    grouped.setdefault(row[name_col], []).append(float(row[toa_col]))
                except Exception:
                    continue
        return [
            {"audio": audio, "toa": sorted({round(float(t), 6) for t in times if float(t) >= 0.0})}
            for audio, times in sorted(grouped.items())
            if times
        ]

    if annotation_path.suffix.lower() == ".mat":
        import scipy.io

        mat = scipy.io.loadmat(annotation_path, squeeze_me=True, struct_as_record=False)
        table = mat.get("Annotations_Dominica")
        if table is None:
            raise ValueError(f"{annotation_path} does not contain Annotations_Dominica")
        tasks = []
        for row in table[1:]:
            audio = str(row[0])
            toa = np.asarray(row[1], dtype=np.float64).reshape(-1)
            toa = sorted({round(float(t), 6) for t in toa if float(t) >= 0.0})
            if toa:
                tasks.append({"audio": audio, "toa": toa})
        return tasks

    tasks = []
    for item in load_annotation_tasks(annotation_path):
        times: list[float] = []
        for interval in item["intervals"]:
            times.extend(_toa_from_region(interval, toa_label, toa_from))
        times = sorted({round(float(t), 6) for t in times if float(t) >= 0.0})
        if times:
            tasks.append({"audio": item["audio"], "toa": times})
    return tasks


def resolve_toa_dataset(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import soundfile as sf

    tasks = load_toa_tasks(args.annotation, args.toa_label, args.toa_from)
    wavs = list(args.data_dir.rglob("*.wav"))
    wavs_by_name = {p.name: p for p in wavs}
    wavs_by_name.update({normalize_audio_name(p.name): p for p in wavs})

    resolved = []
    missing = []
    for task in tasks:
        key = normalize_audio_name(task["audio"])
        path = wavs_by_name.get(key)
        if path is None:
            missing.append(task["audio"])
            continue
        duration = float(sf.info(str(path)).duration)
        toa = [float(t) for t in task["toa"] if 0.0 <= float(t) <= duration]
        if toa:
            resolved.append({"audio": task["audio"], "path": path, "toa": toa, "duration": duration})
    if not resolved:
        raise SystemExit(f"No ToA annotation audio names matched wav files under {args.data_dir}")
    return resolved, {
        "layout": "toa",
        "annotation": str(args.annotation),
        "matched_audio_files": len(resolved),
        "missing_audio_files": len(missing),
        "missing_audio_examples": missing[:20],
        "n_toa": int(sum(len(task["toa"]) for task in resolved)),
    }


def split_toa_files(tasks: list[dict[str, Any]], test_fraction: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    shuffled = tasks[:]
    rng.shuffle(shuffled)
    n_test = max(1, int(round(len(shuffled) * test_fraction)))
    n_test = min(n_test, len(shuffled) - 1)
    return shuffled[n_test:], shuffled[:n_test]


def chunk_starts(duration_s: float, window_s: float, hop_s: float) -> list[float]:
    if duration_s <= window_s:
        return [0.0]
    starts = [float(x) for x in np.arange(0.0, duration_s - window_s + 1e-9, hop_s)]
    last = duration_s - window_s
    if not starts or abs(starts[-1] - last) > 1e-6:
        starts.append(float(last))
    return starts


def frame_labels_for_toa(toa: Sequence[float], start_s: float, window_s: float, n_frames: int) -> np.ndarray:
    labels = np.zeros(n_frames, dtype=np.int64)
    fps = n_frames / window_s
    end_s = start_s + window_s
    for t in toa:
        if start_s <= float(t) < end_s:
            idx = int(math.floor((float(t) - start_s) * fps))
            labels[min(max(idx, 0), n_frames - 1)] = 1
    return labels


def layer_index_from_name(layer: str, n_layers: int) -> int:
    if layer == "final":
        return n_layers - 1
    if re.fullmatch(r"L\d+", layer):
        idx = int(layer[1:])
        if 0 <= idx < n_layers - 1:
            return idx
    raise ValueError(f"Layer must be final or L0..L{n_layers - 2}; got {layer!r}")


def collect_toa_frames(
    model: torch.nn.Module,
    tasks: list[dict[str, Any]],
    model_sr: int,
    target_samples: int,
    device: str,
    layer_name: str,
    hop_s: float,
    split_name: str,
    max_negatives_per_positive: int | None,
    max_frames: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    rng = np.random.default_rng(42 if split_name == "train" else 43)
    window_s = target_samples / model_sr
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    meta: list[dict[str, Any]] = []
    n_chunks = 0
    n_positive = 0
    n_total_before_sampling = 0
    frame_dt_s: float | None = None
    selected_layer_idx: int | None = None

    for task in tasks:
        starts = chunk_starts(float(task["duration"]), window_s, hop_s)
        for start_s in starts:
            wav = read_audio_window(task["path"], start_s, window_s, model_sr, target_samples)
            seqs = embed_layer_sequences(model, wav, device)
            if selected_layer_idx is None:
                selected_layer_idx = layer_index_from_name(layer_name, len(seqs))
            seq = seqs[selected_layer_idx]
            frame_dt_s = window_s / seq.shape[0]
            labels = frame_labels_for_toa(task["toa"], start_s, window_s, seq.shape[0])
            indices = np.arange(seq.shape[0])
            positives = indices[labels == 1]
            negatives = indices[labels == 0]
            n_total_before_sampling += int(seq.shape[0])
            n_positive += int(len(positives))

            if max_negatives_per_positive is not None:
                n_neg = min(len(negatives), max(len(positives), 1) * max_negatives_per_positive)
                negatives = rng.choice(negatives, size=n_neg, replace=False) if n_neg < len(negatives) else negatives
                indices = np.sort(np.concatenate([positives, negatives]))

            if max_frames > 0:
                remaining = max_frames - sum(len(part) for part in y_parts)
                if remaining <= 0:
                    break
                if len(indices) > remaining:
                    indices = np.sort(rng.choice(indices, size=remaining, replace=False))

            X_parts.append(seq[indices])
            y_parts.append(labels[indices])
            fps = seq.shape[0] / window_s
            for idx in indices:
                meta.append({
                    "file": task["path"].name,
                    "time": float(start_s + idx / fps),
                    "frame": int(idx),
                    "chunk_start": float(start_s),
                })
            n_chunks += 1
        log(f"  {split_name}: embedded {task['path'].name} ({len(starts)} chunks)")
        if max_frames > 0 and sum(len(part) for part in y_parts) >= max_frames:
            break

    if not X_parts:
        raise RuntimeError(f"no {split_name} ToA frames collected")
    y = np.concatenate(y_parts)
    if len(np.unique(y)) < 2:
        raise RuntimeError(f"{split_name} split has one class only: positives={int(y.sum())}, frames={len(y)}")
    return np.vstack(X_parts), y, meta, {
        f"{split_name}_chunks": n_chunks,
        f"{split_name}_frames": int(len(y)),
        f"{split_name}_positive_frames": int(y.sum()),
        f"{split_name}_total_frames_before_sampling": n_total_before_sampling,
        f"{split_name}_positive_frames_before_sampling": n_positive,
        f"{split_name}_frame_seconds": float(frame_dt_s or 0.0),
    }


def event_metrics(
    test_tasks: list[dict[str, Any]],
    frame_meta: list[dict[str, Any]],
    y_score: np.ndarray,
    start_threshold: float,
    end_threshold: float,
    min_sound_s: float,
    min_gap_s: float,
    tolerance_s: float,
    frame_dt_s: float,
) -> dict[str, float | int]:
    true_by_file = {task["path"].name: list(task["toa"]) for task in test_tasks}
    scores_by_file: dict[str, list[tuple[float, float]]] = {}
    for item, score in zip(frame_meta, y_score):
        scores_by_file.setdefault(item["file"], []).append((float(item["time"]), float(score)))

    pred_events: dict[str, list[float]] = {}
    for file, points in scores_by_file.items():
        points = sorted(points)
        segments: list[list[tuple[float, float]]] = []
        active = False
        current: list[tuple[float, float]] = []
        for point in points:
            score = point[1]
            if not active and score >= start_threshold:
                active = True
                current = [point]
            elif active:
                current.append(point)
                if score < end_threshold:
                    segments.append(current)
                    active = False
                    current = []
        if active and current:
            segments.append(current)

        filtered: list[list[tuple[float, float]]] = []
        for segment in segments:
            duration = max(frame_dt_s, segment[-1][0] - segment[0][0] + frame_dt_s)
            if duration >= min_sound_s:
                filtered.append(segment)

        merged: list[list[tuple[float, float]]] = []
        for segment in filtered:
            if not merged:
                merged.append(segment)
                continue
            gap = segment[0][0] - merged[-1][-1][0] - frame_dt_s
            if gap < min_gap_s:
                merged[-1].extend(segment)
            else:
                merged.append(segment)

        pred_events[file] = [max(segment, key=lambda x: x[1])[0] for segment in merged]

    tp = fp = fn = 0
    for file, truth in true_by_file.items():
        preds = sorted(pred_events.get(file, []))
        truth = sorted(float(t) for t in truth)
        truth_i = 0
        for pred in preds:
            while truth_i < len(truth) and truth[truth_i] < pred - tolerance_s:
                fn += 1
                truth_i += 1
            if truth_i < len(truth) and abs(pred - truth[truth_i]) <= tolerance_s:
                tp += 1
                truth_i += 1
            else:
                fp += 1
        fn += len(truth) - truth_i
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "event_precision": float(precision),
        "event_recall": float(recall),
        "event_f1": float(f1),
        "event_tp": int(tp),
        "event_fp": int(fp),
        "event_fn": int(fn),
        "event_tolerance_s": float(tolerance_s),
        "event_start_threshold": float(start_threshold),
        "event_end_threshold": float(end_threshold),
        "event_min_sound_s": float(min_sound_s),
        "event_min_gap_s": float(min_gap_s),
        "predicted_events": int(sum(len(v) for v in pred_events.values())),
        "true_events": int(sum(len(v) for v in true_by_file.values())),
    }


def validate_toa_checkpoint(
    ckpt_arg: Path,
    args: argparse.Namespace,
    train_tasks: list[dict[str, Any]],
    test_tasks: list[dict[str, Any]],
    dataset_meta: dict[str, Any],
    run_name: str,
) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    with checkpoint_path(ckpt_arg.resolve()) as ckpt_file:
        model, ckpt_meta = load_feature_model(ckpt_file, device)
        chronology = check_embedding_chronology(
            model, ckpt_meta["sample_rate"], ckpt_meta["max_sample_size"], device, args.layer
        )
        if not chronology["chronological"]:
            raise RuntimeError(f"{ckpt_arg.name}: animal2vec embeddings failed chronological-order check")
        window_s = ckpt_meta["max_sample_size"] / ckpt_meta["sample_rate"]
        log(f"{run_name}: ToA validation with {window_s:.3f}s chunks, hop={args.hop_seconds}s, layer={args.layer}")
        X_train, y_train, _, train_meta = collect_toa_frames(
            model, train_tasks, ckpt_meta["sample_rate"], ckpt_meta["max_sample_size"], device,
            args.layer, args.hop_seconds, "train", args.max_train_negatives_per_positive, args.max_train_frames,
        )
        X_test, y_test, frame_meta, test_meta = collect_toa_frames(
            model, test_tasks, ckpt_meta["sample_rate"], ckpt_meta["max_sample_size"], device,
            args.layer, args.hop_seconds, "test", None, args.max_test_frames,
        )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"))
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    score = clf.predict_proba(X_test)[:, 1]
    frame_dt_s = float(test_meta["test_frame_seconds"])
    tolerance_s = frame_dt_s / 2.0
    frame_metrics: dict[str, float | None] = {
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "positive_f1": float(f1_score(y_test, pred, pos_label=1)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "average_precision": float(average_precision_score(y_test, score)),
    }
    try:
        frame_metrics["auc"] = float(roc_auc_score(y_test, score))
    except ValueError:
        frame_metrics["auc"] = None
    events = event_metrics(
        test_tasks,
        frame_meta,
        score,
        args.start_threshold,
        args.end_threshold,
        args.min_sound_seconds,
        args.min_gap_seconds,
        tolerance_s,
        frame_dt_s,
    )
    result = {
        "tag": ckpt_arg.stem,
        "checkpoint": str(ckpt_arg),
        "checkpoint_meta": ckpt_meta,
        "chronology": chronology,
        "task": f"{run_name}_toa_click_detection",
        "primary_metric": "event_f1",
        "layer": args.layer,
        "frame": frame_metrics,
        "event": events,
        "data": {
            **dataset_meta,
            **train_meta,
            **test_meta,
            "train_files": [task["path"].name for task in train_tasks],
            "test_files": [task["path"].name for task in test_tasks],
            "window_seconds": window_s,
            "hop_seconds": args.hop_seconds,
            "frame_seconds": frame_dt_s,
        },
    }
    log(
        f"{display_tag(result)} ToA: event-F1={events['event_f1']:.4f}, "
        f"frame macro-F1={frame_metrics['macro_f1']:.4f}"
    )
    del clf, X_train, X_test
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def write_toa_outputs(results: list[dict[str, Any]], out_dir: Path, run_name: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{run_name}_toa_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    rows = []
    for result in results:
        rows.append({
            "checkpoint": display_tag(result),
            "sample_rate": result["checkpoint_meta"]["sample_rate"],
            "layer": result["layer"],
            "event_f1": result["event"]["event_f1"],
            "event_precision": result["event"]["event_precision"],
            "event_recall": result["event"]["event_recall"],
            "frame_macro_f1": result["frame"]["macro_f1"],
            "frame_positive_f1": result["frame"]["positive_f1"],
            "balanced_accuracy": result["frame"]["balanced_accuracy"],
            "average_precision": result["frame"]["average_precision"],
            "auc": result["frame"]["auc"],
            "train_files": len(result["data"]["train_files"]),
            "test_files": len(result["data"]["test_files"]),
            "train_frames": result["data"]["train_frames"],
            "test_frames": result["data"]["test_frames"],
            "tolerance_s": result["event"]["event_tolerance_s"],
            "start_threshold": result["event"]["event_start_threshold"],
            "end_threshold": result["event"]["event_end_threshold"],
            "min_sound_s": result["event"]["event_min_sound_s"],
            "min_gap_s": result["event"]["event_min_gap_s"],
            "chronological": result["chronology"]["chronological"],
            "input_samples": result["chronology"]["input_samples"],
            "output_embeddings": result["chronology"]["output_embeddings"],
            "samples_per_embedding": result["chronology"]["samples_per_embedding"],
            "embedding_ms": result["chronology"]["embedding_seconds"] * 1000.0,
            "impulse_samples": json.dumps(result["chronology"]["impulse_samples"]),
            "peak_embedding_indices": json.dumps(result["chronology"]["peak_embedding_indices"]),
        })
    with (out_dir / f"{run_name}_toa_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    chronology_fields = [
        "checkpoint", "chronological", "input_samples", "output_embeddings",
        "samples_per_embedding", "embedding_ms", "impulse_samples", "peak_embedding_indices",
    ]
    with (out_dir / f"{run_name}_chronology_check.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=chronology_fields)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in chronology_fields} for row in rows])

    labels = [row["checkpoint"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(labels, [row["event_f1"] for row in rows], color="#4477aa")
    axes[0].set_title("Event F1 with tolerance")
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(labels, [row["frame_positive_f1"] for row in rows], color="#66aa55")
    axes[1].set_title("Frame positive F1")
    axes[1].set_ylim(0, 1)
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle(f"animal2vec {run_name} ToA validation")
    fig.tight_layout()
    fig.savefig(out_dir / f"{run_name}_toa_validation.png", dpi=160)
    plt.close(fig)

    table_columns = [
        "checkpoint", "event_f1", "event_precision", "event_recall",
        "frame_positive_f1", "average_precision", "tolerance_s",
    ]
    table_labels = ["Checkpoint", "Event F1", "Precision", "Recall", "Frame F1", "AP", "Tolerance, s"]
    table_data = []
    for row in rows:
        table_data.append([
            row["checkpoint"],
            f"{row['event_f1']:.4f}",
            f"{row['event_precision']:.4f}",
            f"{row['event_recall']:.4f}",
            f"{row['frame_positive_f1']:.4f}",
            f"{row['average_precision']:.4f}",
            f"{row['tolerance_s']:.5f}",
        ])
    table_fig, table_ax = plt.subplots(figsize=(11.5, 1.4 + 0.55 * len(table_data)))
    table_ax.axis("off")
    table = table_ax.table(cellText=table_data, colLabels=table_labels, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.55)
    for (row_idx, _), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor("#4477aa")
            cell.set_text_props(color="white", weight="bold")
        elif row_idx % 2 == 0:
            cell.set_facecolor("#eef3f8")
    table_fig.tight_layout()
    table_fig.savefig(out_dir / f"{run_name}_toa_summary_table.png", dpi=180, bbox_inches="tight")
    plt.close(table_fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", nargs="+", type=Path, help="animal2vec checkpoint .pt/.pth/.ckpt/.zip")
    parser.add_argument("--data-dir", type=Path, required=True, help="Dominica wav folder, usually data/Dominica_dataset/Signal_parts")
    parser.add_argument("--annotation", type=Path, required=True, help="Dominica CSV/MAT with Click time of arrivals (ToA) [sec]")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--run-name", default="dominica_toa", help="name used in output filenames")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--layer", default="final", help="animal2vec layer to use: final or L0/L1/...")
    parser.add_argument("--hop-seconds", type=float, default=5.0, help="chunk hop; 5..9 seconds is typical for 10s/8k windows")
    parser.add_argument("--test-fraction", type=float, default=0.25, help="train/test split by audio files")
    parser.add_argument("--start-threshold", type=float, default=0.8, help="probability threshold that starts a predicted click segment")
    parser.add_argument("--end-threshold", type=float, default=0.5, help="probability threshold below which a predicted segment ends")
    parser.add_argument("--min-sound-seconds", type=float, default=0.0, help="drop predicted sound segments shorter than this")
    parser.add_argument("--min-gap-seconds", type=float, default=0.0, help="merge predicted segments separated by a shorter gap")
    parser.add_argument("--toa-label", default="clicks", help="only used for JSON interval fallback; Dominica CSV uses explicit ToA")
    parser.add_argument("--toa-from", choices=("center", "start", "end"), default="center")
    parser.add_argument("--max-train-negatives-per-positive", type=int, default=3,
                        help="sampled train negatives per positive; all positive frames are retained")
    parser.add_argument("--max-train-frames", type=int, default=0, help="0 means no cap after train negative sampling")
    parser.add_argument("--max-test-frames", type=int, default=0, help="0 means evaluate all collected test frames")
    parser.add_argument("--split-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.out_dir = args.out_dir.resolve()
    if getattr(args, "annotation", None):
        args.annotation = args.annotation.resolve()

    run_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.run_name).strip("_")
    tasks, dataset_meta = resolve_toa_dataset(args)
    train_tasks, test_tasks = split_toa_files(tasks, args.test_fraction, args.split_seed)
    log(
        f"ToA dataset: {len(train_tasks)} train files, {len(test_tasks)} test files, "
        f"{dataset_meta['n_toa']} clicks"
    )
    results = []
    out_dir = args.out_dir / run_name
    for ckpt in args.checkpoints:
        results.append(validate_toa_checkpoint(ckpt, args, train_tasks, test_tasks, dataset_meta, run_name))
        write_toa_outputs(results, out_dir, run_name)


if __name__ == "__main__":
    main()
