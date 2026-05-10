"""Prepare Label Studio audio annotations for Voxaboxen.

This script writes Raven-style selection tables plus train/val/test info CSVs.
It does not copy audio; generated CSVs point to the original audio files.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf


def _labels(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [str(x) for x in raw if x is not None and str(x)]
    return [str(raw)]


def _strip_upload_prefix(name: str) -> str:
    # Label Studio exports often look like "a85ee650-audio.wav".
    suffix = name.split("-", 1)[-1]
    return suffix or name


def _audio_keys(item: dict[str, Any]) -> set[str]:
    candidates = []
    data = item.get("data")
    if isinstance(data, dict):
        candidates.append(data.get("audio"))
    candidates.extend([item.get("audio"), item.get("file_upload")])

    keys: set[str] = set()
    for raw in candidates:
        if not raw:
            continue
        name = str(raw).replace("\\", "/")
        basename = Path(name).name
        stripped = _strip_upload_prefix(basename)
        keys.update({name, basename, stripped, _strip_upload_prefix(name)})
    return {key for key in keys if key}


def _annotation_index(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_name: dict[str, dict[str, Any]] = {}
    for item in items:
        for key in _audio_keys(item):
            by_name[key] = item
    return by_name


def _events_from_item(item: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not item:
        return []
    direct = item.get("label")
    if isinstance(direct, list):
        return [event for event in direct if isinstance(event, dict)]

    events: list[dict[str, Any]] = []
    annotations = item.get("annotations", [])
    if not isinstance(annotations, list):
        return events
    for annotation in annotations:
        if not isinstance(annotation, dict) or annotation.get("was_cancelled"):
            continue
        results = annotation.get("result", [])
        if not isinstance(results, list):
            continue
        for result in results:
            if not isinstance(result, dict):
                continue
            value = result.get("value", {})
            if not isinstance(value, dict):
                continue
            if "start" not in value or "end" not in value:
                continue
            events.append(
                {
                    "start": value.get("start"),
                    "end": value.get("end"),
                    "labels": value.get("labels") or value.get("choices"),
                }
            )
    return events


def _write_selection_table(
    target: Path,
    events: list[dict[str, Any]],
    allowed_labels: set[str],
    unknown_labels: set[str],
    unknown_label: str,
) -> dict[str, int]:
    target.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    rows: list[dict[str, str | float]] = []

    for event in events:
        if not isinstance(event, dict):
            continue
        start = float(event.get("start", 0.0))
        end = float(event.get("end", 0.0))
        if end <= start:
            continue
        labels = _labels(event.get("labels")) or [unknown_label]
        label = labels[0]
        if label in unknown_labels:
            counts["background_from_unknown"] = counts.get("background_from_unknown", 0) + 1
            continue
        if label not in allowed_labels:
            counts["background_from_unknown"] = counts.get("background_from_unknown", 0) + 1
            continue
        counts[label] = counts.get(label, 0) + 1
        rows.append(
            {
                "Begin Time (s)": start,
                "End Time (s)": end,
                "Annotation": label,
            }
        )

    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Begin Time (s)", "End Time (s)", "Annotation"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)
    return counts


def _write_info_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["fn", "audio_fp", "selection_table_fp"])
        writer.writeheader()
        writer.writerows(rows)


def _write_project_config(cfg: DictConfig, train_fp: Path, val_fp: Path, test_fp: Path) -> None:
    project_dir = Path(hydra.utils.to_absolute_path(str(cfg.output_project_dir)))
    project_dir.mkdir(parents=True, exist_ok=True)
    positive_labels = [str(x) for x in cfg.positive_labels]
    unknown_labels = [str(x) for x in cfg.unknown_labels]
    label_mapping = {str(cfg.unknown_label): str(cfg.unknown_label)}
    for label in unknown_labels:
        label_mapping[label] = str(cfg.unknown_label)
    for label in positive_labels:
        label_mapping[label] = str(cfg.binary_label)

    project_config = {
        "data_dir": str(Path(hydra.utils.to_absolute_path(str(cfg.output_dataset_dir)))),
        "project_dir": str(project_dir),
        "train_info_fp": str(train_fp),
        "val_info_fp": str(val_fp),
        "test_info_fp": str(test_fp),
        "unknown_label": str(cfg.unknown_label),
        "label_mapping": label_mapping,
        "label_set": [str(cfg.binary_label)],
    }
    with (project_dir / "project_config.yaml").open("w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(OmegaConf.create(project_config), sort_keys=False))


def _split_rows(rows: list[dict[str, str]], train_fraction: float, val_fraction: float, seed: int):
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    n_total = len(shuffled)
    n_train = max(1, int(round(n_total * train_fraction)))
    n_val = int(round(n_total * val_fraction))
    if n_total >= 3:
        n_train = min(n_train, n_total - 2)
        n_val = max(1, min(n_val, n_total - n_train - 1))
    test_start = n_train + n_val
    return shuffled[:n_train], shuffled[n_train:test_start], shuffled[test_start:]


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(config: DictConfig) -> None:
    cfg = config.voxaboxen
    audio_dir = Path(hydra.utils.to_absolute_path(str(cfg.audio_dir)))
    annotations_fp = Path(hydra.utils.to_absolute_path(str(cfg.annotations_json)))
    out_dir = Path(hydra.utils.to_absolute_path(str(cfg.output_dataset_dir)))
    selection_dir = out_dir / "selection_tables"

    audio_files = sorted(audio_dir.glob(str(cfg.audio_glob)))
    if not audio_files:
        raise ValueError(f"No audio files matched {cfg.audio_glob!r} in {audio_dir}")

    with annotations_fp.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError("Expected Label Studio annotations JSON as a top-level list.")

    annotations = _annotation_index(payload)
    allowed_labels = set(str(x) for x in cfg.positive_labels) | set(str(x) for x in cfg.unknown_labels)
    unknown_labels = set(str(x) for x in cfg.unknown_labels)

    rows: list[dict[str, str]] = []
    summary_rows: list[dict[str, str | int]] = []
    for audio_fp in audio_files:
        item = annotations.get(audio_fp.name)
        events = _events_from_item(item)
        fn = audio_fp.stem
        table_fp = selection_dir / f"{fn}.txt"
        counts = _write_selection_table(
            table_fp,
            events,
            allowed_labels=allowed_labels,
            unknown_labels=unknown_labels,
            unknown_label=str(cfg.unknown_label),
        )
        rows.append(
            {
                "fn": fn,
                "audio_fp": str(audio_fp.resolve()),
                "selection_table_fp": str(table_fp.resolve()),
            }
        )
        summary = {"fn": fn, "events": sum(counts.values())}
        summary.update(counts)
        summary_rows.append(summary)

    train_rows, val_rows, test_rows = _split_rows(
        rows,
        train_fraction=float(cfg.train_fraction),
        val_fraction=float(cfg.val_fraction),
        seed=int(cfg.seed),
    )

    all_fp = out_dir / "all_info.csv"
    train_fp = out_dir / "train_info.csv"
    val_fp = out_dir / "val_info.csv"
    test_fp = out_dir / "test_info.csv"
    _write_info_csv(all_fp, rows)
    _write_info_csv(train_fp, train_rows)
    _write_info_csv(val_fp, val_rows)
    _write_info_csv(test_fp, test_rows)

    summary_fp = out_dir / "summary.csv"
    keys = sorted({k for row in summary_rows for k in row.keys()})
    with summary_fp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summary_rows)

    _write_project_config(cfg, train_fp=train_fp.resolve(), val_fp=val_fp.resolve(), test_fp=test_fp.resolve())

    print(f"Audio files: {len(rows)}")
    print(f"Train/val/test: {len(train_rows)}/{len(val_rows)}/{len(test_rows)}")
    print(f"Dataset: {out_dir.resolve()}")
    print(f"Project config: {Path(hydra.utils.to_absolute_path(str(cfg.output_project_dir))) / 'project_config.yaml'}")


if __name__ == "__main__":
    main()
