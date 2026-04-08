from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .types import SplitData


def save_split_manifest(split: SplitData, output_dir: Path) -> None:
    split.manifest.to_csv(output_dir / f"{split.name}_manifest.csv", index=False)


def save_json(obj: dict, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def plot_macro_f1_summary(metrics: dict, output_dir: Path, primary_file_method: str) -> Path:
    splits, window_vals, file_vals = [], [], []
    for split_name in ("train", "val", "test"):
        split_metrics = metrics.get(split_name)
        if split_metrics is None:
            continue
        splits.append(split_name)
        window_vals.append(float(split_metrics["window"]["macro_f1"]))
        file_vals.append(float(split_metrics["file"][primary_file_method]["macro_f1"]))

    x = np.arange(len(splits))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, window_vals, width, label="window_macro_f1")
    ax.bar(
        x + width / 2, file_vals, width, label=f"file_{primary_file_method}_macro_f1"
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Macro-F1 by split")
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.legend()
    fig.tight_layout()
    out_path = output_dir / "macro_f1_by_split.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_confusion_matrix_image(
    cm: list[list[int]], class_names: list[str], title: str, out_path: Path
) -> Path:
    cm_arr = np.asarray(cm, dtype=float)
    row_sums = cm_arr.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.divide(cm_arr, row_sums, where=row_sums > 0)
    cm_norm = np.nan_to_num(cm_norm)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    ax.set_yticklabels(class_names, fontsize=6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def create_report_images(
    metrics: dict,
    class_names: list[str],
    output_dir: Path,
    primary_file_method: str,
) -> dict[str, str]:
    report_dir = output_dir / "report_images"
    report_dir.mkdir(parents=True, exist_ok=True)
    image_paths: dict[str, str] = {
        "macro_f1_by_split": str(
            plot_macro_f1_summary(metrics, report_dir, primary_file_method)
        )
    }
    for split_name in ("train", "val", "test"):
        split_metrics = metrics.get(split_name)
        if split_metrics is None:
            continue
        cm_window = split_metrics["window"]["confusion_matrix"]
        cm_file = split_metrics["file"][primary_file_method]["confusion_matrix"]
        window_path = report_dir / f"{split_name}_window_confusion_matrix.png"
        file_path = report_dir / f"{split_name}_file_{primary_file_method}_confusion_matrix.png"
        image_paths[f"{split_name}_window_confusion_matrix"] = str(
            plot_confusion_matrix_image(
                cm_window,
                class_names,
                f"{split_name.capitalize()} window confusion matrix",
                window_path,
            )
        )
        image_paths[f"{split_name}_file_confusion_matrix"] = str(
            plot_confusion_matrix_image(
                cm_file,
                class_names,
                f"{split_name.capitalize()} file confusion matrix ({primary_file_method})",
                file_path,
            )
        )
    return image_paths
