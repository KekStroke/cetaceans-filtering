"""Build a compact report for a Voxaboxen experiment."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import hydra
import matplotlib.pyplot as plt
import yaml
from omegaconf import DictConfig


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _load_yaml(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _flatten(prefix: str, value: Any, rows: list[dict[str, str]]) -> None:
    if value is None:
        return
    if isinstance(value, dict):
        for key, nested in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            _flatten(name, nested, rows)
        return
    rows.append({"metric": prefix, "value": str(value)})


def _train_losses(train_history: Any) -> list[tuple[int, float]]:
    losses: list[tuple[int, float]] = []
    if not isinstance(train_history, dict):
        return losses
    for epoch, values in train_history.items():
        if not isinstance(values, dict) or "loss" not in values:
            continue
        try:
            losses.append((int(epoch), float(values["loss"])))
        except (TypeError, ValueError):
            continue
    return sorted(losses)


def _write_loss_plot(losses: list[tuple[int, float]], path: Path) -> None:
    if not losses:
        return
    epochs = [epoch for epoch, _ in losses]
    values = [loss for _, loss in losses]
    fig, ax = plt.subplots(figsize=(7, 4), dpi=140)
    ax.plot(epochs, values, marker="o", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train loss")
    ax.set_title("Voxaboxen training loss")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_metrics_csv(report_dir: Path, val_results: Any, test_results: Any) -> None:
    rows: list[dict[str, str]] = []
    _flatten("val", val_results, rows)
    _flatten("test", test_results, rows)
    with (report_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def _artifact_rows(experiment_dir: Path, report_dir: Path) -> list[tuple[str, str]]:
    candidates = [
        ("Training history", experiment_dir / "train_history.yaml"),
        ("Validation metrics", experiment_dir / "val_results.yaml"),
        ("Test metrics", experiment_dir / "test_results.yaml"),
        ("Full test results", experiment_dir / "test_full_results.json"),
        ("Training progress", experiment_dir / "train_progress.svg"),
        ("Loss plot", report_dir / "loss.png"),
        ("Metrics CSV", report_dir / "metrics.csv"),
    ]
    rows: list[tuple[str, str]] = []
    for label, path in candidates:
        if path.exists():
            rows.append((label, path.relative_to(experiment_dir).as_posix()))
    return rows


def _short_metrics(results: Any) -> list[str]:
    if not isinstance(results, dict):
        return []
    keys = [
        "mean_ap@0.5",
        "mean_ap@0.8",
        "macro-f1@0.5",
        "macro-f1@0.8",
        "micro-f1@0.5",
        "micro-f1@0.8",
    ]
    return [f"- `{key}`: {results[key]}" for key in keys if key in results]


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(config: DictConfig) -> None:
    cfg = config.voxaboxen
    project_root = Path(hydra.utils.get_original_cwd())
    project_dir = Path(hydra.utils.to_absolute_path(str(cfg.output_project_dir)))
    experiment_dir = project_dir / str(cfg.experiment_name)
    report_dir = experiment_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    train_history = _load_yaml(experiment_dir / "train_history.yaml")
    val_results = _load_yaml(experiment_dir / "val_results.yaml")
    test_results = _load_yaml(experiment_dir / "test_results.yaml")

    losses = _train_losses(train_history)
    _write_loss_plot(losses, report_dir / "loss.png")
    _write_metrics_csv(report_dir, val_results, test_results)
    if (experiment_dir / "train_progress.svg").exists():
        shutil.copy2(
            experiment_dir / "train_progress.svg",
            report_dir / "train_progress.svg",
        )

    report_lines = [
        f"# Voxaboxen Report: {cfg.experiment_name}",
        "",
        "## Run",
        "",
        f"- Dataset: `{cfg.dataset_name}`",
        f"- Experiment: `{cfg.experiment_name}`",
        f"- Experiment directory: `{_display_path(experiment_dir, project_root)}`",
        f"- Report directory: `{_display_path(report_dir, project_root)}`",
        "",
    ]
    if losses:
        report_lines.extend(["## Train Loss", ""])
        report_lines.extend(f"- epoch {epoch}: {loss:.6f}" for epoch, loss in losses)
        report_lines.extend(["", "![Loss](loss.png)", ""])
    else:
        report_lines.extend(
            [
                "## Train Loss",
                "",
                "No `train_history.yaml` with loss values was found for this experiment.",
                "",
            ]
        )

    if val_results:
        report_lines.extend(["## Validation Metrics", ""])
        report_lines.extend(
            _short_metrics(val_results) or ["See `report/metrics.csv`."]
        )
        report_lines.append("")
    else:
        report_lines.extend(
            ["## Validation Metrics", "", "No `val_results.yaml` was found.", ""]
        )

    if test_results:
        report_lines.extend(["## Test Metrics", ""])
        report_lines.extend(_short_metrics(test_results) or ["See `report/metrics.csv`."])
        report_lines.append("")
    else:
        report_lines.extend(["## Test Metrics", "", "No `test_results.yaml` was found.", ""])

    if (report_dir / "train_progress.svg").exists():
        report_lines.extend(
            ["## Training Progress", "", "![Training progress](train_progress.svg)", ""]
        )

    report_lines.extend(["## Artifacts", "", "| Artifact | Path |", "| --- | --- |"])
    for label, path in _artifact_rows(experiment_dir, report_dir):
        report_lines.append(f"| {label} | `{path}` |")

    (report_dir / "report.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print(f"Experiment: {experiment_dir}")
    print(f"Report: {report_dir / 'report.md'}")
    if (report_dir / "loss.png").exists():
        print(f"Loss plot: {report_dir / 'loss.png'}")
    print(f"Metrics CSV: {report_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
