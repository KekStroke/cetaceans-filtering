"""SED binary classifier training pipeline (window-level only)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .data import (
    SplitData,
    build_split,
    get_file_labels,
    load_and_label,
    split_files,
)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def build_model(args) -> Pipeline:
    steps = []
    if not args.no_scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(
        (
            "clf",
            LogisticRegression(
                max_iter=args.max_iter,
                class_weight="balanced",
                random_state=int(args.random_state),
            ),
        )
    )
    return Pipeline(steps)


# ---------------------------------------------------------------------------
# Metrics & evaluation
# ---------------------------------------------------------------------------


def compute_metrics(y_true, y_pred, class_names: list[str]) -> dict:
    labels = np.arange(len(class_names))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=class_names,
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def evaluate_split(split: SplitData, model: Pipeline, encoder: LabelEncoder) -> dict:
    class_names = encoder.classes_.tolist()
    pred = model.predict(split.X)
    return {
        "n_windows": len(split.y),
        "n_files": int(split.manifest["filename"].nunique()),
        **compute_metrics(split.y, pred, class_names),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _save_json(obj: dict, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _plot_macro_f1(metrics: dict, out_dir: Path) -> Path:
    splits, vals = [], []
    for s in ("train", "val", "test"):
        m = metrics.get(s)
        if m is None:
            continue
        splits.append(s)
        vals.append(float(m["macro_f1"]))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(splits, vals)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro-F1")
    ax.set_title("SED Binary Classifier — Window Macro-F1")
    fig.tight_layout()
    p = out_dir / "macro_f1_by_split.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def _plot_cm(cm, class_names, title, out_path) -> Path:
    arr = np.asarray(cm, dtype=float)
    sums = arr.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = np.divide(arr, sums, where=sums > 0)
    norm = np.nan_to_num(norm)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(norm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{int(arr[i, j])}", ha="center", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def _create_report_images(metrics, class_names, output_dir):
    rd = output_dir / "report_images"
    rd.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {"macro_f1": str(_plot_macro_f1(metrics, rd))}
    for s in ("train", "val", "test"):
        m = metrics.get(s)
        if m is None:
            continue
        paths[f"{s}_cm"] = str(
            _plot_cm(
                m["confusion_matrix"],
                class_names,
                f"{s.capitalize()} window CM",
                rd / f"{s}_window_cm.png",
            )
        )
    return paths


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_training(args) -> None:
    seed = int(args.random_state)
    random.seed(seed)
    np.random.seed(seed)

    X, manifest = load_and_label(
        embeddings_path=args.embeddings,
        embeddings_manifest_path=args.embeddings_manifest,
        annotations_manifest_path=args.annotations_manifest,
        treat_artifact_as_noise=args.treat_artifact_as_noise,
        min_overlap_s=args.min_overlap_s,
    )

    n_sound = int((manifest["label"] == "sound").sum())
    n_noise = int((manifest["label"] == "noise").sum())
    print(f"Window label distribution: sound={n_sound}, noise={n_noise}")

    file_df = get_file_labels(manifest)
    train_f, val_f, test_f = split_files(
        file_df, args.test_size, args.val_size, args.random_state
    )

    encoder = LabelEncoder()
    encoder.fit(np.array(["noise", "sound"]))

    train = build_split("train", manifest, X, train_f, encoder)
    val = build_split("val", manifest, X, val_f, encoder)
    test = build_split("test", manifest, X, test_f, encoder)
    if len(train.indices) == 0:
        raise ValueError("Training split is empty.")

    train.manifest.to_csv(args.output_dir / "train_manifest.csv", index=False)
    val.manifest.to_csv(args.output_dir / "val_manifest.csv", index=False)
    test.manifest.to_csv(args.output_dir / "test_manifest.csv", index=False)

    model = build_model(args)
    model.fit(train.X, train.y)

    class_names = encoder.classes_.tolist()
    metrics = {
        "config": {
            "embeddings": str(args.embeddings),
            "embeddings_manifest": str(args.embeddings_manifest),
            "annotations_manifest": str(args.annotations_manifest),
            "treat_artifact_as_noise": bool(args.treat_artifact_as_noise),
            "min_overlap_s": float(args.min_overlap_s),
            "test_size": args.test_size,
            "val_size": args.val_size,
            "random_state": args.random_state,
            "max_iter": args.max_iter,
            "scale": not args.no_scale,
        },
        "classes": class_names,
        "label_distribution": {"sound": n_sound, "noise": n_noise},
        "split_counts": {
            "train_files": len(train_f),
            "val_files": len(val_f),
            "test_files": len(test_f),
            "train_windows": len(train.indices),
            "val_windows": len(val.indices),
            "test_windows": len(test.indices),
        },
        "train": evaluate_split(train, model, encoder),
        "val": evaluate_split(val, model, encoder) if len(val.indices) else None,
        "test": evaluate_split(test, model, encoder) if len(test.indices) else None,
    }

    joblib.dump(
        {"model": model, "label_encoder": encoder}, args.output_dir / "model.joblib"
    )
    _save_json(metrics, args.output_dir / "metrics.json")

    summary: dict = {
        "treat_artifact_as_noise": bool(args.treat_artifact_as_noise),
        "train_macro_f1": metrics["train"]["macro_f1"],
    }
    for s in ("val", "test"):
        sm = metrics.get(s)
        if sm is None:
            continue
        summary[f"{s}_macro_f1"] = sm["macro_f1"]

    if args.generate_report_images:
        summary["report_images"] = _create_report_images(
            metrics, class_names, args.output_dir
        )
    _save_json(summary, args.output_dir / "summary.json")

    print("Saved model to:", args.output_dir / "model.joblib")
    print("Saved metrics to:", args.output_dir / "metrics.json")
    print("Saved summary to:", args.output_dir / "summary.json")
    if args.generate_report_images:
        print("Saved report images to:", args.output_dir / "report_images")
    print("\nSplit counts:")
    print(json.dumps(metrics["split_counts"], indent=2))
    print("\nWindow Macro-F1 summary:")
    print(json.dumps(summary, indent=2))
