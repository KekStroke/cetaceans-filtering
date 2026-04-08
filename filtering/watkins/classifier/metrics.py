from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from .types import SplitData


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]
) -> dict:
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


def aggregate_file_predictions(
    probs: np.ndarray,
    manifest: pd.DataFrame,
    strategy: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if strategy not in {"mean", "max"}:
        raise ValueError(f"Unknown aggregation strategy: {strategy}")

    grouped = manifest.reset_index(drop=True).groupby("filename", sort=False)
    agg_probs = []
    y_true_text = []
    filenames = []
    for filename, group in grouped:
        idx = group.index.to_numpy()
        file_probs = probs[idx]
        agg = file_probs.mean(axis=0) if strategy == "mean" else file_probs.max(axis=0)
        agg_probs.append(agg)
        y_true_text.append(group["label"].iloc[0])
        filenames.append(filename)
    return np.vstack(agg_probs), np.array(y_true_text), filenames


def evaluate_split(
    split: SplitData,
    model: Pipeline,
    encoder: LabelEncoder,
) -> dict:
    class_names = encoder.classes_.tolist()
    window_probs = model.predict_proba(split.X)
    window_pred = np.argmax(window_probs, axis=1)
    out = {
        "n_windows": int(len(split.y)),
        "n_files": int(split.manifest["filename"].nunique()),
        "window": compute_metrics(split.y, window_pred, class_names),
    }

    file_metrics = {}
    for strategy in ("mean", "max"):
        file_probs, y_true_text, filenames = aggregate_file_predictions(
            window_probs, split.manifest, strategy
        )
        y_true = encoder.transform(y_true_text)
        y_pred = np.argmax(file_probs, axis=1)
        metrics = compute_metrics(y_true, y_pred, class_names)
        metrics["filenames"] = filenames
        metrics["y_true"] = y_true_text.tolist()
        metrics["y_pred"] = encoder.inverse_transform(y_pred).tolist()
        file_metrics[strategy] = metrics
    out["file"] = file_metrics
    return out
