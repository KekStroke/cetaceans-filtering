"""Data loading, labeling, and file-level splitting for SED binary classifier."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .labeling import label_windows


@dataclass
class SplitData:
    name: str
    indices: np.ndarray
    manifest: pd.DataFrame
    X: np.ndarray
    y_text: np.ndarray
    y: np.ndarray


def load_embeddings(path: Path) -> np.ndarray:
    X = np.load(path)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D embeddings array, got shape {X.shape}")
    return X


def load_and_label(
    embeddings_path: Path,
    embeddings_manifest_path: Path,
    annotations_manifest_path: Path,
    treat_artifact_as_noise: bool,
    min_overlap_s: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Load embeddings and assign binary labels to each window.

    Returns (X, manifest) where manifest has columns:
    filename, start_s, end_s, label.
    """
    X = load_embeddings(embeddings_path)
    emb_manifest = pd.read_csv(embeddings_manifest_path)
    ann_manifest = pd.read_csv(annotations_manifest_path)

    labels, artifact_mask = label_windows(
        emb_manifest, ann_manifest, treat_artifact_as_noise, min_overlap_s
    )

    manifest = pd.DataFrame(
        {
            "filename": emb_manifest["filename"].astype(str),
            "start_s": emb_manifest["start_s"],
            "end_s": emb_manifest["end_s"],
            "label": labels,
        }
    )

    if not treat_artifact_as_noise:
        keep = ~artifact_mask | (labels == "sound")
        X = X[keep]
        manifest = manifest[keep].reset_index(drop=True)

    if len(X) != len(manifest):
        raise ValueError(
            f"Embeddings ({len(X)}) and manifest ({len(manifest)}) size mismatch."
        )
    return X, manifest


def get_file_labels(manifest: pd.DataFrame) -> pd.DataFrame:
    """One row per file with a stratification label (has_sound)."""
    has_sound = (
        manifest.groupby("filename")["label"]
        .apply(lambda s: "sound" if (s == "sound").any() else "noise")
        .reset_index()
    )
    has_sound.columns = ["filename", "file_label"]
    return has_sound


def split_files(
    file_df: pd.DataFrame,
    test_size: float,
    val_size: float,
    random_state: int,
) -> tuple[set[str], set[str], set[str]]:
    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size must be < 1.0")

    names = file_df["filename"].to_numpy()
    strat = file_df["file_label"].to_numpy()

    def _safe_split(arr, y, size, seed):
        if size <= 0:
            return arr, np.array([], dtype=arr.dtype), y, np.array([], dtype=y.dtype)
        try:
            return train_test_split(
                arr, y, test_size=size, stratify=y, random_state=seed
            )
        except ValueError as exc:
            warnings.warn(
                f"Stratified split failed, falling back to unstratified: {exc}"
            )
            return train_test_split(arr, y, test_size=size, random_state=seed)

    trainval, test_f, trainval_y, _ = _safe_split(
        names, strat, test_size, random_state
    )
    if val_size > 0:
        val_frac = val_size / (1.0 - test_size)
        train_f, val_f, _, _ = _safe_split(
            trainval, trainval_y, val_frac, random_state + 1
        )
    else:
        train_f, val_f = trainval, np.array([], dtype=names.dtype)

    return set(train_f.tolist()), set(val_f.tolist()), set(test_f.tolist())


def build_split(
    name: str,
    manifest: pd.DataFrame,
    X: np.ndarray,
    files: set[str],
    encoder: LabelEncoder,
) -> SplitData:
    mask = manifest["filename"].isin(files).to_numpy()
    indices = np.flatnonzero(mask)
    sub = manifest.iloc[indices].reset_index(drop=True)
    y_text = sub["label"].to_numpy()
    return SplitData(
        name=name,
        indices=indices,
        manifest=sub,
        X=X[indices],
        y_text=y_text,
        y=encoder.transform(y_text),
    )
