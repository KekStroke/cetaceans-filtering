from __future__ import annotations

import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .types import SplitData


def infer_manifest_columns(
    df: pd.DataFrame, args: SimpleNamespace
) -> tuple[int, int, int, int | None]:
    ncols = df.shape[1]
    lower_cols = {str(c).strip().lower(): i for i, c in enumerate(df.columns)}

    if args.filename_col is not None:
        filename_col = args.filename_col
    elif "filename" in lower_cols:
        filename_col = lower_cols["filename"]
    elif "file" in lower_cols:
        filename_col = lower_cols["file"]
    elif ncols >= 6:
        filename_col = 3
    elif ncols == 5:
        filename_col = 2
    else:
        raise ValueError(
            "Could not infer filename column. Set filename_col explicitly."
        )

    if args.start_col is not None:
        start_col = args.start_col
    elif "start_s" in lower_cols:
        start_col = lower_cols["start_s"]
    elif "start" in lower_cols:
        start_col = lower_cols["start"]
    elif ncols >= 6:
        start_col = 4
    elif ncols == 5:
        start_col = 3
    else:
        raise ValueError("Could not infer start-time column. Set start_col explicitly.")

    if args.end_col is not None:
        end_col = args.end_col
    elif "end_s" in lower_cols:
        end_col = lower_cols["end_s"]
    elif "end" in lower_cols:
        end_col = lower_cols["end"]
    elif ncols >= 6:
        end_col = 5
    elif ncols == 5:
        end_col = 4
    else:
        raise ValueError("Could not infer end-time column. Set end_col explicitly.")

    return filename_col, start_col, end_col, args.label_col


def parse_label_from_filename(filename: str, label_regex: str | None = None) -> str:
    import re

    basename = Path(filename).name
    if label_regex is not None:
        match = re.match(label_regex, basename)
        if not match:
            raise ValueError(
                f"Filename {basename!r} did not match label_regex {label_regex!r}."
            )
        return match.group(1)

    parts = Path(basename).stem.split("_")
    if len(parts) < 3:
        raise ValueError(
            f"Could not infer label from filename {basename!r}. "
            "Expected wmms_00001_Spinner_Dolphin.wav format."
        )
    return "_".join(parts[2:])


def load_manifest(path: Path, args: SimpleNamespace) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        if all(isinstance(c, int) for c in df.columns):
            raise ValueError("Reload as headerless")
    except Exception:
        df = pd.read_csv(path, header=None)

    filename_col, start_col, end_col, label_col = infer_manifest_columns(df, args)
    out = pd.DataFrame(
        {
            "filename": df.iloc[:, filename_col].astype(str),
            "start_s": pd.to_numeric(df.iloc[:, start_col], errors="coerce"),
            "end_s": pd.to_numeric(df.iloc[:, end_col], errors="coerce"),
        }
    )
    out["row_index"] = np.arange(len(out))
    if label_col is not None:
        out["label"] = df.iloc[:, label_col].astype(str)
    else:
        out["label"] = out["filename"].apply(
            lambda x: parse_label_from_filename(x, args.label_regex)
        )

    if out[["filename", "start_s", "end_s", "label"]].isnull().any().any():
        bad = out[out[["filename", "start_s", "end_s", "label"]].isnull().any(axis=1)]
        raise ValueError(f"Manifest contains invalid rows after parsing:\n{bad.head()}")
    return out


def load_embeddings(path: Path) -> np.ndarray:
    X = np.load(path)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D embeddings array, got shape {X.shape}")
    return X


def unique_files_with_labels(manifest: pd.DataFrame) -> pd.DataFrame:
    file_df = manifest[["filename", "label"]].drop_duplicates().reset_index(drop=True)
    dup_counts = file_df["filename"].value_counts()
    if (dup_counts > 1).any():
        raise ValueError("A filename appears with more than one label.")
    return file_df


def split_files(
    file_df: pd.DataFrame, test_size: float, val_size: float, random_state: int
) -> tuple[set[str], set[str], set[str]]:
    if not (0.0 <= test_size < 1.0) or not (0.0 <= val_size < 1.0):
        raise ValueError("test_size and val_size must be in [0, 1).")
    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size must be < 1.0")

    names = file_df["filename"].to_numpy()
    labels = file_df["label"].to_numpy()

    def _safe_split(arr_names: np.ndarray, arr_y: np.ndarray, size: float, seed: int):
        if size <= 0:
            return arr_names, np.array([], dtype=arr_names.dtype), arr_y, np.array([])
        try:
            return train_test_split(
                arr_names, arr_y, test_size=size, stratify=arr_y, random_state=seed
            )
        except ValueError as exc:
            warnings.warn(
                "Stratified split failed; falling back to unstratified split. "
                f"Original error: {exc}"
            )
            return train_test_split(arr_names, arr_y, test_size=size, random_state=seed)

    trainval_files, test_files, trainval_y, _ = _safe_split(
        names, labels, test_size, random_state
    )
    if val_size > 0:
        val_fraction = val_size / (1.0 - test_size)
        train_files, val_files, _, _ = _safe_split(
            trainval_files, trainval_y, val_fraction, random_state + 1
        )
    else:
        train_files, val_files = trainval_files, np.array([], dtype=names.dtype)

    return set(train_files.tolist()), set(val_files.tolist()), set(test_files.tolist())


def build_split(
    name: str,
    manifest: pd.DataFrame,
    X: np.ndarray,
    files: set[str],
    encoder: LabelEncoder,
) -> SplitData:
    mask = manifest["filename"].isin(files).to_numpy()
    indices = np.flatnonzero(mask)
    sub_manifest = manifest.iloc[indices].reset_index(drop=True)
    y_text = sub_manifest["label"].to_numpy()
    y = encoder.transform(y_text)
    return SplitData(
        name=name,
        indices=indices,
        manifest=sub_manifest,
        X=X[indices],
        y_text=y_text,
        y=y,
    )
