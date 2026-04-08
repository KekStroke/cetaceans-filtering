"""Label Perch embedding windows using SED annotation events.

Each embedding window is assigned a binary label ("sound" or "noise") based on
temporal overlap with annotated events.  Artifact events can optionally be
flagged for exclusion from training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _build_event_index(
    ann: pd.DataFrame, target_label: str
) -> dict[str, list[tuple[float, float]]]:
    events: dict[str, list[tuple[float, float]]] = {}
    for _, row in ann.iterrows():
        if str(row.get("label", "")).lower().strip() == target_label:
            audio = str(row["audio"])
            events.setdefault(audio, []).append(
                (float(row["start_s"]), float(row["end_s"]))
            )
    for k in events:
        events[k].sort()
    return events


def _has_overlap(
    w_start: float,
    w_end: float,
    events: list[tuple[float, float]],
    min_overlap_s: float,
) -> bool:
    for e_start, e_end in events:
        if e_start > w_end:
            break
        overlap = min(w_end, e_end) - max(w_start, e_start)
        if overlap >= min_overlap_s:
            return True
    return False


def label_windows(
    emb_manifest: pd.DataFrame,
    ann_manifest: pd.DataFrame,
    treat_artifact_as_noise: bool,
    min_overlap_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign binary labels to each embedding window.

    Returns
    -------
    labels : ndarray of str
        "sound" or "noise" per window.
    artifact_mask : ndarray of bool
        True for windows that overlap an artifact event (useful for exclusion
        when ``treat_artifact_as_noise=False``).
    """
    n = len(emb_manifest)
    labels = np.full(n, "noise", dtype=object)
    artifact_mask = np.zeros(n, dtype=bool)

    sound_idx = _build_event_index(ann_manifest, "sound")
    artifact_idx = _build_event_index(ann_manifest, "artifact")

    for i in range(n):
        row = emb_manifest.iloc[i]
        fname = str(row["filename"])
        ws, we = float(row["start_s"]), float(row["end_s"])

        if _has_overlap(ws, we, sound_idx.get(fname, []), min_overlap_s):
            labels[i] = "sound"

        if _has_overlap(ws, we, artifact_idx.get(fname, []), min_overlap_s):
            artifact_mask[i] = True

    return labels, artifact_mask
