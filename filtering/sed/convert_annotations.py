from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig


def _normalize_labels(raw_labels: Any) -> list[str]:
    if raw_labels is None:
        return []
    if isinstance(raw_labels, str):
        return [raw_labels]
    if isinstance(raw_labels, list):
        return [str(x) for x in raw_labels if x is not None and str(x) != ""]
    return [str(raw_labels)]


def _audio_keys(file_item: dict[str, Any]) -> list[str]:
    data = file_item.get("data")
    candidates: list[Any] = []
    if isinstance(data, dict):
        candidates.extend([data.get("audio"), data.get("file"), data.get("wav")])
    candidates.extend([file_item.get("audio"), file_item.get("file_upload")])
    out: list[str] = []
    for raw in candidates:
        if not raw:
            continue
        name = str(raw).replace("\\", "/")
        out.append(name)
        out.append(Path(name).name)
    return [value for i, value in enumerate(out) if value and value not in out[:i]]


def _events(file_item: dict[str, Any]) -> list[dict[str, Any]]:
    direct = file_item.get("label")
    if isinstance(direct, list):
        return [event for event in direct if isinstance(event, dict)]

    events: list[dict[str, Any]] = []
    annotations = file_item.get("annotations", [])
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
                    "channel": value.get("channel"),
                    "labels": value.get("labels") or value.get("choices"),
                }
            )
    return events


def _build_rows(
    annotations: list[dict[str, Any]],
    join_multilabels: bool,
    label_separator: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for file_idx, file_item in enumerate(annotations):
        audio = next(iter(_audio_keys(file_item)), "")
        source_id = file_item.get("id")
        events = _events(file_item)

        for event_idx, event in enumerate(events):
            if not isinstance(event, dict):
                continue
            start = float(event.get("start", 0.0))
            end = float(event.get("end", 0.0))
            channel = event.get("channel")
            labels = _normalize_labels(event.get("labels"))
            duration = max(0.0, end - start)

            if not labels:
                labels = ["unknown"]

            if join_multilabels:
                rows.append(
                    {
                        "row_id": len(rows),
                        "file_index": file_idx,
                        "event_index": event_idx,
                        "source_id": source_id,
                        "audio": audio,
                        "start_s": start,
                        "end_s": end,
                        "duration_s": duration,
                        "channel": channel,
                        "label": label_separator.join(labels),
                        "label_count": len(labels),
                    }
                )
            else:
                for label_pos, label in enumerate(labels):
                    rows.append(
                        {
                            "row_id": len(rows),
                            "file_index": file_idx,
                            "event_index": event_idx,
                            "label_index": label_pos,
                            "source_id": source_id,
                            "audio": audio,
                            "start_s": start,
                            "end_s": end,
                            "duration_s": duration,
                            "channel": channel,
                            "label": label,
                            "label_count": len(labels),
                        }
                    )
    return rows


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(config: DictConfig) -> None:
    cfg = config["sound_event_detection"]
    input_json = Path(hydra.utils.to_absolute_path(str(cfg["input_json"])))
    output_csv = Path(hydra.utils.to_absolute_path(str(cfg["output_csv"])))
    join_multilabels = bool(cfg["join_multilabels"])
    label_separator = str(cfg["label_separator"])

    with open(input_json, "r", encoding="utf-8-sig") as f:
        payload = json.load(f)

    if not isinstance(payload, list):
        raise ValueError("Expected top-level JSON array in annotations file.")

    rows = _build_rows(
        annotations=payload,
        join_multilabels=join_multilabels,
        label_separator=label_separator,
    )
    if not rows:
        raise ValueError("No annotation rows were produced. Check input schema/content.")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Input:  {input_json.resolve()}")
    print(f"Rows:   {len(rows)}")
    print(f"Output: {output_csv.resolve()}")


if __name__ == "__main__":
    main()
