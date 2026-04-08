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


def _build_rows(
    annotations: list[dict[str, Any]],
    join_multilabels: bool,
    label_separator: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for file_idx, file_item in enumerate(annotations):
        audio = str(file_item.get("audio", ""))
        source_id = file_item.get("id")
        events = file_item.get("label", [])
        if not isinstance(events, list):
            continue

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

    with open(input_json, "r", encoding="utf-8") as f:
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
