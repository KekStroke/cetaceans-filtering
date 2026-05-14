import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import soundfile as sf


ManifestRecord = dict[str, Any]


def _audio_metadata(audio_path: Path) -> tuple[float, int] | None:
    try:
        info = sf.info(str(audio_path))
    except Exception as exc:
        print(f"Warning: failed to inspect '{audio_path.name}' for manifest: {exc}")
        return None

    duration = float(getattr(info, "duration", 0.0) or 0.0)
    sample_rate = int(getattr(info, "samplerate", 0) or 0)
    if duration <= 0 or sample_rate <= 0:
        print(f"Warning: skip manifest entry with invalid metadata: {audio_path.name}")
        return None

    return duration, sample_rate


def build_manifest_record(
    audio_path: Path,
    manifest_path: Path,
    extra_fields: Mapping[str, Any] | None = None,
) -> ManifestRecord | None:
    metadata = _audio_metadata(audio_path)
    if metadata is None:
        return None

    duration, sample_rate = metadata
    row: ManifestRecord = dict(extra_fields or {})
    row.update(
        {
            "audio_filepath": audio_path.relative_to(manifest_path.parent).as_posix(),
            "duration": duration,
            "sample_rate": sample_rate,
        }
    )
    return row


def write_manifest(
    audio_dir: Path,
    manifest_path: Path,
    extra_fields: Mapping[str, Any] | None = None,
) -> int:
    entries: list[ManifestRecord] = []
    for audio_path in sorted(audio_dir.rglob("*.wav")):
        row = build_manifest_record(
            audio_path=audio_path,
            manifest_path=manifest_path,
            extra_fields=extra_fields,
        )
        if row is not None:
            entries.append(row)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=True) + "\n")
    return len(entries)


def append_manifest_records(
    manifest_path: Path,
    audio_paths: Sequence[Path],
    extra_fields: Mapping[str, Any] | None = None,
) -> int:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with manifest_path.open("a", encoding="utf-8") as f:
        for audio_path in sorted(audio_paths, key=lambda p: str(p)):
            row = build_manifest_record(
                audio_path=audio_path,
                manifest_path=manifest_path,
                extra_fields=extra_fields,
            )
            if row is None:
                continue
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
            count += 1
    return count
