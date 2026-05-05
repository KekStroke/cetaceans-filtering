import random
from math import ceil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import boto3
import hydra
import soundfile as sf
from audio_saver import process_large_audio, sanitize_stem
from botocore import UNSIGNED
from botocore.client import Config
from manifest_utils import write_manifest
from omegaconf import DictConfig

AudioObject = Tuple[str, int]
Source = Tuple[str, str]

# Public Orcasound Open Data sources on AWS Open Data:
# https://registry.opendata.aws/orcasound/
DEFAULT_SOURCES: List[Source] = [
    ("acoustic-sandbox", "2017-09-27_OS_SRKW-wav/"),
    ("acoustic-sandbox", "2017-09-05-SRKW-highlight-hour/"),
    ("acoustic-sandbox", "2017-09-05-SRKW/"),
    ("acoustic-sandbox", "2017-09-27-OS-continuous-wavs/"),
    ("acoustic-sandbox", "2017_8_VesselsAndWavS/"),
    ("acoustic-sandbox", "2018-sperm-whale-Yukusam/"),
    ("acoustic-sandbox", "2019-11-14_PT_SRKW_HLS/"),
    ("acoustic-sandbox", "2019-Orcasound-examples/"),
    ("acoustic-sandbox", "2020-06-26-SRKW-Lpod/"),
    ("acoustic-sandbox", "2021_9_12_OS_YearsBestVocalPassby/"),
    ("acoustic-sandbox", "acoustic-separation/"),
    ("acoustic-sandbox", "clap-model/"),
    ("acoustic-sandbox", "data-audio-raw/"),
    ("acoustic-sandbox", "humpbacks/"),
    ("acoustic-sandbox", "labeled-data/"),
    ("acoustic-sandbox", "machineLearningFile/"),
    ("acoustic-sandbox", "orcaal-dev/"),
    ("acoustic-sandbox", "orcasounds/"),
    ("acoustic-sandbox", "wholistener/"),
]

AUDIO_EXTENSIONS = {".wav", ".flac", ".aif", ".aiff", ".mp3"}


def _normalize_prefix(prefix: str) -> str:
    prefix = str(prefix).strip().lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return prefix


def _unique_sources(sources: Iterable[Source]) -> List[Source]:
    unique: List[Source] = []
    seen = set()
    for bucket, raw_prefix in sources:
        prefix = _normalize_prefix(raw_prefix)
        key = (str(bucket).strip(), prefix)
        if not key[0] or not key[1] or key in seen:
            continue
        seen.add(key)
        unique.append(key)
    return unique


def _source_folder_name(bucket: str, prefix: str) -> str:
    return sanitize_stem(f"{bucket}_{prefix.rstrip('/').replace('/', '_')}")


def _object_stem(bucket: str, key: str) -> str:
    key_stem = Path(key).with_suffix("").as_posix().replace("/", "_")
    return sanitize_stem(f"{bucket}_{key_stem}")


def _processed_audio_outputs(processed_dir: Path, stem: str) -> List[Path]:
    direct = processed_dir / f"{stem}.wav"
    outputs: List[Path] = []
    if direct.exists():
        outputs.append(direct)
    outputs.extend(sorted(processed_dir.glob(f"{stem}_[0-9][0-9][0-9][0-9][0-9].wav")))
    return sorted(set(outputs))


def _processed_audio_exists(processed_dir: Path, stem: str) -> bool:
    return bool(_processed_audio_outputs(processed_dir, stem))


def _list_s3_audio_objects(s3, bucket: str, prefix: str) -> List[AudioObject]:
    objects: List[AudioObject] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = str(obj.get("Key", ""))
            if not key or key.endswith("/"):
                continue
            ext = Path(key).suffix.lower()
            if ext in AUDIO_EXTENSIONS:
                objects.append((key, int(obj.get("Size", 0))))
    return objects


def _parse_sources_from_config(orcasound_cfg: Dict[str, object]) -> List[Source]:
    cfg_sources = orcasound_cfg.get("sources")
    if not cfg_sources:
        return _unique_sources(DEFAULT_SOURCES)

    parsed: List[Source] = []
    for item in cfg_sources:
        if isinstance(item, dict) or hasattr(item, "get"):
            bucket = str(item.get("bucket", "")).strip()
            prefix = str(item.get("prefix", "")).strip()
            if bucket and prefix:
                parsed.append((bucket, prefix))
    return _unique_sources(parsed) or _unique_sources(DEFAULT_SOURCES)


def _filter_sources_by_prefixes(
    sources: List[Source], orcasound_cfg: Dict[str, object]
) -> List[Source]:
    selected_prefixes_cfg = orcasound_cfg.get("selected_prefixes")
    if not selected_prefixes_cfg:
        return sources

    selected = {
        _normalize_prefix(str(p)) for p in selected_prefixes_cfg if str(p).strip()
    }
    if not selected:
        return sources

    filtered = [(bucket, prefix) for bucket, prefix in sources if prefix in selected]
    return filtered


def _pick_objects(
    objects: Sequence[AudioObject],
    max_files: Optional[int],
    rng: random.Random,
) -> List[AudioObject]:
    chosen = list(objects)
    if max_files is None or len(chosen) <= max_files:
        return chosen
    rng.shuffle(chosen)
    return chosen[:max_files]


def _probe_duration_seconds(path: Path) -> Optional[float]:
    try:
        info = sf.info(str(path))
        if info is None:
            return None
        duration = float(getattr(info, "duration", 0.0) or 0.0)
        if duration <= 0:
            return None
        return duration
    except Exception:
        return None


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(config: DictConfig):
    dl = config["data_loading"]
    orcasound_cfg = dl["sources"]["orcasound"]

    out_root = Path(dl["raw_datasets_path"])
    dataset_root = out_root / str(orcasound_cfg.get("output_dir_name", "orcasound"))
    dataset_root.mkdir(parents=True, exist_ok=True)

    sr_target_cfg = dl.get("raw_sample_rate")
    if sr_target_cfg is None or str(sr_target_cfg).strip().lower() in {
        "none",
        "null",
        "",
    }:
        sr_target = int(orcasound_cfg.get("assume_sample_rate_hz", 48000))
    else:
        sr_target = int(sr_target_cfg)
    chunk_sec = float(dl["raw_segment_duration"])

    only_new_files = bool(orcasound_cfg.get("only_new_files", False))
    delete_downloaded = bool(
        orcasound_cfg.get("delete_downloaded_after_processing", False)
    )
    delete_nonmatching = bool(orcasound_cfg.get("delete_nonmatching_downloads", True))
    max_files_cfg = orcasound_cfg.get("max_files_per_source")
    if max_files_cfg is None or str(max_files_cfg).strip().lower() in {
        "none",
        "null",
        "",
    }:
        max_files_per_source: Optional[int] = None
    else:
        max_files_per_source = int(max_files_cfg)
    target_hours_cfg = orcasound_cfg.get("target_hours_total")
    if target_hours_cfg is None or str(target_hours_cfg).strip().lower() in {
        "none",
        "null",
        "",
    }:
        target_hours_total: Optional[float] = None
    else:
        target_hours_total = float(target_hours_cfg)
    target_duration_seconds: Optional[float] = (
        None if target_hours_total is None else target_hours_total * 3600.0
    )
    duration_min_cfg = orcasound_cfg.get("duration_min_minutes")
    duration_max_cfg = orcasound_cfg.get("duration_max_minutes")
    duration_min_minutes: Optional[float] = (
        None
        if duration_min_cfg is None
        or str(duration_min_cfg).strip().lower() in {"none", "null", ""}
        else float(duration_min_cfg)
    )
    duration_max_minutes: Optional[float] = (
        None
        if duration_max_cfg is None
        or str(duration_max_cfg).strip().lower() in {"none", "null", ""}
        else float(duration_max_cfg)
    )
    if (
        duration_min_minutes is not None
        and duration_max_minutes is not None
        and duration_min_minutes > duration_max_minutes
    ):
        raise ValueError(
            "data_loading.sources.orcasound.duration_min_minutes must be <= "
            "data_loading.sources.orcasound.duration_max_minutes"
        )
    duration_min_seconds = (
        None if duration_min_minutes is None else duration_min_minutes * 60.0
    )
    duration_max_seconds = (
        None if duration_max_minutes is None else duration_max_minutes * 60.0
    )
    assume_minutes_per_file = float(orcasound_cfg.get("assume_minutes_per_file", 5.0))
    if assume_minutes_per_file <= 0:
        raise ValueError(
            "data_loading.sources.orcasound.assume_minutes_per_file must be > 0"
        )
    max_total_files: Optional[int] = None
    if target_hours_total is not None:
        max_total_files = int((target_hours_total * 60.0) // assume_minutes_per_file)
        max_total_files = max(max_total_files, 1)
    seed = int(orcasound_cfg.get("random_seed", 42))

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    sources = _filter_sources_by_prefixes(
        _parse_sources_from_config(orcasound_cfg), orcasound_cfg
    )
    if not sources:
        raise ValueError(
            "No Orcasound sources selected. "
            "Check data_loading.sources.orcasound.selected_prefixes or "
            "data_loading.sources.orcasound.sources."
        )

    total_seconds = [0.0]
    total_downloaded_bytes = 0
    processed_files = 0

    print(f"Sources configured: {len(sources)}")
    print(f"Only new files: {only_new_files}")
    print(
        f"Max files per source: "
        f"{max_files_per_source if max_files_per_source is not None else 'unlimited'}"
    )
    print(
        f"Global files cap: "
        f"{max_total_files if max_total_files is not None else 'unlimited'}"
    )
    print(f"Delete downloads after processing: {delete_downloaded}")
    print(f"Delete non-matching downloads: {delete_nonmatching}")
    print(
        f"Duration cap: {target_hours_total:.2f} h actual audio"
        if target_hours_total is not None
        else "Duration cap: disabled"
    )
    print(
        "Preferred file duration: "
        f"{duration_min_minutes if duration_min_minutes is not None else '-inf'}.."
        f"{duration_max_minutes if duration_max_minutes is not None else '+inf'} min"
        if duration_min_minutes is not None or duration_max_minutes is not None
        else "Preferred file duration: any"
    )
    print(f"Chunking: {'disabled' if chunk_sec == -1 else f'{chunk_sec:.2f}s'}")

    for source_idx, (bucket, prefix) in enumerate(sources):
        if (
            target_duration_seconds is not None
            and total_seconds[0] >= target_duration_seconds
        ):
            print("\nReached target total duration. Stopping.")
            break
        if max_total_files is not None and processed_files >= max_total_files:
            print("\nReached global file cap. Stopping.")
            break

        source_name = _source_folder_name(bucket=bucket, prefix=prefix)
        source_dir = dataset_root / source_name
        source_download_dir = source_dir / "downloads"
        source_audio_dir = source_dir / "audio"
        manifest_path = source_dir / "manifest.jsonl"
        source_download_dir.mkdir(parents=True, exist_ok=True)
        source_audio_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== {bucket}/{prefix} ===")
        try:
            objects = _list_s3_audio_objects(s3=s3, bucket=bucket, prefix=prefix)
        except Exception as exc:
            print(f"Skip source due to listing error: {exc}")
            continue

        if not objects:
            print("No audio files found in this source.")
            continue

        if only_new_files:
            pending: List[AudioObject] = []
            for key, size in objects:
                stem = _object_stem(bucket=bucket, key=key)
                already_processed = _processed_audio_exists(source_audio_dir, stem)
                if already_processed:
                    continue
                pending.append((key, size))
            objects = pending
            if not objects:
                print("No new files for this source.")
                manifest_entries = write_manifest(
                    audio_dir=source_audio_dir,
                    manifest_path=manifest_path,
                )
                print(
                    f"Manifest entries: {manifest_entries} ({manifest_path.resolve()})"
                )
                continue

        effective_source_cap = max_files_per_source
        if max_total_files is not None:
            remaining = max_total_files - processed_files
            if remaining <= 0:
                print("Global file cap reached before this source.")
                break
            remaining_sources = len(sources) - source_idx
            fair_share_cap = max(1, ceil(remaining / max(remaining_sources, 1)))
            if effective_source_cap is None:
                effective_source_cap = fair_share_cap
            else:
                effective_source_cap = min(effective_source_cap, fair_share_cap)

        selection_cap = effective_source_cap
        # When filtering by duration, oversample candidates so we can skip non-matching files.
        if selection_cap is not None and (
            duration_min_seconds is not None or duration_max_seconds is not None
        ):
            selection_cap = max(selection_cap * 8, selection_cap)

        rng = random.Random(seed + source_idx)
        selected = _pick_objects(objects=objects, max_files=selection_cap, rng=rng)
        print(f"Selected files: {len(selected)}")

        stop_all_sources = False
        accepted_source_files = 0
        for file_idx, (key, size) in enumerate(selected):
            if max_total_files is not None and processed_files >= max_total_files:
                print("Reached global file cap. Stopping.")
                stop_all_sources = True
                break
            if (
                effective_source_cap is not None
                and accepted_source_files >= effective_source_cap
            ):
                print("Reached source file cap.")
                break

            src_name = Path(key).name
            # Keep full key-relative path to avoid basename collisions.
            local_src = source_download_dir / Path(key)
            local_src.parent.mkdir(parents=True, exist_ok=True)
            downloaded_now = False

            if not local_src.exists() or local_src.stat().st_size != size:
                print(f"Downloading [{file_idx + 1}/{len(selected)}]: {src_name}")
                try:
                    s3.download_file(bucket, key, str(local_src))
                    total_downloaded_bytes += size
                    downloaded_now = True
                except Exception as exc:
                    print(
                        f"Skip file due to download error: s3://{bucket}/{key} ({exc})"
                    )
                    continue
            else:
                print(f"Already downloaded: {src_name}")

            file_duration = _probe_duration_seconds(local_src)
            if file_duration is None and (
                duration_min_seconds is not None or duration_max_seconds is not None
            ):
                print(f"Skip '{src_name}': failed to probe duration.")
                if delete_nonmatching and downloaded_now:
                    try:
                        local_src.unlink(missing_ok=True)
                    except Exception:
                        pass
                continue
            if (
                duration_min_seconds is not None
                and file_duration is not None
                and file_duration < duration_min_seconds
            ):
                print(
                    f"Skip '{src_name}': {file_duration / 60:.2f} min < "
                    f"min {duration_min_seconds / 60:.2f} min."
                )
                if delete_nonmatching and downloaded_now:
                    try:
                        local_src.unlink(missing_ok=True)
                    except Exception:
                        pass
                continue
            if (
                duration_max_seconds is not None
                and file_duration is not None
                and file_duration > duration_max_seconds
            ):
                print(
                    f"Skip '{src_name}': {file_duration / 60:.2f} min > "
                    f"max {duration_max_seconds / 60:.2f} min."
                )
                if delete_nonmatching and downloaded_now:
                    try:
                        local_src.unlink(missing_ok=True)
                    except Exception:
                        pass
                continue

            if target_duration_seconds is not None:
                remaining = target_duration_seconds - total_seconds[0]
                if remaining <= 0:
                    print("Reached target total duration. Stopping.")
                    stop_all_sources = True
                    break
                # If one file is much longer than the remaining budget, skip it.
                if (
                    file_duration is not None
                    and file_duration > remaining
                    and remaining < (0.5 * file_duration)
                    and processed_files > 0
                ):
                    print(
                        f"Skip '{src_name}' (duration {file_duration / 60:.1f} min) "
                        f"to keep target budget (~{remaining / 60:.1f} min left)."
                    )
                    continue

            stem = _object_stem(bucket=bucket, key=key)
            try:
                before_seconds = total_seconds[0]
                process_large_audio(
                    src_path=local_src,
                    out_dir=source_audio_dir,
                    stem_base=stem,
                    total_seconds_ref=total_seconds,
                    sr_target=sr_target,
                    chunk_sec=chunk_sec,
                )
                processed_files += 1
                accepted_source_files += 1
            except Exception as exc:
                print(f"Error processing '{src_name}': {exc}")
                continue

            if target_duration_seconds is not None:
                added = total_seconds[0] - before_seconds
                if added > 0 and total_seconds[0] >= target_duration_seconds:
                    print(
                        f"Reached target total duration after '{src_name}' "
                        f"(added {added / 60:.1f} min)."
                    )
                    stop_all_sources = True
                    break

            if delete_downloaded:
                try:
                    local_src.unlink(missing_ok=True)
                except Exception:
                    pass

        manifest_entries = write_manifest(
            audio_dir=source_audio_dir,
            manifest_path=manifest_path,
        )
        print(f"Manifest entries: {manifest_entries} ({manifest_path.resolve()})")

        if stop_all_sources:
            break

    print("\nFinished Orcasound download + processing")
    print(f"Processed source files: {processed_files}")
    print(f"Total duration (output WAVs): {total_seconds[0] / 3600:.2f} h")
    print(f"Downloaded data: {total_downloaded_bytes / (1024**3):.2f} GB")
    print(f"Dataset dir: {dataset_root.resolve()}")


if __name__ == "__main__":
    main()
