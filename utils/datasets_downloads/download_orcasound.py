import random
from math import ceil
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import boto3
import hydra
import soundfile as sf
from audio_saver import process_large_audio, sanitize_stem
from botocore import UNSIGNED
from botocore.client import Config
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
    ("acoustic-sandbox", "2017-09-27_OS_SRKW-wav/"),
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


def _parse_sources_from_config(dl: DictConfig) -> List[Source]:
    cfg_sources = dl.get("orcasound_sources")
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


def _filter_sources_by_prefixes(sources: List[Source], dl: DictConfig) -> List[Source]:
    selected_prefixes_cfg = dl.get("orcasound_selected_prefixes")
    if not selected_prefixes_cfg:
        return sources

    selected = {_normalize_prefix(str(p)) for p in selected_prefixes_cfg if str(p).strip()}
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

    out_root = Path(dl["raw_datasets_path"])
    dataset_root = out_root / str(dl.get("orcasound_output_dir", "orcasound"))
    downloads_dir = dataset_root / "downloads"
    processed_dir = dataset_root / "audio"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    sr_target_cfg = dl.get("raw_sample_rate")
    if sr_target_cfg is None or str(sr_target_cfg).strip().lower() in {"none", "null", ""}:
        sr_target = int(dl.get("orcasound_assume_sample_rate_hz", 48000))
    else:
        sr_target = int(sr_target_cfg)
    chunk_sec = float(dl["raw_segment_duration"])

    only_new_files = bool(dl.get("orcasound_only_new_files", False))
    delete_downloaded = bool(dl.get("orcasound_delete_downloaded_after_processing", False))
    max_files_cfg = dl.get("orcasound_max_files_per_source")
    if max_files_cfg is None or str(max_files_cfg).strip().lower() in {"none", "null", ""}:
        max_files_per_source: Optional[int] = None
    else:
        max_files_per_source = int(max_files_cfg)
    target_hours_cfg = dl.get("orcasound_target_hours_total")
    if target_hours_cfg is None or str(target_hours_cfg).strip().lower() in {"none", "null", ""}:
        target_hours_total: Optional[float] = None
    else:
        target_hours_total = float(target_hours_cfg)
    target_duration_seconds: Optional[float] = (
        None if target_hours_total is None else target_hours_total * 3600.0
    )
    assume_minutes_per_file = float(dl.get("orcasound_assume_minutes_per_file", 5.0))
    if assume_minutes_per_file <= 0:
        raise ValueError("data_loading.orcasound_assume_minutes_per_file must be > 0")
    max_total_files: Optional[int] = None
    if target_hours_total is not None:
        max_total_files = int((target_hours_total * 60.0) // assume_minutes_per_file)
        max_total_files = max(max_total_files, 1)
    seed = int(dl.get("orcasound_random_seed", 42))

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    sources = _filter_sources_by_prefixes(_parse_sources_from_config(dl), dl)
    if not sources:
        raise ValueError(
            "No Orcasound sources selected. "
            "Check data_loading.orcasound_selected_prefixes or data_loading.orcasound_sources."
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
    print(
        f"Duration cap: "
        f"{target_hours_total:.2f} h actual audio"
        if target_hours_total is not None
        else "Duration cap: disabled"
    )
    print(f"Chunking: {'disabled' if chunk_sec == -1 else f'{chunk_sec:.2f}s'}")

    for source_idx, (bucket, prefix) in enumerate(sources):
        if target_duration_seconds is not None and total_seconds[0] >= target_duration_seconds:
            print("\nReached target total duration. Stopping.")
            break
        if max_total_files is not None and processed_files >= max_total_files:
            print("\nReached global file cap. Stopping.")
            break

        source_name = sanitize_stem(f"{bucket}_{prefix.rstrip('/').replace('/', '_')}")
        source_download_dir = downloads_dir / source_name
        source_download_dir.mkdir(parents=True, exist_ok=True)

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
                local_src = source_download_dir / Path(key)
                if local_src.exists() and local_src.stat().st_size == size:
                    continue
                pending.append((key, size))
            objects = pending
            if not objects:
                print("No new files for this source.")
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

        rng = random.Random(seed + source_idx)
        selected = _pick_objects(objects=objects, max_files=effective_source_cap, rng=rng)
        print(f"Selected files: {len(selected)}")

        for file_idx, (key, size) in enumerate(selected):
            src_name = Path(key).name
            # Keep full key-relative path to avoid basename collisions.
            local_src = source_download_dir / Path(key)
            local_src.parent.mkdir(parents=True, exist_ok=True)

            if not local_src.exists() or local_src.stat().st_size != size:
                print(f"Downloading [{file_idx + 1}/{len(selected)}]: {src_name}")
                try:
                    s3.download_file(bucket, key, str(local_src))
                    total_downloaded_bytes += size
                except Exception as exc:
                    print(f"Skip file due to download error: s3://{bucket}/{key} ({exc})")
                    continue
            else:
                print(f"Already downloaded: {src_name}")

            if target_duration_seconds is not None:
                remaining = target_duration_seconds - total_seconds[0]
                if remaining <= 0:
                    print("Reached target total duration. Stopping.")
                    break
                file_duration = _probe_duration_seconds(local_src)
                # If one file is much longer than the remaining budget, skip it.
                if (
                    file_duration is not None
                    and file_duration > remaining
                    and remaining < (0.5 * file_duration)
                    and processed_files > 0
                ):
                    print(
                        f"Skip '{src_name}' (duration {file_duration/60:.1f} min) "
                        f"to keep target budget (~{remaining/60:.1f} min left)."
                    )
                    continue

            key_stem = Path(key).with_suffix("").as_posix().replace("/", "_")
            stem = sanitize_stem(f"{bucket}_{key_stem}")
            try:
                before_seconds = total_seconds[0]
                process_large_audio(
                    src_path=local_src,
                    out_dir=processed_dir,
                    stem_base=stem,
                    total_seconds_ref=total_seconds,
                    sr_target=sr_target,
                    chunk_sec=chunk_sec,
                )
                processed_files += 1
            except Exception as exc:
                print(f"Error processing '{src_name}': {exc}")
                continue

            if target_duration_seconds is not None:
                added = total_seconds[0] - before_seconds
                if added > 0 and total_seconds[0] >= target_duration_seconds:
                    print(
                        f"Reached target total duration after '{src_name}' "
                        f"(added {added/60:.1f} min)."
                    )
                    break

            if delete_downloaded:
                try:
                    local_src.unlink(missing_ok=True)
                except Exception:
                    pass
        else:
            continue
        break

    print("\nFinished Orcasound download + processing")
    print(f"Processed source files: {processed_files}")
    print(f"Total duration (output WAVs): {total_seconds[0] / 3600:.2f} h")
    print(f"Downloaded data: {total_downloaded_bytes / (1024**3):.2f} GB")
    print(f"Download cache dir: {downloads_dir.resolve()}")
    print(f"Audio dir: {processed_dir.resolve()}")


if __name__ == "__main__":
    main()
