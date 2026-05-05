import json
import random
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

import hydra
from audio_saver import process_large_audio, sanitize_stem
from manifest_utils import write_manifest
from omegaconf import DictConfig


def _normalize_prefix(value: str) -> str:
    value = value.strip()
    if value.startswith("gs://"):
        _, rest = value.split("gs://", 1)
        parts = rest.split("/", 1)
        value = parts[1] if len(parts) > 1 else ""
    if "storage/browser/" in value:
        marker = "storage/browser/"
        value = value.split(marker, 1)[1]
        parts = value.split("/", 1)
        value = parts[1] if len(parts) > 1 else ""
    return value.strip("/")


def _list_objects_json(bucket: str, prefix: str) -> List[Dict[str, int | str]]:
    objects: List[Dict[str, int | str]] = []
    page_token: Optional[str] = None
    while True:
        params = {"prefix": prefix, "maxResults": "1000"}
        if page_token:
            params["pageToken"] = page_token
        query = urllib.parse.urlencode(params)
        url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o?{query}"
        with urllib.request.urlopen(url) as resp:
            payload = json.loads(resp.read().decode("utf-8"))

        for item in payload.get("items", []):
            name = item.get("name")
            size = int(item.get("size", 0))
            if name:
                objects.append({"name": name, "size": size})

        page_token = payload.get("nextPageToken")
        if not page_token:
            break
    return objects


def _list_objects_xml(bucket: str, prefix: str) -> List[Dict[str, int | str]]:
    objects: List[Dict[str, int | str]] = []
    marker: Optional[str] = None
    while True:
        params = {"prefix": prefix, "max-keys": "1000"}
        if marker:
            params["marker"] = marker
        query = urllib.parse.urlencode(params)
        url = f"https://storage.googleapis.com/{bucket}?{query}"
        with urllib.request.urlopen(url) as resp:
            root = ET.fromstring(resp.read())

        ns = {"s3": "http://doc.s3.amazonaws.com/2006-03-01"}
        contents = root.findall("s3:Contents", ns)
        if not contents:
            break

        for c in contents:
            key = c.findtext("s3:Key", default="", namespaces=ns)
            size_text = c.findtext("s3:Size", default="0", namespaces=ns)
            if key:
                objects.append({"name": key, "size": int(size_text)})

        is_truncated = root.findtext("s3:IsTruncated", default="false", namespaces=ns)
        if str(is_truncated).lower() != "true":
            break
        marker = contents[-1].findtext("s3:Key", default="", namespaces=ns)
        if not marker:
            break
    return objects


def _list_objects(bucket: str, prefix: str) -> List[Dict[str, int | str]]:
    try:
        return _list_objects_json(bucket=bucket, prefix=prefix)
    except Exception as exc:
        print(f"JSON listing failed for '{prefix}': {exc}. Trying XML fallback...")
        return _list_objects_xml(bucket=bucket, prefix=prefix)


def _is_audio_object(name: str) -> bool:
    ext = Path(name).suffix.lower()
    return ext in {".wav", ".flac", ".aif", ".aiff", ".mp3"}


def _source_name_from_prefix(prefix: str) -> str:
    first_part = _normalize_prefix(prefix).split("/", 1)[0]
    return sanitize_stem(first_part or "unknown")


def _processed_audio_exists(processed_dir: Path, stem: str) -> bool:
    return any(processed_dir.glob(f"{stem}*.wav"))


def _estimate_duration_seconds(
    size_bytes: int,
    sample_rate_hz: int,
    sample_bits: int,
    channels: int,
) -> float:
    bytes_per_sample = max(sample_bits // 8, 1)
    bytes_per_second = max(sample_rate_hz * channels * bytes_per_sample, 1)
    return float(size_bytes) / float(bytes_per_second)


def _download_file(
    bucket: str,
    object_name: str,
    dst_path: Path,
    expected_size: int,
    max_retries: int = 5,
) -> None:
    url = f"https://storage.googleapis.com/{bucket}/{object_name}"
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst_path.with_suffix(dst_path.suffix + ".part")

    for attempt in range(1, max_retries + 1):
        try:
            if tmp_path.exists():
                tmp_path.unlink()
            if dst_path.exists():
                dst_path.unlink()

            with (
                urllib.request.urlopen(url, timeout=120) as resp,
                tmp_path.open("wb") as f,
            ):
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)

            actual_size = tmp_path.stat().st_size
            if expected_size > 0 and actual_size != expected_size:
                raise IOError(
                    f"incomplete download: got {actual_size} of {expected_size} bytes"
                )

            tmp_path.replace(dst_path)
            return
        except Exception as exc:
            if attempt >= max_retries:
                raise RuntimeError(
                    f"failed to download '{object_name}' after {max_retries} attempts: {exc}"
                ) from exc
            wait_s = min(2 ** (attempt - 1), 10)
            print(
                f"Retry {attempt}/{max_retries} for '{Path(object_name).name}' after error: {exc}"
            )
            time.sleep(wait_s)


def _pick_random_files(
    files: List[Dict[str, int | str]],
    target_seconds: float,
    sample_rate_hz: int,
    sample_bits: int,
    channels: int,
    rng: random.Random,
    max_files: Optional[int] = None,
) -> List[Dict[str, int | str]]:
    pool = list(files)
    rng.shuffle(pool)
    selected: List[Dict[str, int | str]] = []
    acc = 0.0
    for item in pool:
        size = int(item.get("size", 0))
        est_sec = _estimate_duration_seconds(
            size_bytes=size,
            sample_rate_hz=sample_rate_hz,
            sample_bits=sample_bits,
            channels=channels,
        )
        selected.append(item)
        acc += est_sec
        if max_files is not None and len(selected) >= max_files:
            break
        if acc >= target_seconds:
            break
    return selected


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(config: DictConfig):
    dl = config["data_loading"]
    noaa_cfg = dl["sources"]["noaa"]

    bucket = str(noaa_cfg.get("bucket", "noaa-passive-bioacoustic"))
    prefixes = list(noaa_cfg.get("deployment_prefixes", []))
    if not prefixes:
        raise ValueError(
            "data_loading.sources.noaa.deployment_prefixes is empty. "
            "Add at least one deployment prefix."
        )

    out_root = Path(dl["raw_datasets_path"])
    out_dir = out_root / str(noaa_cfg.get("output_dir_name", "noaa_onms"))
    out_dir.mkdir(parents=True, exist_ok=True)

    sr_target_cfg = dl.get("raw_sample_rate")
    if sr_target_cfg is None or str(sr_target_cfg).strip().lower() in {
        "none",
        "null",
        "",
    }:
        sr_target = int(noaa_cfg.get("assume_sample_rate_hz", 48000))
    else:
        sr_target = int(sr_target_cfg)

    chunk_sec = float(dl["raw_segment_duration"])
    target_hours = float(noaa_cfg.get("hours_per_deployment", 1.5))
    target_seconds = target_hours * 3600.0
    only_new_files = bool(noaa_cfg.get("only_new_files", False))
    delete_downloaded = bool(noaa_cfg.get("delete_downloaded_after_processing", False))

    max_files_cfg = noaa_cfg.get("max_files_per_deployment")
    if max_files_cfg is None or str(max_files_cfg).strip().lower() in {
        "none",
        "null",
        "",
    }:
        max_files_per_deployment: Optional[int] = None
    else:
        max_files_per_deployment = int(max_files_cfg)

    sample_rate_hz = int(noaa_cfg.get("assume_sample_rate_hz", 48000))
    sample_bits = int(noaa_cfg.get("sample_bits", 16))
    channels = int(noaa_cfg.get("channels", 1))
    seed = int(noaa_cfg.get("random_seed", 42))

    total_seconds = [0.0]
    total_downloaded_bytes = 0

    print(f"Bucket: {bucket}")
    print(f"Deployments requested: {len(prefixes)}")
    print(f"Target per deployment: {target_hours:.2f} h")
    print(f"Only new files: {only_new_files}")
    print(f"Delete downloads after processing: {delete_downloaded}")
    print(
        f"Max files per deployment: "
        f"{max_files_per_deployment if max_files_per_deployment is not None else 'unlimited'}"
    )
    print(f"Chunking: {'disabled' if chunk_sec == -1 else f'{chunk_sec:.2f}s'}")

    for dep_idx, raw_prefix in enumerate(prefixes):
        prefix = _normalize_prefix(str(raw_prefix))
        if not prefix:
            continue
        if not prefix.endswith("/"):
            prefix = prefix + "/"

        source_name = _source_name_from_prefix(prefix)
        source_dir = out_dir / source_name
        source_audio_dir = source_dir / "audio"
        source_downloads_dir = source_dir / "downloads"
        manifest_path = source_dir / "manifest.jsonl"

        dep_name = sanitize_stem(Path(prefix.rstrip("/")).name)
        dep_download_dir = source_downloads_dir / dep_name
        dep_download_dir.mkdir(parents=True, exist_ok=True)
        source_audio_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== {source_name}/{dep_name} ===")
        print(f"Listing objects under: {prefix}")
        objects = _list_objects(bucket=bucket, prefix=prefix)
        audio_files = [o for o in objects if _is_audio_object(str(o.get("name", "")))]

        if not audio_files:
            print("No audio files found, skipping deployment.")
            continue

        if only_new_files:
            new_only: List[Dict[str, int | str]] = []
            for item in audio_files:
                object_name = str(item.get("name", ""))
                size = int(item.get("size", 0))
                src_name = Path(object_name).name
                local_src = dep_download_dir / src_name
                stem = sanitize_stem(f"{dep_name}_{Path(src_name).stem}")

                already_downloaded = (
                    local_src.exists() and local_src.stat().st_size == size
                )
                already_processed = _processed_audio_exists(source_audio_dir, stem)

                if already_downloaded or already_processed:
                    continue

                new_only.append(item)

            audio_files = new_only
            if not audio_files:
                print("No new files left for this deployment, skipping.")
                manifest_entries = write_manifest(
                    audio_dir=source_audio_dir,
                    manifest_path=manifest_path,
                )
                print(
                    f"Manifest entries: {manifest_entries} ({manifest_path.resolve()})"
                )
                continue

        rng = random.Random(seed + dep_idx)
        selected = _pick_random_files(
            files=audio_files,
            target_seconds=target_seconds,
            sample_rate_hz=sample_rate_hz,
            sample_bits=sample_bits,
            channels=channels,
            rng=rng,
            max_files=max_files_per_deployment,
        )
        print(f"Selected files: {len(selected)}")

        for file_idx, item in enumerate(selected):
            object_name = str(item.get("name"))
            size = int(item.get("size", 0))
            src_name = Path(object_name).name
            local_src = dep_download_dir / src_name
            stem = sanitize_stem(f"{dep_name}_{Path(src_name).stem}")

            if only_new_files and _processed_audio_exists(source_audio_dir, stem):
                print(f"Already processed: {src_name}")
                continue

            if not local_src.exists() or local_src.stat().st_size != size:
                print(f"Downloading [{file_idx + 1}/{len(selected)}]: {src_name}")
                try:
                    _download_file(
                        bucket=bucket,
                        object_name=object_name,
                        dst_path=local_src,
                        expected_size=size,
                    )
                    total_downloaded_bytes += size
                except Exception as exc:
                    print(f"Skip '{src_name}' due to download error: {exc}")
                    continue
            else:
                print(f"Already downloaded: {src_name}")

            try:
                process_large_audio(
                    src_path=local_src,
                    out_dir=source_audio_dir,
                    stem_base=stem,
                    total_seconds_ref=total_seconds,
                    sr_target=sr_target,
                    chunk_sec=chunk_sec,
                )
            except Exception as exc:
                print(f"Error processing '{src_name}': {exc}")
                continue

            if delete_downloaded:
                try:
                    local_src.unlink(missing_ok=True)
                except Exception as exc:
                    print(
                        f"Warning: failed to delete downloaded file '{src_name}': {exc}"
                    )

        manifest_entries = write_manifest(
            audio_dir=source_audio_dir,
            manifest_path=manifest_path,
        )
        print(f"Manifest entries: {manifest_entries} ({manifest_path.resolve()})")

    print("\nFinished NOAA ONMS sampling")
    print(f"Total duration (output WAVs): {total_seconds[0] / 3600:.2f} h")
    print(f"Downloaded data: {total_downloaded_bytes / (1024**3):.2f} GB")
    print(f"Dataset dir: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
