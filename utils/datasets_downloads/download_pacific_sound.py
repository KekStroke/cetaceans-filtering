import shutil
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import boto3
import hydra
from audio_saver import process_large_audio, resolve_min_sample_rate, sanitize_stem
from botocore import UNSIGNED
from botocore.client import Config
from manifest_utils import write_manifest
from omegaconf import DictConfig
from tqdm import tqdm

AUDIO_EXTS = (".wav", ".WAV")


def _norm_months(months: Sequence[Union[int, str]]) -> List[str]:
    out = []
    for m in months:
        if isinstance(m, int):
            out.append(f"{m:02d}")
        else:
            mm = m.strip().zfill(2)
            out.append(mm[-2:])
    return out


def list_256khz_keys(
    s3, years: Sequence[int], months: Optional[Sequence[Union[int, str]]] = None
) -> Iterable[Tuple[str, str]]:
    """
    Yield (bucket, key) for 256 kHz archive:
      bucket = pacific-sound-256khz-YYYY
      keys   = 'MM/<files...>' (10-minute WAVs)
    """
    months = _norm_months(months or [f"{i:02d}" for i in range(1, 13)])
    for y in sorted(set(int(y) for y in years)):
        bucket = f"pacific-sound-256khz-{y}"
        for mm in months:
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket, Prefix=f"{mm}/"):
                for obj in page.get("Contents", []) or []:
                    key = obj["Key"]
                    if key.lower().endswith((".wav",)):
                        yield bucket, key


def list_decimated_keys(
    s3,
    tier: str,
    years: Sequence[int],
    months: Optional[Sequence[Union[int, str]]] = None,
) -> Iterable[Tuple[str, str]]:
    """
    Yield (bucket, key) for decimated archives:
      tier '16khz' -> bucket 'pacific-sound-16khz'
      tier '2khz'  -> bucket 'pacific-sound-2khz'
      keys         -> 'YYYY/MM/<files...>' (daily WAVs)
    """
    tier = tier.lower()
    assert tier in {"16khz", "2khz"}
    bucket = f"pacific-sound-{tier}"
    months = _norm_months(months or [f"{i:02d}" for i in range(1, 13)])
    for y in sorted(set(int(y) for y in years)):
        for mm in months:
            prefix = f"{y:04d}/{mm}/"
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get("Contents", []) or []:
                    key = obj["Key"]
                    if key.lower().endswith((".wav",)):
                        yield bucket, key


def iter_pacific_sound_objects(
    s3,
    tier: str,
    years: Sequence[int],
    months: Optional[Sequence[Union[int, str]]] = None,
) -> Iterable[Tuple[str, str]]:
    if tier.lower() == "256khz":
        yield from list_256khz_keys(s3, years=years, months=months)
    elif tier.lower() in {"16khz", "2khz"}:
        yield from list_decimated_keys(s3, tier=tier, years=years, months=months)
    else:
        raise ValueError("tier must be one of: '256khz', '16khz', '2khz'")


def _config_list(value, default: Optional[Sequence] = None) -> Optional[List]:
    if value is None or str(value).strip().lower() in {"none", "null", ""}:
        return None if default is None else list(default)
    return list(value)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    dl = cfg["data_loading"]
    pacific_cfg = dl["sources"]["pacific_sound"]
    out_root = Path(dl["raw_datasets_path"])
    out_dir = out_root / str(pacific_cfg.get("output_dir_name", "pacific_sound"))
    audio_dir = out_dir / "audio"
    manifest_path = out_dir / "manifest.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    tier = str(pacific_cfg.get("tier", "256khz")).lower()
    years = [
        int(y)
        for y in (
            _config_list(pacific_cfg.get("years", [2016]), [2016])
            or []
        )
    ]
    if not years:
        raise ValueError(
            "data_loading.sources.pacific_sound.years must contain at least one year"
        )
    months = _config_list(pacific_cfg.get("months"))

    sr_target = dl["raw_sample_rate"]
    chunk_sec = float(dl["raw_segment_duration"])
    min_sample_rate = resolve_min_sample_rate(
        raw_sample_rate=dl.get("raw_sample_rate"),
        raw_skip_below_sample_rate=bool(dl.get("raw_skip_below_sample_rate", False)),
    )

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    # build iterator of (bucket, key)
    pairs = iter_pacific_sound_objects(s3, tier=tier, years=years, months=months)

    total_seconds = [0.0]
    processed = 0

    # tmp area
    tmp_dir = out_root / "_tmp" / f"pacific-sound-{tier}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        for bucket, key in tqdm(
            pairs, desc=f"Downloading & processing pacific-sound-{tier}"
        ):
            # mirror path under tmp
            local_tmp = tmp_dir / bucket / key
            local_tmp.parent.mkdir(parents=True, exist_ok=True)

            # download
            try:
                s3.download_file(bucket, key, str(local_tmp))
            except Exception as e:
                print(f"skip s3://{bucket}/{key}: {e}")
                continue

            # output stem: include bucket, year/mo if present, and filename stem
            stem_bits = [bucket, Path(key).with_suffix("").as_posix().replace("/", "_")]
            rel_stem = sanitize_stem("_".join(stem_bits))

            try:
                process_large_audio(
                    src_path=local_tmp,
                    out_dir=audio_dir,
                    stem_base=rel_stem,
                    total_seconds_ref=total_seconds,
                    sr_target=sr_target,
                    chunk_sec=chunk_sec,
                    min_sample_rate=min_sample_rate,
                )
            except Exception as e:
                print(f"error processing {bucket}/{key}: {e}")

            processed += 1

            # remove tmp
            try:
                local_tmp.unlink(missing_ok=True)
            except Exception:
                pass

    finally:
        # clean tmp
        try:
            shutil.rmtree(tmp_dir)
        except Exception:
            pass

    manifest_entries = write_manifest(audio_dir=audio_dir, manifest_path=manifest_path)
    print(f"Manifest entries: {manifest_entries} ({manifest_path.resolve()})")

    print("\nFinished")
    print(f"Processed source files: {processed}")
    print(f"Total duration (output WAVs): {total_seconds[0] / 3600:.2f} h")
    print(f"Audio dir: {audio_dir.resolve()}")


if __name__ == "__main__":
    main()
