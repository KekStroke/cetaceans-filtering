import random
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
from parallel_utils import iter_threaded
from tqdm import tqdm

AUDIO_EXTS = (".wav", ".WAV")
PacificObject = Tuple[str, str]


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
) -> Iterable[PacificObject]:
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
) -> Iterable[PacificObject]:
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
) -> Iterable[PacificObject]:
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


def _object_year_month(tier: str, bucket: str, key: str) -> Tuple[int, str]:
    if tier == "256khz":
        year = int(bucket.rsplit("-", 1)[-1])
        month = key.split("/", 1)[0].zfill(2)
        return year, month

    parts = key.split("/", 2)
    if len(parts) < 2:
        raise ValueError(f"cannot parse year/month from Pacific Sound key: {key}")
    return int(parts[0]), parts[1].zfill(2)


def _estimated_file_hours(tier: str) -> float:
    if tier == "256khz":
        return 10.0 / 60.0
    if tier in {"16khz", "2khz"}:
        return 24.0
    raise ValueError("tier must be one of: '256khz', '16khz', '2khz'")


def _audio_subdir(out_dir: Path, tier: str, bucket: str, key: str) -> Path:
    year, month = _object_year_month(tier=tier, bucket=bucket, key=key)
    return out_dir / f"{year:04d}" / month / "audio"


def _select_hours_per_month(
    pairs: Iterable[PacificObject],
    tier: str,
    hours_per_month: Optional[float],
    seed: int,
) -> List[PacificObject]:
    grouped: dict[Tuple[int, str], List[PacificObject]] = {}
    for bucket, key in pairs:
        grouped.setdefault(
            _object_year_month(tier=tier, bucket=bucket, key=key), []
        ).append(
            (
                bucket,
                key,
            )
        )

    selected: List[PacificObject] = []
    file_hours = _estimated_file_hours(tier)
    for idx, ym in enumerate(sorted(grouped)):
        month_pairs = grouped[ym]
        if hours_per_month is None:
            chosen = sorted(month_pairs)
        else:
            n_files = max(1, round(float(hours_per_month) / file_hours))
            rng = random.Random(seed + idx)
            chosen = list(month_pairs)
            rng.shuffle(chosen)
            chosen = sorted(chosen[: min(n_files, len(chosen))])
        selected.extend(chosen)
        approx_hours = len(chosen) * file_hours
        print(
            f"Selected {len(chosen)}/{len(month_pairs)} files for "
            f"{ym[0]}-{ym[1]} (~{approx_hours:.1f} h)"
        )
    return selected


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    dl = cfg["data_loading"]
    pacific_cfg = dl["sources"]["pacific_sound"]
    out_root = Path(dl["raw_datasets_path"])
    out_dir = out_root / str(pacific_cfg.get("output_dir_name", "pacific_sound"))
    manifest_path = out_dir / "manifest.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)

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
    hours_per_month_cfg = pacific_cfg.get("hours_per_month")
    hours_per_month = (
        None
        if hours_per_month_cfg is None
        or str(hours_per_month_cfg).strip().lower() in {"none", "null", ""}
        else float(hours_per_month_cfg)
    )
    random_seed = int(pacific_cfg.get("random_seed", 42))

    sr_target = dl["raw_sample_rate"]
    chunk_sec = float(dl["raw_segment_duration"])
    min_sample_rate = resolve_min_sample_rate(
        raw_sample_rate=dl.get("raw_sample_rate"),
        raw_skip_below_sample_rate=bool(dl.get("raw_skip_below_sample_rate", False)),
    )
    download_workers = max(1, int(pacific_cfg.get("download_workers", 1)))

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    pairs = _select_hours_per_month(
        pairs=iter_pacific_sound_objects(s3, tier=tier, years=years, months=months),
        tier=tier,
        hours_per_month=hours_per_month,
        seed=random_seed,
    )

    total_seconds = [0.0]
    processed = 0

    # tmp area
    tmp_dir = out_root / "_tmp" / f"pacific-sound-{tier}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    def download_and_process(pair: PacificObject) -> tuple[bool, float, str]:
        bucket, key = pair
        local_tmp = tmp_dir / bucket / key
        local_tmp.parent.mkdir(parents=True, exist_ok=True)

        try:
            s3.download_file(bucket, key, str(local_tmp))
        except Exception as e:
            return False, 0.0, f"skip s3://{bucket}/{key}: {e}"

        stem_bits = [bucket, Path(key).with_suffix("").as_posix().replace("/", "_")]
        rel_stem = sanitize_stem("_".join(stem_bits))
        out_audio_dir = _audio_subdir(
            out_dir=out_dir,
            tier=tier,
            bucket=bucket,
            key=key,
        )
        seconds = [0.0]

        try:
            process_large_audio(
                src_path=local_tmp,
                out_dir=out_audio_dir,
                stem_base=rel_stem,
                total_seconds_ref=seconds,
                sr_target=sr_target,
                chunk_sec=chunk_sec,
                min_sample_rate=min_sample_rate,
            )
        except Exception as e:
            return False, 0.0, f"error processing {bucket}/{key}: {e}"
        finally:
            try:
                local_tmp.unlink(missing_ok=True)
            except Exception:
                pass

        return True, seconds[0], ""

    try:
        print(f"Download workers: {download_workers}")
        for ok, seconds, message in tqdm(
            iter_threaded(download_and_process, pairs, download_workers),
            desc=f"Downloading & processing pacific-sound-{tier}",
            total=len(pairs),
        ):
            if message:
                print(message)
            if ok:
                total_seconds[0] += seconds
                processed += 1

    finally:
        # clean tmp
        try:
            shutil.rmtree(tmp_dir)
        except Exception:
            pass

    month_dirs = sorted(
        {
            _audio_subdir(out_dir=out_dir, tier=tier, bucket=bucket, key=key).parent
            for bucket, key in pairs
        }
    )
    for month_dir in month_dirs:
        entries = write_manifest(
            audio_dir=month_dir / "audio",
            manifest_path=month_dir / "manifest.jsonl",
        )
        month_manifest_path = month_dir / "manifest.jsonl"
        print(f"Manifest entries: {entries} ({month_manifest_path.resolve()})")

    manifest_entries = write_manifest(audio_dir=out_dir, manifest_path=manifest_path)
    print(f"Manifest entries: {manifest_entries} ({manifest_path.resolve()})")

    print("\nFinished")
    print(f"Processed source files: {processed}")
    print(f"Total duration (output WAVs): {total_seconds[0] / 3600:.2f} h")
    print(f"Audio dir: {audio_dir.resolve()}")


if __name__ == "__main__":
    main()
