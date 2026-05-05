"""
Download hydrophone archive files from Ocean Networks Canada (Oceans 3.0 API),
convert via audio_saver, and record a per-target manifest.jsonl.

Layout per target subdirectory (split by location or device):

  <raw_datasets_path>/<onc_output_dir>/<KEY>/
      audio/*.wav
      manifest.jsonl

Token: env ONC_TOKEN, or sources.onc.token in config (see
https://oceannetworkscanada.github.io/api-python-client/).

Example (in configs/data_loading/data_loading.yaml, under the nested ``sources`` block)::

    sources:
      onc:
        token: null            # omit and use ONC_TOKEN
        output_dir_name: onc
        tmp_dir_name: _tmp_onc
        split_by: auto         # auto | location_code | device_code
        delete_downloaded_after: true
        progress_every: 50
        device_category_code: HYDROPHONE
        default_archive_params: {}   # optional shared camelCase filters
        extra_params: {}              # optional global extras
        targets:
          - location_code: SCVIP
            date_from: "2021-05-01"
            date_to: "2021-05-02"
            extension: flac
            include_diverted: false
            max_files: null
            subfolder: null            # override output folder name
            device_category_code: null  # per-target category override
            extra_params: {}             # merges into getArchivefile params
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Set

import hydra
from audio_saver import process_large_audio, sanitize_stem
from manifest_utils import append_manifest_records
from omegaconf import DictConfig, OmegaConf
from onc import ONC
from tqdm import tqdm


def _ensure_iso_utc(s: str) -> str:
    """Accept 'YYYY-MM-DD' or ISO; return UTC ISO ending in 'Z'."""
    if "T" not in s:
        dt = datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
        return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    return s.replace("+00:00", "Z")


def _to_plain(obj: Optional[Any]) -> Any:
    """Convert OmegaConf nodes to plain nested dict/list; shallow-copy plain dict."""
    if obj is None:
        return {}
    if OmegaConf.is_config(obj):
        return OmegaConf.to_container(obj, resolve=True)
    if isinstance(obj, dict):
        return dict(obj)
    return obj


def _resolve_split_subfolder(target: Mapping[str, Any], split_by: str) -> str:
    """
    Directory name under ONC output root for this target.
    split_by: 'location_code' | 'device_code' | 'auto'
        auto: location_code if set, else device_code, else 'unknown'.
    """
    if target.get("subfolder"):
        return str(target["subfolder"]).strip() or "unknown"

    lc = target.get("location_code")
    dc = target.get("device_code")

    if split_by == "location_code":
        return str(lc).strip() if lc else str(dc).strip() if dc else "unknown"
    if split_by == "device_code":
        return str(dc).strip() if dc else str(lc).strip() if lc else "unknown"

    # auto
    if lc:
        return str(lc).strip()
    if dc:
        return str(dc).strip()
    return "unknown"


def _resolve_token(onc_cfg: Mapping[str, Any]) -> str:
    token = onc_cfg.get("token") or os.environ.get("ONC_TOKEN")
    if not token:
        raise RuntimeError(
            "Missing ONC API token: set environment variable ONC_TOKEN or sources.onc.token in config."
        )
    return str(token)


def _build_archive_params(
    target: Mapping[str, Any],
    onc_cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Merge: default_archive_params → global/per-target extra_params → target core fields win.
    extra_params accepts any Oceans 3.0 archive filter key (camelCase strings).
    """
    t = _to_plain(target)
    g = _to_plain(onc_cfg)
    merged: MutableMapping[str, Any] = {}
    merged.update(g.get("default_archive_params") or {})
    merged.update(g.get("extra_params") or {})
    merged.update(t.get("extra_params") or {})

    merged["deviceCategoryCode"] = (
        t.get("device_category_code") or g.get("device_category_code") or "HYDROPHONE"
    )
    merged["dateFrom"] = _ensure_iso_utc(str(t["date_from"]))
    merged["dateTo"] = _ensure_iso_utc(str(t["date_to"]))
    merged["extension"] = str(t.get("extension") or "flac").lower()
    if t.get("location_code"):
        merged["locationCode"] = t["location_code"]
    if t.get("device_code"):
        merged["deviceCode"] = t["device_code"]
    return dict(merged)


def _wav_paths_snapshot(dir_path: Path) -> Set[Path]:
    if not dir_path.is_dir():
        return set()
    return set(dir_path.glob("*.wav"))


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    dl = cfg["data_loading"]
    sc = dl.get("sources") or cfg.get("sources") or {}
    onc_plain = _to_plain((sc.get("onc") if sc else {}) or {})
    onc_cfg: Dict[str, Any] = onc_plain if isinstance(onc_plain, dict) else {}

    raw_root = Path(dl["raw_datasets_path"])
    onc_root_name = str(onc_cfg.get("output_dir_name") or "onc")
    tmp_name = str(onc_cfg.get("tmp_dir_name") or "_tmp_onc")

    out_root = raw_root / onc_root_name
    out_root.mkdir(parents=True, exist_ok=True)

    tmp_dir = raw_root / tmp_name
    tmp_dir.mkdir(parents=True, exist_ok=True)

    split_by = str(onc_cfg.get("split_by") or "auto").lower()
    if split_by not in ("auto", "location_code", "device_code"):
        raise ValueError(
            f"sources.onc.split_by must be auto|location_code|device_code, got {split_by!r}"
        )

    delete_source_after = bool(onc_cfg.get("delete_downloaded_after", True))
    progress_every = int(onc_cfg.get("progress_every", 50))

    sr_target = dl["raw_sample_rate"]
    chunk_sec = float(dl["raw_segment_duration"])

    targets = onc_cfg.get("targets") or []
    if not targets:
        print(
            "sources.onc.targets is empty; nothing to do. "
            "Put targets under data_loading.sources.onc.targets in data_loading.yaml "
            "(Hydra nests that file under cfg.data_loading, not the config root)."
        )
        return

    token = _resolve_token(onc_cfg)
    onc = ONC(token, outPath=str(tmp_dir))

    total_seconds: List[float] = [0.0]
    processed = 0

    for target in targets:
        t_plain = _to_plain(target)
        t: Dict[str, Any] = t_plain if isinstance(t_plain, dict) else {}
        params = _build_archive_params(t, onc_cfg)

        include_diverted = bool(t.get("include_diverted", False))
        max_files = t.get("max_files")

        sub_name = sanitize_stem(_resolve_split_subfolder(t, split_by))
        target_root = out_root / sub_name
        audio_dir = target_root / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = target_root / "manifest.jsonl"

        res = onc.getArchivefile(params, allPages=True)
        files = res["files"] if isinstance(res, dict) and "files" in res else res

        if not include_diverted:
            files = [fn for fn in files if "-HPF" not in fn and "-LPF" not in fn]

        if max_files is not None:
            files = files[: int(max_files)]

        lc = t.get("location_code")
        dc = t.get("device_code")
        key_desc = lc or dc or sub_name

        if not files:
            print(
                f"No archived {params['extension'].upper()} files for {key_desc} {params['dateFrom']}..{params['dateTo']}"
            )
            continue

        print(
            f"Found {len(files)} files for {key_desc} ({params['dateFrom']}..{params['dateTo']}, {params['extension']})"
        )

        base_record = {
            "location_code": lc,
            "device_code": dc,
            "date_from": params["dateFrom"],
            "date_to": params["dateTo"],
            "extension": params["extension"],
            "device_category_code": params.get("deviceCategoryCode"),
            "split_subfolder": sub_name,
        }

        pbar = tqdm(total=len(files), desc=f"ONC {key_desc}")
        for fname in files:
            local_path: Optional[Path] = None
            try:
                onc.downloadArchivefile(fname, overwrite=False)
                local_path = tmp_dir / fname
                if not local_path.exists():
                    hits = list(tmp_dir.rglob(Path(fname).name))
                    if hits:
                        local_path = hits[0]
                if local_path is None or not local_path.exists():
                    raise FileNotFoundError(f"downloaded file not found: {fname}")

                stem_base = f"onc_{sanitize_stem(local_path.stem)}"
                before = _wav_paths_snapshot(audio_dir)
                process_large_audio(
                    src_path=local_path,
                    out_dir=audio_dir,
                    stem_base=stem_base,
                    total_seconds_ref=total_seconds,
                    sr_target=sr_target,
                    chunk_sec=chunk_sec,
                )
                after = _wav_paths_snapshot(audio_dir)
                new_wavs = sorted(after - before, key=lambda p: str(p))
                if new_wavs:
                    append_manifest_records(
                        manifest_path=manifest_path,
                        audio_paths=new_wavs,
                        extra_fields={
                            **base_record,
                            "archive_file": fname,
                        },
                    )
                processed += 1
            except Exception as e:
                print(f"error processing {fname}: {e}")
            finally:
                if delete_source_after and local_path is not None:
                    try:
                        local_path.unlink(missing_ok=True)
                    except OSError:
                        pass

            pbar.update(1)
            if processed % progress_every == 0:
                pbar.set_postfix_str(f"total {total_seconds[0] / 3600:.2f} h")

        pbar.close()

    print("\nFinished")
    print(f"Processed source files: {processed}")
    print(f"Total duration (output WAVs): {total_seconds[0] / 3600:.2f} h")
    print(f"Output root: {out_root.resolve()}")


if __name__ == "__main__":
    main()
