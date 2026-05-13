import io
from pathlib import Path

import hydra
import soundfile as sf
from audio_saver import (
    process_array_audio,
    process_large_audio,
    resolve_min_sample_rate,
    sanitize_stem,
)
from datasets import Audio, concatenate_datasets, load_dataset
from manifest_utils import write_manifest
from omegaconf import DictConfig
from tqdm import tqdm


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(config: DictConfig):
    dl = config["data_loading"]
    watkins_cfg = dl["sources"]["watkins"]
    out_root = Path(dl["raw_datasets_path"])
    out_dir = out_root / str(watkins_cfg.get("output_dir_name", "watkins"))
    manifest_path = out_dir / "manifest.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)

    hf_name = str(watkins_cfg.get("hf_name", "confit/wmms-parquet"))
    splits = list(watkins_cfg.get("splits", ["train", "test"]))
    sr_target = None
    chunk_sec = float(dl["raw_segment_duration"])
    min_sample_rate = resolve_min_sample_rate(
        raw_sample_rate=dl.get("raw_sample_rate"),
        raw_skip_below_sample_rate=bool(dl.get("raw_skip_below_sample_rate", False)),
    )
    progress_every = int(watkins_cfg.get("progress_every", 50))

    total_seconds = [0.0]
    processed = 0

    # Load + concat splits
    dsets = [load_dataset(hf_name, split=s) for s in splits]
    ds_all = dsets[0] if len(dsets) == 1 else concatenate_datasets(dsets)

    # IMPORTANT: no torchcodec; we’ll read bytes ourselves
    ds_all = ds_all.cast_column("audio", Audio(decode=False))

    pbar = tqdm(total=len(ds_all), desc="Processing WMMS")
    for i, ex in enumerate(ds_all):
        a = ex.get("audio")
        if not a:
            pbar.update(1)
            continue

        species = ex.get("species") or "unknown"
        stem = f"wmms_{i:05d}_{sanitize_stem(species)}"

        try:
            # 1) Prefer embedded bytes (Parquet stores audio content here)
            if isinstance(a, dict) and a.get("bytes") is not None:
                data, sr = sf.read(io.BytesIO(a["bytes"]), always_2d=False)
                process_array_audio(
                    data=data,
                    sr=int(sr),
                    out_dir=out_dir,
                    stem_base=stem,
                    total_seconds_ref=total_seconds,
                    sr_target=sr_target,
                    chunk_sec=chunk_sec,
                    min_sample_rate=min_sample_rate,
                )

            # 2) Fallback: only use path if it actually exists on disk
            elif isinstance(a, dict) and a.get("path") and Path(a["path"]).exists():
                process_large_audio(
                    src_path=Path(a["path"]),
                    out_dir=out_dir,
                    stem_base=stem,
                    total_seconds_ref=total_seconds,
                    sr_target=sr_target,
                    chunk_sec=chunk_sec,
                    min_sample_rate=min_sample_rate,
                )
            else:
                # Nothing usable for this row
                print(
                    f"skip row {i}: no bytes and missing local file '{a.get('path') if isinstance(a, dict) else None}'"
                )

        except Exception as e:
            print(f"error processing row {i}: {e}")

        processed += 1
        pbar.update(1)
        if processed % progress_every == 0:
            pbar.set_postfix_str(f"total {total_seconds[0] / 3600:.2f} h")
    pbar.close()

    manifest_entries = write_manifest(audio_dir=out_dir, manifest_path=manifest_path)
    print(f"Manifest entries: {manifest_entries} ({manifest_path.resolve()})")

    print("\nFinished")
    print(f"Processed source clips: {processed}")
    print(f"Total duration (output WAVs): {total_seconds[0] / 3600:.2f} h")
    print(f"Audio dir: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
