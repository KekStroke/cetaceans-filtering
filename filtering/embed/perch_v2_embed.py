"""Compute Perch embeddings for audio files and export them to the filesystem.

Outputs:
  - Hoplite DB in db_path (hoplite.sqlite + usearch.index)
  - embeddings.npy in export_dir
  - manifest.csv in export_dir
  - optionally one .npy per clip in export_dir/per_clip_npy/
"""

from __future__ import annotations

import csv
from pathlib import Path

import hydra
import numpy as np
from ml_collections import config_dict
from omegaconf import DictConfig

from perch_hoplite.agile import embed
from perch_hoplite.agile import source_info
from perch_hoplite.db import db_loader
from perch_hoplite.db import sqlite_usearch_impl
from perch_hoplite.zoo import model_configs


def build_worker(
    audio_dir: str,
    file_glob: str,
    dataset_name: str,
    db_path: str,
    model_name: str,
    shard_len_s: float | None,
    max_shards_per_file: int | None,
) -> tuple[embed.EmbedWorker, object]:
    audio_sources = source_info.AudioSources(
        (
            source_info.AudioSourceConfig(
                dataset_name=dataset_name,
                base_path=audio_dir,
                file_glob=file_glob,
                min_audio_len_s=1.0,
                target_sample_rate_hz=-2,
                shard_len_s=shard_len_s,
                max_shards_per_file=max_shards_per_file,
            ),
        )
    )

    preset = model_configs.get_preset_model_config(model_name)

    model_config = embed.ModelConfig(
        model_key=preset.model_key,
        embedding_dim=preset.embedding_dim,
        model_config=preset.model_config,
    )

    db_cfg = db_loader.DBConfig(
        db_key="sqlite_usearch",
        db_config=config_dict.ConfigDict(
            {
                "db_path": db_path,
                "usearch_cfg": sqlite_usearch_impl.get_default_usearch_config(
                    preset.embedding_dim
                ),
            }
        ),
    )
    db = db_cfg.load_db()

    worker = embed.EmbedWorker(
        audio_sources=audio_sources,
        model_config=model_config,
        db=db,
    )
    return worker, db


def maybe_drop_existing_db(db_path: Path) -> None:
    if not db_path.exists():
        return
    for name in (
        "hoplite.sqlite",
        "hoplite.sqlite-shm",
        "hoplite.sqlite-wal",
        "usearch.index",
    ):
        p = db_path / name
        if p.exists():
            p.unlink()


def export_db_embeddings(
    db,
    export_dir: Path,
    save_per_clip_npy: bool,
) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    per_clip_dir = export_dir / "per_clip_npy"
    if save_per_clip_npy:
        per_clip_dir.mkdir(parents=True, exist_ok=True)

    windows = db.get_all_windows(include_embedding=False)
    rows: list[dict[str, str | int | float]] = []
    vectors: list[np.ndarray] = []

    for row_idx, window in enumerate(windows):
        recording = db.get_recording(window.recording_id)
        embedding = db.get_embedding(window.id)
        vectors.append(np.asarray(embedding, dtype=np.float32))

        start_s = float(window.offsets[0])
        end_s = float(window.offsets[1])
        out_row = {
            "row_index": row_idx,
            "window_id": int(window.id),
            "recording_id": int(window.recording_id),
            "filename": recording.filename,
            "start_s": start_s,
            "end_s": end_s,
        }
        rows.append(out_row)

        if save_per_clip_npy:
            safe_name = recording.filename.replace("/", "__")
            clip_fp = (
                per_clip_dir
                / f"{row_idx:06d}__{safe_name}__{start_s:.3f}_{end_s:.3f}.npy"
            )
            np.save(clip_fp, np.asarray(embedding, dtype=np.float32))

    if vectors:
        X = np.stack(vectors, axis=0)
    else:
        X = np.zeros((0, 0), dtype=np.float32)

    np.save(export_dir / "embeddings.npy", X)

    with open(export_dir / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["row_index", "window_id", "recording_id", "filename", "start_s", "end_s"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved embeddings matrix: {export_dir / 'embeddings.npy'}")
    print(f"Saved manifest:          {export_dir / 'manifest.csv'}")
    if save_per_clip_npy:
        print(f"Saved per-clip .npy:     {per_clip_dir}")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(config: DictConfig) -> None:
    cfg = config["perch_embeddings"]

    audio_dir = hydra.utils.to_absolute_path(str(cfg["audio_dir"]))
    db_path = Path(hydra.utils.to_absolute_path(str(cfg["db_path"])))
    export_dir = Path(hydra.utils.to_absolute_path(str(cfg["export_dir"])))

    db_path.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    if cfg["drop_existing_db"]:
        maybe_drop_existing_db(db_path)

    worker, db = build_worker(
        audio_dir=audio_dir,
        file_glob=str(cfg["file_glob"]),
        dataset_name=str(cfg["dataset_name"]),
        db_path=str(db_path),
        model_name=str(cfg["model_name"]),
        shard_len_s=cfg["shard_len_s"],
        max_shards_per_file=cfg["max_shards_per_file"],
    )

    worker.process_all(target_dataset_name=str(cfg["dataset_name"]))
    export_db_embeddings(db, export_dir, bool(cfg["save_per_clip_npy"]))

    print(f"\nHoplite DB directory: {db_path}")
    print(f"Export directory:     {export_dir}")


if __name__ == "__main__":
    main()
