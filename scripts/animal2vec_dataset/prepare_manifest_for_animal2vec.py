#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import h5py
import numpy as np


def parse_manifest_line(line: str) -> tuple[str, int] | None:
    line = line.strip()
    if not line:
        return None

    if "\t" in line:
        parts = line.split("\t")
        if len(parts) != 2:
            raise ValueError(f"Bad manifest line with tabs: {line!r}")
        rel_path, n_frames = parts
    else:
        rel_path, n_frames = line.rsplit(maxsplit=1)

    return rel_path, int(n_frames)


def make_empty_label_file(h5_path: Path) -> None:
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    if h5_path.exists():
        return

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("start_time_lbl", data=np.array([], dtype=np.float32))
        f.create_dataset("end_time_lbl", data=np.array([], dtype=np.float32))
        f.create_dataset("start_frame_lbl", data=np.array([], dtype=np.int64))
        f.create_dataset("end_frame_lbl", data=np.array([], dtype=np.int64))
        f.create_dataset("lbl", data=np.array([], dtype="S"))
        f.create_dataset("lbl_cat", data=np.array([], dtype=np.int64))
        f.create_dataset("foc", data=np.array([], dtype=np.int64))


def link_audio(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return

    if mode == "symlink":
        os.symlink(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown link mode: {mode}")


def convert_manifest(
    input_manifest: Path,
    output_root: Path,
    output_split_name: str,
    mode: str,
    limit: int | None = None,
) -> None:
    input_manifest = input_manifest.resolve()
    output_root = output_root.resolve()

    wav_root = output_root / "wav"
    lbl_root = output_root / "lbl"
    manifest_root = output_root / "manifest"

    manifest_root.mkdir(parents=True, exist_ok=True)
    wav_root.mkdir(parents=True, exist_ok=True)
    lbl_root.mkdir(parents=True, exist_ok=True)

    output_manifest = manifest_root / f"{output_split_name}.tsv"
    n_total = 0
    n_missing_audio = 0

    with input_manifest.open("r", encoding="utf-8") as fin, output_manifest.open("w", encoding="utf-8") as fout:
        old_root = Path(fin.readline().strip()).resolve()
        fout.write(str(output_root) + "\n")

        for line in fin:
            parsed = parse_manifest_line(line)
            if parsed is None:
                continue

            rel_path, n_frames = parsed
            old_audio = old_root / rel_path
            if not old_audio.exists():
                print(f"[MISSING AUDIO] {old_audio}")
                n_missing_audio += 1
                continue

            new_audio_rel = Path("wav") / rel_path
            new_audio = output_root / new_audio_rel
            new_label = (lbl_root / rel_path).with_suffix(".h5")

            link_audio(old_audio, new_audio, mode=mode)
            make_empty_label_file(new_label)
            fout.write(f"{new_audio_rel.as_posix()}\t{n_frames}\n")
            n_total += 1

            if limit is not None and n_total >= limit:
                break

    print("Done")
    print(f"input_manifest:  {input_manifest}")
    print(f"old_root:        {old_root}")
    print(f"output_root:     {output_root}")
    print(f"output_manifest: {output_manifest}")
    print(f"linked/copied:   {n_total}")
    print(f"missing audio:   {n_missing_audio}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--split-name", default="pretrain")
    parser.add_argument("--mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    convert_manifest(
        input_manifest=args.input_manifest,
        output_root=args.output_root,
        output_split_name=args.split_name,
        mode=args.mode,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
