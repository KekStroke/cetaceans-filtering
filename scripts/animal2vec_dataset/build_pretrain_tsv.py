#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from os import scandir
from pathlib import Path


def find_manifests(root: Path, wav_limit: int) -> list[Path]:
    manifests = []

    def walk(dir_path: Path) -> None:
        manifest_path = dir_path / "manifest.jsonl"
        if manifest_path.is_file():
            manifests.append(manifest_path)

        wav_count = 0
        subdirs = []
        try:
            with scandir(dir_path) as entries:
                for entry in entries:
                    try:
                        if entry.is_file(follow_symlinks=False):
                            if entry.name.lower().endswith(".wav"):
                                wav_count += 1
                                if wav_count > wav_limit:
                                    return
                        elif entry.is_dir(follow_symlinks=False):
                            subdirs.append(Path(entry.path))
                    except OSError:
                        continue
        except PermissionError:
            print(f"Warning: permission denied: {dir_path}", file=sys.stderr)
            return

        for subdir in sorted(subdirs):
            walk(subdir)

    if root.is_file():
        if root.name == "manifest.jsonl":
            manifests.append(root)
        else:
            print(f"Warning: not a manifest.jsonl file: {root}", file=sys.stderr)
    elif root.is_dir():
        walk(root)
    else:
        print(f"Warning: path does not exist: {root}", file=sys.stderr)

    return manifests


def resolve_audio_path(manifest_path: Path, audio_filepath: str) -> Path:
    audio_path = Path(audio_filepath)
    if audio_path.is_absolute():
        return audio_path.resolve()
    return (manifest_path.parent / audio_path).resolve()


def iter_manifest_records(manifest_path: Path, default_sample_rate: int | None):
    with manifest_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: invalid JSON: {manifest_path}:{line_num}", file=sys.stderr)
                continue

            audio_filepath = item.get("audio_filepath")
            if not audio_filepath:
                print(f"Warning: missing audio_filepath: {manifest_path}:{line_num}", file=sys.stderr)
                continue

            try:
                duration = float(item["duration"])
            except KeyError:
                print(f"Warning: missing duration: {manifest_path}:{line_num}", file=sys.stderr)
                continue
            except (TypeError, ValueError):
                print(f"Warning: invalid duration: {manifest_path}:{line_num}", file=sys.stderr)
                continue

            sample_rate = item.get("sample_rate", default_sample_rate)
            if sample_rate is None:
                print(f"Warning: missing sample_rate: {manifest_path}:{line_num}", file=sys.stderr)
                continue

            try:
                sample_rate = int(sample_rate)
            except (TypeError, ValueError):
                print(f"Warning: invalid sample_rate: {manifest_path}:{line_num}", file=sys.stderr)
                continue

            yield resolve_audio_path(manifest_path, audio_filepath), int(round(duration * sample_rate))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build one deduplicated pretrain.tsv from manifest.jsonl files")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--wav-limit", type=int, default=50)
    parser.add_argument("--default-sample-rate", type=int, default=None)
    parser.add_argument("--check-exists", action="store_true")
    parser.add_argument("--no-dedup", action="store_true")
    args = parser.parse_args()

    manifest_paths = []
    for path in args.paths:
        manifest_paths.extend(find_manifests(path.resolve(), args.wav_limit))
    manifest_paths = sorted(set(manifest_paths))

    if not manifest_paths:
        print("Error: no manifest.jsonl files found", file=sys.stderr)
        sys.exit(1)

    print(f"Found manifests: {len(manifest_paths)}", file=sys.stderr)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    seen_audio_paths = set()
    seen_num_samples = {}
    total_input_records = 0
    total_output_records = 0
    duplicate_records = 0
    skipped_missing_files = 0
    sample_count_conflicts = 0
    common_root = None

    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as tmp:
        tmp_path = Path(tmp.name)

        for manifest_path in manifest_paths:
            manifest_input_records = 0
            manifest_output_records = 0

            for audio_path, num_samples in iter_manifest_records(manifest_path, args.default_sample_rate):
                manifest_input_records += 1
                total_input_records += 1
                audio_path_str = str(audio_path)

                if args.check_exists and not audio_path.is_file():
                    skipped_missing_files += 1
                    continue

                if not args.no_dedup:
                    if audio_path_str in seen_audio_paths:
                        duplicate_records += 1
                        previous_num_samples = seen_num_samples.get(audio_path_str)
                        if previous_num_samples != num_samples:
                            sample_count_conflicts += 1
                            print(
                                "Warning: duplicate path with different num_samples: "
                                f"{audio_path_str}; first={previous_num_samples}, current={num_samples}",
                                file=sys.stderr,
                            )
                        continue
                    seen_audio_paths.add(audio_path_str)
                    seen_num_samples[audio_path_str] = num_samples

                common_root = audio_path_str if common_root is None else os.path.commonpath([common_root, audio_path_str])
                tmp.write(f"{audio_path_str}\t{num_samples}\n")
                total_output_records += 1
                manifest_output_records += 1

            print(
                f"Manifest: {manifest_path} | input={manifest_input_records} | added={manifest_output_records}",
                file=sys.stderr,
            )

    if total_output_records == 0:
        tmp_path.unlink(missing_ok=True)
        print("Error: no valid audio records found", file=sys.stderr)
        sys.exit(1)

    root = args.root.resolve() if args.root is not None else Path(common_root)
    if root.is_file():
        root = root.parent

    with args.output.open("w", encoding="utf-8") as out:
        out.write(str(root) + "\n")
        with tmp_path.open("r", encoding="utf-8") as tmp:
            for line in tmp:
                audio_path_str, num_samples = line.rstrip("\n").split("\t", 1)
                audio_path = Path(audio_path_str)
                try:
                    rel_audio_path = audio_path.relative_to(root)
                except ValueError:
                    print(f"Warning: audio file is outside root, writing absolute path: {audio_path}", file=sys.stderr)
                    rel_audio_path = audio_path
                out.write(f"{rel_audio_path}\t{num_samples}\n")

    tmp_path.unlink(missing_ok=True)

    print(f"Output: {args.output}", file=sys.stderr)
    print(f"Root: {root}", file=sys.stderr)
    print(f"Input records: {total_input_records}", file=sys.stderr)
    print(f"Output records: {total_output_records}", file=sys.stderr)
    print(f"Duplicate records skipped: {duplicate_records}", file=sys.stderr)
    if skipped_missing_files:
        print(f"Skipped missing audio files: {skipped_missing_files}", file=sys.stderr)
    if sample_count_conflicts:
        print(f"Duplicate sample count conflicts: {sample_count_conflicts}", file=sys.stderr)


if __name__ == "__main__":
    main()
