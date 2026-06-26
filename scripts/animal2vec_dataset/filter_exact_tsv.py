#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frames", required=True, type=int)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0
    skipped = 0
    bad = 0

    with args.input.open("r", encoding="utf-8") as fin, args.output.open(
        "w", encoding="utf-8"
    ) as fout:
        first_line = fin.readline()
        if not first_line:
            raise RuntimeError(f"empty TSV: {args.input}")
        fout.write(first_line)

        for line in fin:
            stripped = line.strip()
            if not stripped:
                continue
            total += 1
            try:
                rel_path, frames_s = stripped.rsplit(None, 1)
                frames = int(frames_s)
            except Exception:
                bad += 1
                continue
            if frames == args.frames:
                fout.write(f"{rel_path}\t{frames}\n")
                kept += 1
            else:
                skipped += 1

    args.report.write_text(
        "\n".join(
            [
                f"input={args.input}",
                f"output={args.output}",
                f"root={first_line.strip()}",
                f"target_frames={args.frames}",
                f"total_rows={total}",
                f"kept_rows={kept}",
                f"skipped_non_exact_rows={skipped}",
                f"bad_rows={bad}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
