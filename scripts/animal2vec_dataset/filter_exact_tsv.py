#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import soundfile as sf


def audio_reject_reason(
    audio_path: Path,
    expected_frames: int,
    std_min: float,
    rms_min: float,
    peak_min: float,
) -> tuple[str | None, tuple[int, float, float, float]]:
    audio, _sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
    data = np.asarray(audio, dtype=np.float32).reshape(-1)
    sample_count = int(data.size)
    if sample_count != expected_frames:
        return "wrong_frame_count", (sample_count, math.nan, math.nan, math.nan)
    if not bool(np.isfinite(data).all()):
        return "non_finite", (sample_count, math.nan, math.nan, math.nan)

    std = float(np.std(data, dtype=np.float64))
    rms = float(np.sqrt(np.mean(np.square(data, dtype=np.float64))))
    peak = float(np.max(np.abs(data))) if sample_count else 0.0
    if std < std_min:
        return "std_lt", (sample_count, std, rms, peak)
    if rms < rms_min:
        return "rms_lt", (sample_count, std, rms, peak)
    if peak < peak_min:
        return "peak_lt", (sample_count, std, rms, peak)
    return None, (sample_count, std, rms, peak)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frames", required=True, type=int)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--check-audio-quality", action="store_true")
    parser.add_argument("--rejects", type=Path, default=None)
    parser.add_argument("--std-min", type=float, default=1e-6)
    parser.add_argument("--rms-min", type=float, default=1e-5)
    parser.add_argument("--peak-min", type=float, default=1e-4)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0
    skipped_non_exact = 0
    skipped_quality = 0
    bad = 0

    reject_file = None
    if args.check_audio_quality:
        if args.rejects is None:
            raise ValueError("--rejects is required with --check-audio-quality")
        args.rejects.parent.mkdir(parents=True, exist_ok=True)
        reject_file = args.rejects.open("w", encoding="utf-8")
        reject_file.write("rel_path\tframes\treason\tsample_count\tstd\trms\tpeak\n")

    with args.input.open("r", encoding="utf-8") as fin, args.output.open(
        "w", encoding="utf-8"
    ) as fout:
        first_line = fin.readline()
        if not first_line:
            raise RuntimeError(f"empty TSV: {args.input}")
        root = Path(first_line.strip())
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
                if args.check_audio_quality:
                    try:
                        reason, stats = audio_reject_reason(
                            root / rel_path,
                            expected_frames=args.frames,
                            std_min=args.std_min,
                            rms_min=args.rms_min,
                            peak_min=args.peak_min,
                        )
                    except Exception as exc:
                        reason = f"read_error:{type(exc).__name__}"
                        stats = (0, math.nan, math.nan, math.nan)
                    if reason is not None:
                        sample_count, std, rms, peak = stats
                        reject_file.write(
                            f"{rel_path}\t{frames}\t{reason}\t{sample_count}\t{std:.9g}\t{rms:.9g}\t{peak:.9g}\n"
                        )
                        skipped_quality += 1
                        continue
                fout.write(f"{rel_path}\t{frames}\n")
                kept += 1
            else:
                skipped_non_exact += 1

    if reject_file is not None:
        reject_file.close()

    args.report.write_text(
        "\n".join(
            [
                f"input={args.input}",
                f"output={args.output}",
                f"root={first_line.strip()}",
                f"target_frames={args.frames}",
                f"total_rows={total}",
                f"kept_rows={kept}",
                f"skipped_non_exact_rows={skipped_non_exact}",
                f"skipped_quality_rows={skipped_quality}",
                f"bad_rows={bad}",
                f"check_audio_quality={args.check_audio_quality}",
                f"std_min={args.std_min}",
                f"rms_min={args.rms_min}",
                f"peak_min={args.peak_min}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
