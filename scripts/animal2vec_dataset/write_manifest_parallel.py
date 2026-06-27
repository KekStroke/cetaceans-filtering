#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import soundfile as sf


def inspect_audio(audio_path: Path, manifest_dir: Path) -> tuple[dict | None, str | None]:
    try:
        info = sf.info(str(audio_path))
        duration = float(getattr(info, "duration", 0.0) or 0.0)
        sample_rate = int(getattr(info, "samplerate", 0) or 0)
        if duration <= 0 or sample_rate <= 0:
            return None, f"invalid metadata: {audio_path}"
        return (
            {
                "audio_filepath": audio_path.relative_to(manifest_dir).as_posix(),
                "duration": duration,
                "sample_rate": sample_rate,
            },
            None,
        )
    except Exception as exc:
        return None, f"{audio_path}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--progress-every", type=int, default=60)
    parser.add_argument("--error-log", required=True, type=Path)
    args = parser.parse_args()

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.error_log.parent.mkdir(parents=True, exist_ok=True)

    wavs = sorted(args.audio_dir.rglob("*.wav"))
    total = len(wavs)
    print(f"audio_dir={args.audio_dir}")
    print(f"manifest={args.manifest}")
    print(f"workers={args.workers}")
    print(f"wav_files={total}")

    done = 0
    ok = 0
    errors = 0
    started = time.monotonic()
    last_progress = started

    tmp_manifest = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    tmp_errors = args.error_log.with_suffix(args.error_log.suffix + ".tmp")

    with tmp_manifest.open("w", encoding="utf-8") as fout, tmp_errors.open(
        "w", encoding="utf-8"
    ) as ferr, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(inspect_audio, audio_path, args.manifest.parent): audio_path
            for audio_path in wavs
        }
        for fut in as_completed(futures):
            row, err = fut.result()
            done += 1
            if row is not None:
                fout.write(json.dumps(row, ensure_ascii=True) + "\n")
                ok += 1
            else:
                ferr.write(str(err) + "\n")
                errors += 1

            now = time.monotonic()
            if now - last_progress >= args.progress_every:
                elapsed = max(now - started, 1e-9)
                print(
                    f"progress done={done}/{total} ok={ok} errors={errors} "
                    f"rate={done / elapsed:.1f}/s",
                    flush=True,
                )
                last_progress = now

    tmp_manifest.replace(args.manifest)
    tmp_errors.replace(args.error_log)
    elapsed = max(time.monotonic() - started, 1e-9)
    print(
        f"finished done={done} ok={ok} errors={errors} "
        f"elapsed_s={elapsed:.1f} rate={done / elapsed:.1f}/s"
    )


if __name__ == "__main__":
    main()
