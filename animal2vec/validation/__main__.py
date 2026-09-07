"""Command-line entry point for canonical animal2vec validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


USAGE = """usage: python -m animal2vec.validation COMMAND [OPTIONS]

commands:
  watkins    canonical Watkins species linear probe
  check-manifest    validate split structure and optional artifact hashes
"""


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        print(USAGE)
        return
    command = arguments.pop(0)
    if command == "check-manifest":
        parser = argparse.ArgumentParser(
            prog="python -m animal2vec.validation check-manifest"
        )
        parser.add_argument("manifest", type=Path)
        parser.add_argument(
            "--data-dir",
            type=Path,
            help="also verify every pinned dataset-artifact SHA256",
        )
        parsed = parser.parse_args(arguments)
        from .protocol import load_split_manifest

        manifest = load_split_manifest(parsed.manifest)
        if parsed.data_dir is not None:
            manifest.verify_artifacts(parsed.data_dir)
        counts = ", ".join(
            f"{role}={len(values)}" for role, values in manifest.splits.items()
        )
        print(f"{manifest.protocol_id}: {counts}")
        return
    if command == "watkins":
        from .watkins import main as watkins_main

        watkins_main(arguments)
        return
    raise SystemExit(f"unknown command {command!r}\n\n{USAGE}")


if __name__ == "__main__":
    main()
