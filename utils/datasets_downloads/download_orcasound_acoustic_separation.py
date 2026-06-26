import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import boto3
from botocore import UNSIGNED
from botocore.client import Config


BUCKET = "acoustic-sandbox"
PREFIX = "acoustic-separation/dataset/"
DEFAULT_OUTPUT_DIR = Path("data") / "orcasound" / "acoustic-separation"

SOURCE_TO_LOCAL_NAME = {
    "mixed.wav": "mixed.wav",
    "noise.wav": "noise.wav",
    "orca.wav": "sound.wav",
}


def _public_s3_client():
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def _list_audio_examples(
    s3, splits: Iterable[str], limit: Optional[int]
) -> Dict[str, Dict[str, Tuple[str, int]]]:
    examples: Dict[str, Dict[str, Tuple[str, int]]] = defaultdict(dict)
    paginator = s3.get_paginator("list_objects_v2")

    for split in splits:
        split_prefix = f"{PREFIX}{split.strip('/')}/"
        print(f"Listing s3://{BUCKET}/{split_prefix}")
        for page in paginator.paginate(Bucket=BUCKET, Prefix=split_prefix):
            for obj in page.get("Contents", []):
                key = str(obj.get("Key", ""))
                if not key or key.endswith("/"):
                    continue

                filename = Path(key).name
                if filename not in SOURCE_TO_LOCAL_NAME:
                    continue

                example_dir = str(Path(key).parent.as_posix()).removeprefix(PREFIX)
                examples[example_dir][filename] = (key, int(obj.get("Size", 0)))

                complete = {
                    example_dir: files
                    for example_dir, files in examples.items()
                    if all(name in files for name in SOURCE_TO_LOCAL_NAME)
                }
                if limit is not None and len(complete) >= limit:
                    return dict(list(sorted(complete.items()))[:limit])

    complete = {
        example_dir: files
        for example_dir, files in examples.items()
        if all(name in files for name in SOURCE_TO_LOCAL_NAME)
    }
    return dict(sorted(complete.items()))


def _download_file(s3, key: str, size: int, destination: Path, force: bool) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if (
        not force
        and destination.exists()
        and destination.stat().st_size == size
    ):
        return False

    s3.download_file(BUCKET, key, str(destination))
    return True


def download_dataset(
    output_dir: Path,
    splits: Iterable[str],
    limit: Optional[int],
    force: bool,
    include_csv: bool,
    flat: bool,
) -> None:
    s3 = _public_s3_client()
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = _list_audio_examples(s3, splits=splits, limit=limit)

    if not examples:
        raise RuntimeError("No complete acoustic-separation examples found.")

    if flat and len(examples) != 1:
        raise ValueError("--flat can only be used when exactly one example is selected")

    if include_csv and not flat:
        for csv_name in ("orca_train.csv", "orca_validation.csv"):
            key = f"{PREFIX}{csv_name}"
            destination = output_dir / csv_name
            print(f"Downloading metadata: {csv_name}")
            s3.download_file(BUCKET, key, str(destination))

    print(f"Examples to download: {len(examples)}")
    downloaded = 0
    reused = 0

    for index, (example_dir, files) in enumerate(examples.items(), start=1):
        local_example_dir = output_dir if flat else output_dir / example_dir
        print(f"[{index}/{len(examples)}] {example_dir}")

        for source_name, local_name in SOURCE_TO_LOCAL_NAME.items():
            key, size = files[source_name]
            destination = local_example_dir / local_name
            if _download_file(s3, key, size, destination, force):
                downloaded += 1
                print(f"  downloaded {local_name}")
            else:
                reused += 1
                print(f"  exists     {local_name}")

    print("\nFinished")
    print(f"Output dir: {output_dir.resolve()}")
    print(f"Downloaded files: {downloaded}")
    print(f"Already present: {reused}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the public Orcasound acoustic-separation dataset and rename "
            "each example to mixed.wav, noise.wav, and sound.wav."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Destination root directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["Train", "Validation"],
        help="Dataset splits to download. Default: Train Validation",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Download only the first N complete examples, useful for a quick test.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even when a same-size local file already exists.",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Do not download orca_train.csv and orca_validation.csv.",
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help=(
            "Put one selected example directly in output-dir as mixed.wav, "
            "noise.wav, and sound.wav. Use with --limit 1."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be a positive integer")

    download_dataset(
        output_dir=args.output_dir,
        splits=args.splits,
        limit=args.limit,
        force=bool(args.force),
        include_csv=not bool(args.no_csv),
        flat=bool(args.flat),
    )


if __name__ == "__main__":
    main()
