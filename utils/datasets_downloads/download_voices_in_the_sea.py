import shutil
import urllib.parse
from pathlib import Path
from typing import Iterable, List, Set

import hydra
import librosa
import requests
import soundfile as sf
from audio_saver import process_array_audio, process_large_audio, sanitize_stem
from bs4 import BeautifulSoup
from manifest_utils import write_manifest
from omegaconf import DictConfig
from tqdm import tqdm

BASE_URL_DEFAULT = (
    "https://voicesinthesea.ucsd.edu/species/spectrogramPlayerComponents/audio/"
)

AUDIO_WAVLIKE = (".wav", ".flac", ".WAV", ".FLAC")
AUDIO_COMPRESSED = (".mp3", ".ogg", ".MP3", ".OGG")
AUDIO_ALL = AUDIO_WAVLIKE + AUDIO_COMPRESSED


def _join_url(base: str, href: str) -> str:
    # Use urllib.parse.urljoin for robust URL joining
    return urllib.parse.urljoin(base, href)


def _is_under_base_url(url: str, base_url: str) -> bool:
    parsed_url = urllib.parse.urlparse(url)
    parsed_base = urllib.parse.urlparse(base_url)
    return (
        parsed_url.scheme in {"http", "https"}
        and parsed_url.netloc == parsed_base.netloc
        and parsed_url.path.startswith(parsed_base.path)
    )


def crawl_listing(
    start_url: str, include_dirs: List[str] | None = None
) -> Iterable[str]:
    """
    Recursively yield file URLs from an Apache-style index.
    If include_dirs is provided, only traverse subpaths that start with any of those prefixes.
    """
    seen: Set[str] = set()
    stack: List[str] = [start_url if start_url.endswith("/") else start_url + "/"]
    include_dirs = [d.strip("/") for d in (include_dirs or [])]

    while stack:
        url = stack.pop()
        if url in seen:
            continue
        seen.add(url)

        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
        except Exception as e:
            print(f"skip list {url}: {e}")
            continue

        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            # skip parent, query anchors, fragments, mailto, empty, dot paths
            if (
                href in {"../", "./"}
                or href.startswith("?")
                or href.startswith("#")
                or href.startswith("mailto:")
                or not href
                or href in {".", ".."}
            ):
                continue
            child = _join_url(url, href)
            if not _is_under_base_url(child, start_url):
                continue

            if href.endswith("/"):
                # dir filtering (optional)
                if include_dirs:
                    rel = child.replace(start_url, "")
                    if not any(
                        rel.startswith(prefix + "/") or rel == prefix
                        for prefix in include_dirs
                    ):
                        continue
                stack.append(child)
            else:
                if any(href.lower().endswith(ext.lower()) for ext in AUDIO_ALL):
                    yield child


def stream_download(url: str, dst: Path, chunk_mb: int = 4) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_mb * 1024 * 1024):
                if chunk:
                    f.write(chunk)
    return dst


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    dl = cfg["data_loading"]
    out_root = Path(dl["raw_datasets_path"])
    out_dir = out_root / "voices_in_the_sea"
    audio_dir = out_dir / "audio"
    manifest_path = out_dir / "manifest.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    base_url = BASE_URL_DEFAULT.rstrip("/") + "/"
    include_dirs = []

    sr_target = dl["raw_sample_rate"]
    chunk_sec = float(dl["raw_segment_duration"])

    tmp_dir = out_root / "_tmp" / "voices_in_the_sea"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    total_seconds = [0.0]
    processed = 0

    try:
        file_urls = crawl_listing(base_url, include_dirs=include_dirs)

        for url in tqdm(
            file_urls,
            desc="Downloading & processing Voices in the Sea",
            unit="file",
        ):
            # local mirror path under tmp
            rel = url.replace(base_url, "")
            local_tmp = tmp_dir / rel
            try:
                stream_download(url, local_tmp)
            except Exception as e:
                print(f"skip download {url}: {e}")
                continue

            # output stem is normalized relative path
            stem = sanitize_stem(
                f"voices_{Path(rel).with_suffix('').as_posix().replace('/', '_')}"
            )

            try:
                # WAV/FLAC -> memory-safe processing
                if local_tmp.suffix.lower() in [ext.lower() for ext in AUDIO_WAVLIKE]:
                    process_large_audio(
                        src_path=local_tmp,
                        out_dir=audio_dir,
                        stem_base=stem,
                        total_seconds_ref=total_seconds,
                        sr_target=sr_target,
                        chunk_sec=chunk_sec,
                    )
                else:
                    # MP3/OGG: decode to array (libsndfile often handles OGG; MP3 -> librosa)
                    try:
                        # First try soundfile (works for many OGGs)
                        data, sr = sf.read(local_tmp, always_2d=False)
                    except Exception:
                        # Fallback for MP3/anything else via librosa/audioread
                        data, sr = librosa.load(local_tmp, sr=None, mono=True)
                    process_array_audio(
                        data=data,
                        sr=int(sr),
                        out_dir=audio_dir,
                        stem_base=stem,
                        total_seconds_ref=total_seconds,
                        sr_target=sr_target,
                        chunk_sec=chunk_sec,
                    )

                processed += 1
            except Exception as e:
                print(f"error processing {rel}: {e}")

            # remove tmp file after processing
            try:
                local_tmp.unlink(missing_ok=True)
            except Exception:
                pass

        manifest_entries = write_manifest(
            audio_dir=audio_dir,
            manifest_path=manifest_path,
        )
        print(f"Manifest entries: {manifest_entries} ({manifest_path.resolve()})")

    finally:
        # cleanup
        try:
            shutil.rmtree(tmp_dir)
        except Exception:
            pass

    print("\nFinished")
    print(f"Processed source items: {processed}")
    print(f"Total duration (output WAVs): {total_seconds[0] / 3600:.2f} h")
    print(f"Audio dir: {audio_dir.resolve()}")


if __name__ == "__main__":
    main()
