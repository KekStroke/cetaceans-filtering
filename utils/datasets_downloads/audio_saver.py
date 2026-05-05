import re
from pathlib import Path
from typing import Any, List, Optional

import librosa
import numpy as np
import soundfile as sf


def sanitize_stem(text: str) -> str:
    """Безопасное имя для файлов."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def to_mono(x: np.ndarray) -> np.ndarray:
    """Среднее по каналам, если стерео/многоканал."""
    if x.ndim > 1:
        return x.mean(axis=1)
    return x


def write_wav(out_path: Path, data: np.ndarray, sr: int) -> None:
    """Гарантирует существование папки и записывает WAV (float32 -> PCM при необходимости)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, data, sr)


def resolve_min_sample_rate(
    raw_sample_rate: Any,
    raw_skip_below_sample_rate: bool,
) -> Optional[int]:
    if not raw_skip_below_sample_rate:
        return None
    if raw_sample_rate is None or str(raw_sample_rate).strip().lower() in {
        "none",
        "null",
        "",
    }:
        raise ValueError(
            "data_loading.raw_sample_rate must be set when "
            "data_loading.raw_skip_below_sample_rate=true"
        )
    return int(raw_sample_rate)


def _should_skip_sample_rate(sr: int, min_sample_rate: Optional[int], label: str) -> bool:
    if min_sample_rate is None or sr >= min_sample_rate:
        return False
    print(f"Skip '{label}': sample rate {sr} Hz < minimum {min_sample_rate} Hz")
    return True


def process_array_audio(
    data: np.ndarray,
    sr: int,
    out_dir: Path,
    stem_base: str,
    total_seconds_ref: Optional[List[float]],
    sr_target: int,
    chunk_sec: float,
    min_sample_rate: Optional[int] = None,
) -> None:
    """
    Обработка уже загруженного массива:
      - моно
      - ресемпл при необходимости
      - запись цельного файла или нарезка по chunk_sec
    """
    if _should_skip_sample_rate(int(sr), min_sample_rate, stem_base):
        return

    data = to_mono(data).astype("float32", copy=False)
    # if sr != sr_target:
    #     data = librosa.resample(data, orig_sr=sr, target_sr=sr_target)
    #     sr = sr_target

    duration = len(data) / sr
    if chunk_sec == -1 or duration <= chunk_sec:
        out_path = out_dir / f"{stem_base}.wav"
        write_wav(out_path, data, sr)
        if total_seconds_ref is not None:
            total_seconds_ref[0] += duration
        return

    samples_per_chunk = int(round(chunk_sec * sr))
    n = len(data)
    n_chunks = (n + samples_per_chunk - 1) // samples_per_chunk
    for i in range(n_chunks):
        st = i * samples_per_chunk
        en = min((i + 1) * samples_per_chunk, n)
        chunk = data[st:en]
        out_path = out_dir / f"{stem_base}_{i:05d}.wav"
        write_wav(out_path, chunk, sr)
        if total_seconds_ref is not None:
            total_seconds_ref[0] += len(chunk) / sr


def process_large_audio(
    src_path: Path,
    out_dir: Path,
    stem_base: str,
    total_seconds_ref: Optional[List[float]],
    sr_target: int,
    chunk_sec: float,
    stream_block_sec: float = 30.0,
    min_sample_rate: Optional[int] = None,
) -> None:
    """
    Памяти-бережливая обработка файла на диске:
      - читает блоками через soundfile,
      - приводит к моно,
      - если SR отличается — ресемплит блоки на лету,
      - сохраняет кусками длиной chunk_sec.
    """
    with sf.SoundFile(str(src_path), mode="r") as f:
        orig_sr = f.samplerate
        n_frames = f.frames
        if _should_skip_sample_rate(int(orig_sr), min_sample_rate, src_path.name):
            return

        # Если файл короче chunk_sec — сохраняем единый WAV (в RAM, разово).
        if chunk_sec == -1 or n_frames / orig_sr <= chunk_sec:
            data = f.read(frames=n_frames, dtype="float32", always_2d=False)
            data = to_mono(data)
            # if orig_sr != sr_target:
            #     data = librosa.resample(data, orig_sr=orig_sr, target_sr=sr_target)
            #     out_sr = sr_target
            # else:
            #     out_sr = orig_sr
            out_sr = orig_sr
            out_path = out_dir / f"{stem_base}.wav"
            write_wav(out_path, data, out_sr)
            if total_seconds_ref is not None:
                total_seconds_ref[0] += len(data) / out_sr
            return

        out_idx = 0
        if orig_sr == sr_target:
            # Нарезка без ресемпла
            frames_per_chunk = int(round(chunk_sec * orig_sr))
            f.seek(0)
            while True:
                x = f.read(frames=frames_per_chunk, dtype="float32", always_2d=False)
                if x is None or len(x) == 0:
                    break
                x = to_mono(x)
                out_path = out_dir / f"{stem_base}_{out_idx:05d}.wav"
                write_wav(out_path, x, orig_sr)
                if total_seconds_ref is not None:
                    total_seconds_ref[0] += len(x) / orig_sr
                out_idx += 1
        else:
            # Ресемпл на лету в буфер с фиксированным размером чанка
            frames_per_block = int(round(stream_block_sec * orig_sr))
            target_chunk_samples = int(round(chunk_sec * sr_target))
            buf = np.zeros(0, dtype=np.float32)
            f.seek(0)
            while True:
                x = f.read(frames=frames_per_block, dtype="float32", always_2d=False)
                if x is None or len(x) == 0:
                    break
                x = to_mono(x)
                x = librosa.resample(x, orig_sr=orig_sr, target_sr=sr_target)
                buf = x if buf.size == 0 else np.concatenate([buf, x])

                while buf.size >= target_chunk_samples:
                    chunk = buf[:target_chunk_samples]
                    buf = buf[target_chunk_samples:]
                    out_path = out_dir / f"{stem_base}_{out_idx:05d}.wav"
                    write_wav(out_path, chunk, sr_target)
                    if total_seconds_ref is not None:
                        total_seconds_ref[0] += len(chunk) / sr_target
                    out_idx += 1

            # Хвост запишем как последний неполный чанк
            if buf.size > 0:
                out_path = out_dir / f"{stem_base}_{out_idx:05d}.wav"
                write_wav(out_path, buf, sr_target)
                if total_seconds_ref is not None:
                    total_seconds_ref[0] += len(buf) / sr_target
