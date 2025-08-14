#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# LocalMedia2Text · Offline media-to-text CLI (faster-whisper + PyAV)
# -----------------------------------------------------------------------------
# Author: Pablo Beret
# License: MIT
#
# What it does
#   - Transcribes local AUDIO and VIDEO files to a plain TXT file.
#   - AUDIO direct: .mp3 .wav .m4a .aac .ogg .flac
#   - VIDEO: .mp4 .mkv .mov .avi .webm → extracts audio (WAV 16 kHz mono) and transcribes.
#
# How it works
#   - Uses faster-whisper for transcription (CTranslate2 backend).
#   - For video, uses PyAV to demux/decode/resample to PCM s16le 16 kHz mono.
#   - VAD (Silero) enabled by default, beam_size=5, language autodetection unless fixed.
#   - Auto device selection: CUDA if a GPU is available; if it fails, falls back to CPU.
#
# Install (Python 3.9+):
#   pip install faster-whisper av
#
# Quick usage
#   # If there is a single supported file in the script folder:
#   python local_media2text.py
#
#   # Specifying a file:
#   python local_media2text.py --file "input.mp3"
#   python local_media2text.py --file "video.mp4" --lang es
#
#   # Forcing CPU or GPU and compute type:
#   python local_media2text.py --file "meeting.mkv" --device cpu --compute_type int8
#   python local_media2text.py --file "interview.mkv" --device cuda --compute_type float16
#
# GPU notes (optional):
#   - To use GPU on Windows/Linux: have CUDA 12 + cuDNN 9 available on the system.
#   - If anything fails with CUDA/cuDNN, the script automatically falls back to CPU (INT8).
# =============================================================================

"""
Transcribe an MP3/MP4/MKV (same folder) to TXT, offline, with faster-whisper.
- Direct audio (mp3/wav/m4a/aac/ogg/flac).
- Video (mp4/mkv/mov/avi/webm): extracts audio to WAV 16 kHz mono and transcribes.
- If GPU fails due to CUDA/cuDNN, it automatically falls back to CPU.

Requirements:
    pip install faster-whisper av
Notes:
    - You don't need system FFmpeg: audio/video is decoded with PyAV (FFmpeg bundled in the wheel).
"""

from __future__ import annotations
import argparse
import sys
import tempfile
from pathlib import Path
from typing import Optional, Iterable, Set, Tuple, List

from faster_whisper import WhisperModel

# GPU detection via CTranslate2 (backend used by faster-whisper)
try:
    import ctranslate2  # type: ignore
    _HAS_CT2 = True
except Exception:
    _HAS_CT2 = False

AUDIO_EXTS: Set[str] = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}
VIDEO_EXTS: Set[str] = {".mp4", ".mkv", ".mov", ".avi", ".webm"}
SUPPORTED_EXTS: Set[str] = AUDIO_EXTS | VIDEO_EXTS


def pick_device_and_compute_type(force_device: Optional[str],
                                 force_compute: Optional[str]) -> Tuple[str, str]:
    """Pick (device, compute_type) with simple heuristics without changing transcription logic."""
    if force_device:
        device = force_device
    else:
        if _HAS_CT2:
            try:
                has_cuda = ctranslate2.get_cuda_device_count() > 0
            except Exception:
                has_cuda = False
            device = "cuda" if has_cuda else "cpu"
        else:
            device = "cpu"

    if force_compute:
        ctype = force_compute
    else:
        ctype = "float16" if device == "cuda" else "int8"
    return device, ctype


def find_single_media(search_dir: Path) -> Path:
    """Return the single supported file in the folder, or raise if 0 or >1 (same behavior)."""
    files: List[Path] = [p for p in sorted(search_dir.iterdir())
                         if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    if len(files) == 0:
        raise FileNotFoundError(
            f"No supported files found in {search_dir} "
            f"(extensions: {', '.join(sorted(SUPPORTED_EXTS))}). "
            f"Pass one with --file path.ext"
        )
    if len(files) > 1:
        names = "\n  - " + "\n  - ".join(p.name for p in files)
        raise RuntimeError(
            "Multiple supported files found in the folder. "
            "Specify which one to use with --file.\n" + names
        )
    return files[0]


def extract_audio_to_wav(input_path: Path, out_wav: Path,
                         target_rate: int = 16000, layout: str = "mono") -> None:
    """
    Extract audio from the container to WAV PCM s16le (mono, 16 kHz) using PyAV.
    IMPORTANT (PyAV >= 9): AudioResampler.resample() can return a LIST of frames.
    """
    import av  # PyAV
    from av.audio.resampler import AudioResampler

    in_container = av.open(str(input_path))
    try:
        audio_streams = [s for s in in_container.streams if s.type == "audio"]
        if not audio_streams:
            raise RuntimeError("The file contains no audio stream.")
        stream = audio_streams[0]

        out_container = av.open(str(out_wav), mode="w", format="wav")
        try:
            out_stream = out_container.add_stream("pcm_s16le", rate=target_rate)
            # Ensure mono
            out_stream.layout = layout

            # Resample to s16/mono/16k
            resampler = AudioResampler(format="s16", layout=layout, rate=target_rate)

            # Decode audio -> resample (may return list of frames) -> encode
            for frame in in_container.decode(stream):
                if not isinstance(frame, av.AudioFrame):
                    continue
                frame.pts = None
                res = resampler.resample(frame)

                # Normalize to a list of frames
                if res is None:
                    continue
                if isinstance(res, list):
                    out_frames = [f for f in res if isinstance(f, av.AudioFrame)]
                else:
                    out_frames = [res]

                for of in out_frames:
                    for pkt in out_stream.encode(of):
                        out_container.mux(pkt)

            # Flush encoder
            for pkt in out_stream.encode(None):
                out_container.mux(pkt)
        finally:
            out_container.close()
    finally:
        in_container.close()


def do_transcribe(model: WhisperModel, media_path: Path, out_txt: Path,
                  language: Optional[str]) -> int:
    """Run transcription and write to TXT (one line per segment)."""
    print(f"[i] Transcribing: {media_path.name}")
    segments, info = model.transcribe(
        str(media_path),
        language=language,          # None → autodetect
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        beam_size=5
    )
    n_lines = 0
    with out_txt.open("w", encoding="utf-8") as f:
        for seg in segments:
            line = seg.text.strip()
            if line:
                f.write(line + "\n")
                n_lines += 1

    print(f"[✓] Detected language: {info.language} (p={info.language_probability:.2f})")
    print(f"[✓] Lines written: {n_lines}")
    print(f"[✓] TXT saved to: {out_txt.resolve()}")
    return n_lines


def transcribe_to_txt(
    input_path: Path,
    out_txt: Optional[Path] = None,
    model_size: str = "small",
    language: Optional[str] = None,
    device: Optional[str] = None,
    compute_type: Optional[str] = None,
) -> Path:
    """
    - Audio (mp3/wav/...): pass directly to faster-whisper (PyAV decodes internally).
    - Video (mp4/mkv/...): extract WAV 16 kHz mono and transcribe that WAV.
    - Auto fallback to CPU if GPU fails (CUDA/cuDNN).
    """
    device, compute_type = pick_device_and_compute_type(device, compute_type)
    ext = input_path.suffix.lower()

    if out_txt is None:
        out_txt = input_path.with_suffix(".txt")

    print(f"[i] Loading model: {model_size} | device={device} | compute_type={compute_type}")
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
    except Exception as e:
        msg = str(e).lower()
        needs_fallback = ("cudnn" in msg) or ("cuda" in msg) or ("cublas" in msg) or ("invalid handle" in msg)
        if not needs_fallback or device == "cpu":
            raise
        print("[!] GPU initialization failed. Retrying on CPU (INT8)…")
        model = WhisperModel(model_size, device="cpu", compute_type="int8")

    if ext in AUDIO_EXTS:
        # faster-whisper decodes audio via PyAV; no system FFmpeg required.
        do_transcribe(model, input_path, out_txt, language)
        return out_txt

    elif ext in VIDEO_EXTS:
        # Extract audio to a temporary WAV (16 kHz mono PCM) and transcribe
        with tempfile.TemporaryDirectory() as td:
            tmp_wav = Path(td) / (input_path.stem + "__tmp.wav")
            print("[i] Extracting audio from video to WAV (16 kHz, mono)…")
            extract_audio_to_wav(input_path, tmp_wav, target_rate=16000, layout="mono")
            do_transcribe(model, tmp_wav, out_txt, language)
        return out_txt

    else:
        raise ValueError(f"Unsupported extension: {ext}. Supported: {', '.join(sorted(SUPPORTED_EXTS))}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Transcribe an MP3/MP4/MKV (offline) with faster-whisper. "
                    "If it is a video, extract audio to WAV and transcribe."
    )
    p.add_argument("--file", type=str, default=None,
                   help="Path to the file to transcribe (default: the single supported file in the folder).")
    p.add_argument("--model", type=str, default="small",
                   help='Model: tiny/base/small/medium/large-v3/distil-large-v3/turbo… (default: "small")')
    p.add_argument("--lang", type=str, default=None, help='Fixed language (e.g., "es"). If omitted, autodetect.')
    p.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"],
                   help='Force "cpu" or "cuda". If omitted, auto-detect.')
    p.add_argument("--compute_type", type=str, default=None,
                   choices=["int8", "int8_float16", "int16", "float16", "float32"],
                   help='Numeric precision/quantization. Default: FP16 on GPU, INT8 on CPU.')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    if args.file:
        media_path = Path(args.file).expanduser().resolve()
    else:
        media_path = find_single_media(script_dir)

    if not media_path.exists():
        print(f"[x] File does not exist: {media_path}", file=sys.stderr)
        sys.exit(1)

    try:
        transcribe_to_txt(
            input_path=media_path,
            out_txt=None,
            model_size=args.model,
            language=args.lang,
            device=args.device,
            compute_type=args.compute_type,
        )
    except Exception as e:
        print(f"[x] Error during transcription: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
