"""Bounded audio decoding and transcript export."""
from html import escape
import subprocess
from tempfile import NamedTemporaryFile

import numpy as np

SAMPLE_RATE = 16_000
MAX_MINUTES = 10
MAX_BYTES = 50 * 1024 * 1024


class AudioError(ValueError):
    """An audio error suitable for display."""


def decode_audio(data: bytes) -> np.ndarray:
    if not data or len(data) > MAX_BYTES:
        raise AudioError("Choose a non-empty file under 50 MB.")
    try:
        with NamedTemporaryFile() as source:
            source.write(data)
            source.flush()
            decoded = subprocess.run(
                [
                    "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                    "-protocol_whitelist", "file,pipe",
                    "-format_whitelist", "wav,mp3,mov,mp4,m4a,3gp,3g2,mj2,aac,flac,ogg,matroska,webm,mpeg,asf",
                    "-threads", "1", "-i", source.name,
                    "-map", "0:a:0", "-t", str(MAX_MINUTES * 60 + 1),
                    "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "pipe:1",
                ],
                capture_output=True, timeout=60, check=True,
            )
    except FileNotFoundError as exc:
        raise AudioError("FFmpeg is missing. Install it and restart the app.") from exc
    except subprocess.TimeoutExpired as exc:
        raise AudioError("Audio decoding timed out. Try a shorter recording.") from exc
    except subprocess.CalledProcessError as exc:
        raise AudioError("This file could not be decoded. Try exporting it as WAV or MP3.") from exc
    samples = np.frombuffer(decoded.stdout, dtype="<i2")
    if len(samples) > MAX_MINUTES * 60 * SAMPLE_RATE:
        raise AudioError(f"Choose a recording up to {MAX_MINUTES} minutes.")
    if not samples.size or not samples.any():
        raise AudioError("No audio signal was found in this recording.")
    return samples.astype(np.float32) / 32768


def timestamp(seconds: float, separator: str = ".") -> str:
    minutes, milliseconds = divmod(max(0, round(seconds * 1000)), 60_000)
    hours, minutes = divmod(minutes, 60)
    seconds, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{milliseconds:03d}"


def subtitles(segments: list[dict], *, vtt: bool) -> str:
    separator = "." if vtt else ","
    blocks = ["WEBVTT"] if vtt else []
    for index, segment in enumerate(segments, 1):
        start = timestamp(segment["start"], separator)
        end = timestamp(segment["end"], separator)
        caption = segment["text"].strip()
        if vtt:
            caption = escape(caption, quote=False)
        cue = f"{start} --> {end}\n{caption}"
        blocks.append(cue if vtt else f"{index}\n{cue}")
    return "\n\n".join(blocks) + "\n"
