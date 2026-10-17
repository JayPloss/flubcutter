"""Step 1: verbatim speech-to-text with word-level timestamps.

Uses NVIDIA Parakeet TDT 0.6B (via onnx-asr). Unlike Whisper, a transducer
model emits what it hears frame by frame instead of "writing" the text like a
language model, so it does not smooth away repeated phrases and false
starts, which are exactly what flub detection needs.
"""

import json
import math
import subprocess
from functools import cache
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
MODEL = "nemo-parakeet-tdt-0.6b-v2"


def load_audio(path: Path) -> np.ndarray:
    """Decode any audio file to 16 kHz mono float32 via ffmpeg."""
    raw = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
         "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32)


@cache
def _model():
    import onnx_asr

    return onnx_asr.load_model(MODEL, quantization="int8")


@cache
def _recognizer():
    """For whole recordings: voice activity detection splits them into segments."""
    import onnx_asr

    return _model().with_vad(onnx_asr.load_vad("silero")).with_timestamps()


def _words(segment) -> list[dict]:
    """Group a segment's subword tokens into words with absolute times."""
    words = []
    tokens, times, logprobs = segment.tokens, segment.timestamps, segment.logprobs
    for k, tok in enumerate(tokens):
        t = segment.start + times[k]
        if not words or tok.startswith(" "):
            words.append({"word": tok.strip(), "start": t, "logprobs": [logprobs[k]]})
        else:
            words[-1]["word"] += tok
            words[-1]["logprobs"].append(logprobs[k])
    # A word ends where the next one starts, or at the segment's end.
    for k, w in enumerate(words):
        w["end"] = words[k + 1]["start"] if k + 1 < len(words) else segment.end
    out = []
    for w in words:
        if not any(c.isalnum() for c in w["word"]):
            continue  # stray punctuation token
        lp = w.pop("logprobs")
        out.append({
            "word": w["word"],
            "start": round(w["start"], 3),
            "end": round(max(w["end"], w["start"] + 0.05), 3),
            "prob": round(math.exp(sum(lp) / len(lp)), 3),
        })
    return out


def transcribe(audio: Path, progress=None) -> dict:
    samples = load_audio(audio)
    duration = len(samples) / SAMPLE_RATE
    words = []
    for seg in _recognizer().recognize(samples, sample_rate=SAMPLE_RATE):
        words.extend(_words(seg))
        if progress:
            progress(min(seg.end / duration, 1.0))
    return {
        "audio": str(audio),
        "model": MODEL,
        "duration": round(duration, 3),
        "words": words,
    }


def words_in(samples: np.ndarray) -> list[str]:
    """Words heard in a short 16 kHz clip (a few seconds; no segmentation,
    which can drop speech that starts right at the edge of a clip)."""
    return _model().recognize(samples, sample_rate=SAMPLE_RATE).split()


def save(result: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
