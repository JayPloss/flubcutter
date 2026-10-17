"""Step 1: verbatim speech-to-text with word-level timestamps."""

import json
import subprocess
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000  # what Whisper expects

# Whisper tends to tidy up speech (dropping "um", repeats, false starts).
# Priming it with a disfluent prompt nudges it toward a verbatim transcript,
# which is exactly what flub detection needs.
VERBATIM_PROMPT = (
    "Umm, so, uh, she was, she was sitting by the... by her sister. "
    "Hmm. Wait, no, let me, let me read that again."
)


def load_audio(path: Path) -> np.ndarray:
    """Decode any audio file to 16 kHz mono float32 via ffmpeg."""
    raw = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
         "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32)


def transcribe(audio: Path, model_name: str = "large-v3-turbo") -> dict:
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, info = model.transcribe(
        load_audio(audio),
        language="en",
        word_timestamps=True,
        initial_prompt=VERBATIM_PROMPT,
        # Don't let earlier text bias later text; reduces "smoothing" of flubs.
        condition_on_previous_text=False,
        # Keep every sound, so pauses and yawns show up as gaps between words.
        vad_filter=False,
    )

    words = []
    for seg in segments:
        for w in seg.words:
            words.append(
                {
                    "word": w.word.strip(),
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                    "prob": round(w.probability, 3),
                }
            )

    return {
        "audio": str(audio),
        "model": model_name,
        "duration": round(info.duration, 3),
        "words": words,
    }


def save(result: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
