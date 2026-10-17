"""Step 3: apply the alignment decisions to the audio.

The cleaned audio is a sequence of chunks of the original recording. A new
chunk starts wherever words were cut or a silence was too long. Pauses at the
joins keep real room tone from the recording (not digital silence), capped at
MAX_PAUSE, and chunks are joined with a short crossfade so the edits don't click.
"""

import subprocess
import wave
from pathlib import Path

import numpy as np

RATE = 48000
MAX_PAUSE = 1.2   # longest silence allowed in the output, in seconds
LEAD_IN = 0.3     # silence kept before the very first word
CROSSFADE = 0.03  # seconds


def plan(transcript: dict, alignment: dict, keep_extras: bool = True) -> list[dict]:
    """Return the chunks of original audio to keep, as {start, end} in seconds."""
    words = transcript["words"]
    keep = set(alignment["kept_words"])
    for e in alignment["events"]:
        if e["type"] == "extra" and keep_extras:
            keep.update(e["words"])
    keep = sorted(keep)
    if not keep:
        return []

    def gap_before(i):
        return words[i]["start"] - (words[i - 1]["end"] if i > 0 else 0.0)

    def gap_after(i):
        nxt = words[i + 1]["start"] if i + 1 < len(words) else transcript["duration"]
        return nxt - words[i]["end"]

    half = MAX_PAUSE / 2
    chunks = []
    first = keep[0]
    start = words[first]["start"] - min(gap_before(first), LEAD_IN)
    for a, b in zip(keep, keep[1:]):
        adjacent = b == a + 1
        if adjacent and gap_before(b) <= MAX_PAUSE:
            continue  # natural pause inside a chunk
        chunks.append({"start": start, "end": words[a]["end"] + min(gap_after(a), half)})
        start = words[b]["start"] - min(gap_before(b), half)
    last = keep[-1]
    chunks.append({"start": start, "end": words[last]["end"] + min(gap_after(last), half)})
    return [{"start": round(c["start"], 3), "end": round(c["end"], 3)} for c in chunks]


def load(path: Path) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
         "-f", "f32le", "-ac", "1", "-ar", str(RATE), "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32)


def render(audio: np.ndarray, chunks: list[dict]) -> np.ndarray:
    fade = int(CROSSFADE * RATE)
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    out = np.zeros(0, dtype=np.float32)
    for c in chunks:
        piece = audio[int(c["start"] * RATE):int(c["end"] * RATE)].copy()
        if len(out) >= fade and len(piece) >= fade:
            piece[:fade] = out[-fade:] * ramp[::-1] + piece[:fade] * ramp
            out = out[:-fade]
        out = np.concatenate([out, piece])
    return out


def write_wav(samples: np.ndarray, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(out), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(RATE)
        f.writeframes(pcm.tobytes())
