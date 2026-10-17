"""Step 3: apply the review decisions to the audio.

The cleaned chapter is a sequence of chunks of the original recordings. A new
chunk starts wherever words were cut, a silence was too long, or a recording
part ends. Pauses at the joins keep real room tone from the recording (not
digital silence), capped at MAX_PAUSE, and chunks are joined with a short
crossfade so the edits don't click.
"""

import subprocess
import wave
from pathlib import Path

import numpy as np

RATE = 48000
MAX_PAUSE = 1.2   # longest silence allowed in the output, in seconds
LEAD_IN = 0.3     # silence kept before the first word of each part
CROSSFADE = 0.03  # seconds
SNAP_RADIUS = 0.1  # how far a cut may move to land on the quietest point


def keep_set(alignment: dict, decisions: dict[str, str]) -> set[int]:
    """Transcript word indices to keep, given per-event decisions ("cut"/"keep")."""
    keep = set(alignment["kept_words"])
    for e in alignment["events"]:
        if e["type"] in ("restart", "filler", "extra") and decisions.get(e["id"]) == "keep":
            keep.update(e["words"])
    return keep


def plan(words: list[dict], keep: set[int], durations: list[float]) -> list[dict]:
    """Chunks of original audio to keep, as {part, start, end} in part-local seconds."""
    order = sorted(keep)
    if not order:
        return []

    def gap_before(i):
        same = i > 0 and words[i - 1]["part"] == words[i]["part"]
        return words[i]["start"] - (words[i - 1]["end"] if same else 0.0)

    def gap_after(i):
        same = i + 1 < len(words) and words[i + 1]["part"] == words[i]["part"]
        nxt = words[i + 1]["start"] if same else durations[words[i]["part"]]
        return nxt - words[i]["end"]

    half = MAX_PAUSE / 2
    chunks = []

    def open_at(i):
        first_of_part = i == 0 or words[i - 1]["part"] != words[i]["part"]
        return words[i]["start"] - min(gap_before(i), LEAD_IN if first_of_part else half)

    first = order[0]
    start = open_at(first)
    for a, b in zip(order, order[1:]):
        same_part = words[a]["part"] == words[b]["part"]
        if same_part and b == a + 1 and gap_before(b) <= MAX_PAUSE:
            continue  # natural pause inside a chunk
        chunks.append({"part": words[a]["part"], "start": start,
                       "end": words[a]["end"] + min(gap_after(a), half)})
        start = open_at(b)
    last = order[-1]
    chunks.append({"part": words[last]["part"], "start": start,
                   "end": words[last]["end"] + min(gap_after(last), half)})
    return [{"part": c["part"], "start": round(max(c["start"], 0.0), 3),
             "end": round(c["end"], 3)} for c in chunks]


def load(path: Path, rate: int = RATE) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
         "-f", "f32le", "-ac", "1", "-ar", str(rate), "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32)


def snap(audio: np.ndarray, t: float, radius: float = SNAP_RADIUS) -> int:
    """Sample index of the quietest 10 ms near time t.

    Word timestamps are only accurate to a frame or two, so a cut placed right
    on a word boundary can clip the end of a kept word or leak the start of a
    cut one. Moving the cut to the nearby energy minimum avoids both.
    """
    frame = RATE // 100
    lo = max(0, int((t - radius) * RATE))
    hi = min(len(audio), int((t + radius) * RATE))
    if hi - lo < 2 * frame:
        return min(max(int(t * RATE), 0), len(audio))
    seg = audio[lo:hi]
    n = len(seg) // frame
    energy = (seg[: n * frame].reshape(n, frame) ** 2).mean(axis=1)
    return lo + int(np.argmin(energy)) * frame + frame // 2


def render(audios: list[np.ndarray], chunks: list[dict]) -> np.ndarray:
    fade = int(CROSSFADE * RATE)
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    pieces = []
    tail = None
    for c in chunks:
        audio = audios[c["part"]]
        start = 0 if c["start"] <= 0 else snap(audio, c["start"])
        end = snap(audio, c["end"])
        piece = audio[start:max(end, start)].copy()
        if tail is not None and len(tail) >= fade and len(piece) >= fade:
            piece[:fade] = tail[-fade:] * ramp[::-1] + piece[:fade] * ramp
            pieces[-1] = tail[:-fade]
        pieces.append(piece)
        tail = piece
    return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)


def write_wav(samples: np.ndarray, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(out), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(RATE)
        f.writeframes(pcm.tobytes())
