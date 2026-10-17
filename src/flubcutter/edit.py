"""Step 3: apply the review decisions to the audio.

The cleaned chapter is a sequence of *runs*: stretches of kept words from the
original recordings. Between two runs is a *join*, which is where blips come
from, so each join is built from the audio itself rather than from the word
timestamps (which are only accurate to about 80 ms):

1. Find where the sound of the last kept word really ends, and where the
   sound of the next kept word really starts, using a 5 ms loudness envelope
   measured against the recording's own noise floor.
2. Cut only in silence (room tone), never through sound. If the reader ran
   straight from a kept word into a cut one with no gap at all, the join is a
   *tight* cut: it goes at the quietest point and fades quickly, and the
   render report flags it so it is easy to find and listen to.
3. Build the pause to the reader's own typical length for the punctuation
   there (measured from the rest of the recording: a full stop gets a longer
   pause than a comma), from the original silence on each side plus,
   if that is too short, real room tone taken from a nearby quiet stretch.
4. Join pieces with equal-power crossfades, so noise doesn't dip in level.

Every join has a stable key. When a listener reports a glitch at a join, the
join's *level* goes up and it is rebuilt more cautiously: stricter silence
detection, more room tone and less original audio around the cut, longer
fades and a slightly longer pause (see LEVELS).
"""

import math
import subprocess
import wave
from pathlib import Path

import numpy as np

RATE = 48000
MAX_PAUSE = 1.2    # longest silence allowed in the output, in seconds
LEAD_IN = 0.3      # silence before the first word of the chapter
END_TAIL = 0.8     # silence after the last word of the chapter
HOP = 0.005        # loudness envelope resolution
WIN = 0.02         # loudness envelope window
XFADE = 0.02       # crossfade between pieces of room tone
MIN_FILL = 0.06    # shortest room-tone fill between two pieces
END_QUIET = 0.15   # a silence this long after a word means the word is over

PUNCT = {".": ".", "!": ".", "?": ".", ":": ",", ";": ",", ",": ",", "-": ","}

# How cautiously a join is built. Level 0 is the default; each glitch report
# (or failed listen-back check) on a join moves it one level up. Silence
# detection stays the same at every level -- making it stricter turns room
# noise into "sound" and drags cuts into the neighbouring words -- so the
# levels widen the safety margins instead:
#   margin     how far a cut stays from any sound
#   keep       most original audio kept beside the cut (None: as much as the
#              pause needs); the rest of the pause is clean room tone
#   hard_fade  fade length for a tight cut through sound
#   pace       pause length, as a fraction of the reader's usual pause
#   shift      how much earlier a tight cut may land, to avoid catching the
#              start of the word that was cut
LEVELS = [
    dict(margin=0.015, keep=None, hard_fade=0.035, pace=1.0, shift=0.0),
    dict(margin=0.03, keep=0.12, hard_fade=0.05, pace=0.85, shift=0.03),
    dict(margin=0.05, keep=0.06, hard_fade=0.07, pace=0.7, shift=0.05),
]
QUIET_DB = 8.0     # silence: within this many dB of the recording's noise floor
MIN_QUIET = 0.04   # shortest gap that counts as silence
MAX_LEVEL = len(LEVELS) - 1


def keep_set(alignment: dict, decisions: dict[str, str]) -> set[int]:
    """Transcript word indices to keep, given per-event decisions ("cut"/"keep")."""
    keep = set(alignment["kept_words"])
    for e in alignment["events"]:
        if e["type"] in ("restart", "filler", "extra") and decisions.get(e["id"]) == "keep":
            keep.update(e["words"])
    return keep


def runs(words: list[dict], keep: set[int]) -> list[tuple[int, int]]:
    """Stretches of consecutive kept words, as (first, last) word indices.

    A run ends where words were cut or a recording part ends. (Pauses that
    are too long are found from the audio in render.)
    """
    order = sorted(keep)
    if not order:
        return []
    out = []
    first = order[0]
    for a, b in zip(order, order[1:]):
        same_part = words[a]["part"] == words[b]["part"]
        if same_part and b == a + 1:
            continue
        out.append((first, a))
        first = b
    out.append((first, order[-1]))
    return out


def join_key(words: list[dict], a: int, b: int) -> str:
    """Stable name for the join between kept word a and kept word b."""
    return f"{words[a]['part']}:{a}>{words[b]['part']}:{b}"


def load(path: Path, rate: int = RATE) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
         "-f", "f32le", "-ac", "1", "-ar", str(rate), "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32)


# ---------- acoustics ----------

class Acoustics:
    """Loudness envelope of one recording, relative to its own noise floor."""

    def __init__(self, audio: np.ndarray):
        self.audio = audio
        self.duration = len(audio) / RATE
        self.db = self._envelope(audio)
        self.floor = float(np.percentile(self.db, 10))
        # Soft consonants at word edges (s, f, th, the end of "hedge") carry
        # little energy overall but plenty of hiss, so silence also has to be
        # quiet in the high band (a second difference boosts it ~12 dB/octave).
        hiss = np.diff(audio, n=2, prepend=audio[:1], append=audio[-1:]) if len(audio) > 2 else audio
        self.db_hf = self._envelope(hiss)
        self.floor_hf = float(np.percentile(self.db_hf, 10))
        self._quiet = {}

    @staticmethod
    def _envelope(x: np.ndarray) -> np.ndarray:
        hop, win = int(HOP * RATE), int(WIN * RATE)
        n = max(1, (len(x) - win) // hop + 1)
        c = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2)])
        starts = np.minimum(np.arange(n) * hop, max(0, len(x) - win))
        energy = (c[np.minimum(starts + win, len(x))] - c[starts]) / win
        return 10 * np.log10(energy + 1e-12)

    def k(self, t: float) -> int:
        """Envelope frame whose window starts at time t."""
        return int(np.clip(round(t / HOP), 0, len(self.db) - 1))

    def quiet(self, level: int = 0):
        """(mask, run of quiet frames starting at k, run of quiet frames ending at k)."""
        level = 0  # the same at every level; see LEVELS
        if level not in self._quiet:
            q = (self.db < self.floor + QUIET_DB) & (self.db_hf < self.floor_hf + QUIET_DB)
            fwd = np.zeros(len(q) + 1, dtype=np.int64)
            for k in range(len(q) - 1, -1, -1):
                fwd[k] = fwd[k + 1] + 1 if q[k] else 0
            bwd = np.zeros(len(q) + 1, dtype=np.int64)  # bwd[k + 1]: run ending at k
            for k in range(len(q)):
                bwd[k + 1] = bwd[k] + 1 if q[k] else 0
            self._quiet[level] = (q, fwd[:-1], bwd[1:])
        return self._quiet[level]

    def valley(self, t0: float, t1: float) -> float:
        """Quietest point between t0 and t1 (smoothed over 15 ms)."""
        k0, k1 = self.k(t0), self.k(t1)
        if k1 <= k0:
            return t0
        seg = np.convolve(self.db[k0:k1 + 1], np.ones(3) / 3, mode="same")
        return (k0 + int(np.argmin(seg))) * HOP + WIN / 2

    def room_tone(self, t: float, seconds: float) -> np.ndarray:
        """Real background noise from the quiet stretch nearest to time t."""
        n = max(1, int(seconds * RATE))
        q, fwd, _ = self.quiet(0)
        starts = np.nonzero(q & np.concatenate([[True], ~q[:-1]]))[0]
        lengths = fwd[starts] * HOP + WIN - 2 * LEVELS[0]["margin"]
        good = lengths >= 0.15
        if good.any():
            starts, lengths = starts[good], lengths[good]
            centre = starts * HOP + LEVELS[0]["margin"] + lengths / 2
            i = int(np.argmin(np.abs(centre - t) / np.minimum(lengths, 1.0)))
            a = int((starts[i] * HOP + LEVELS[0]["margin"]) * RATE)
            source = self.audio[a:a + int(lengths[i] * RATE)]
        else:  # never quiet: use the quietest 150 ms there is
            k = int(np.argmin(np.convolve(self.db, np.ones(30), mode="valid")))
            source = self.audio[k * int(HOP * RATE):][: int(0.15 * RATE)]
        if len(source) >= n:
            off = (len(source) - n) // 2
            return source[off:off + n].copy()
        out = source.copy()  # loop it
        while len(out) < n:
            out = _xfade(out, source, min(int(XFADE * RATE), len(source) // 3))
        return out[:n]


def _xfade(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """a then b, overlapped by n samples with an equal-power crossfade."""
    n = min(n, len(a), len(b))
    if n <= 0:
        return np.concatenate([a, b])
    x = np.linspace(0.0, 1.0, n, dtype=np.float32)
    mid = a[-n:] * np.cos(x * math.pi / 2) + b[:n] * np.sin(x * math.pi / 2)
    return np.concatenate([a[:-n], mid, b[n:]])


def out_point(ac: Acoustics, word: dict, hi: float, level: int) -> dict:
    """Where the sound of a kept word ends, and how much silence follows it.

    hi is the start of the next word in the same recording (which is not
    kept here), or the end of the recording.
    """
    L = LEVELS[level]
    q, fwd, _ = ac.quiet()
    m = max(1, round(MIN_QUIET / HOP))
    end = min(word["end"], hi)
    # The word ends at the first real silence after it gets going: one that
    # lasts (END_QUIET) or runs up to the next word. Shorter gaps are inside
    # the word, like the closure before the "s" of "conversations" or the
    # "dge" of "hedge".
    k = ac.k(word["start"] + min(0.12, 0.5 * max(0.0, end - word["start"])))
    last_word = hi >= ac.duration
    # Timestamps can be early, but never by much: don't look far past them.
    stop = ac.k(min(word["end"] + 0.3, hi + 0.02))
    while k <= stop:
        if q[k] and fwd[k] >= m:
            run_end = (k + fwd[k] - 1) * HOP + WIN
            if fwd[k] * HOP >= END_QUIET or (not last_word and run_end >= hi - 0.12):
                t = k * HOP
                avail = run_end - t - L["margin"]
                return {"t": t, "avail": max(0.0, min(avail, ac.duration - t)), "hard": False}
            k += fwd[k]
        else:
            k += 1
    # No silence at all before the next word: cut at the quietest point.
    lo = max(word["start"] + 0.02, word["end"] - 0.10 - L["shift"])
    return {"t": ac.valley(lo, min(hi + 0.03, word["end"] + 0.03)), "avail": 0.0, "hard": True}


def in_point(ac: Acoustics, word: dict, lo: float | None, level: int) -> dict:
    """Where the sound of a kept word starts, and how much silence precedes it.

    lo is the start of the previous word in the same recording (not kept
    here), or None at the start of the recording.
    """
    L = LEVELS[level]
    q, _, bwd = ac.quiet()
    m = max(1, round(MIN_QUIET / HOP))
    # First sound at the word's timestamp, allowing a little lateness.
    k = ac.k(word["start"] - 0.02)
    limit = ac.k(word["start"] + 0.1)
    while k < limit and q[k]:
        k += 1
    onset = k
    floor_k = ac.k(max(word["start"] - 0.6, (lo + 0.05) if lo is not None else 0.0))

    def silence(back: int, need: int):
        j = onset - 1
        while j >= max(back, floor_k):
            if q[j] and bwd[j] >= need:
                return j
            j -= 1
        return None

    # Usually there is silence just before the word (within 120 ms: the
    # onset can come a little before its timestamp). Failing that, the word
    # may be led by a breath; keep the whole breath if a real pause precedes
    # it, rather than chopping it in half.
    found = silence(onset - round(0.12 / HOP), m)
    if found is None:
        found = silence(onset - round(0.6 / HOP), round(END_QUIET / HOP))
    if found is not None:
        t = found * HOP + WIN
        start = (found - bwd[found] + 1) * HOP
        # "speech" is where the word itself starts; anything between t and
        # speech is a breath, which counts as part of the pause.
        return {"t": t, "avail": max(0.0, t - start - L["margin"]), "hard": False,
                "speech": max(t, onset * HOP)}
    if onset * HOP <= L["margin"] or (lo is None and q[: onset].all()):
        return {"t": onset * HOP, "avail": onset * HOP, "hard": False, "speech": onset * HOP}
    t = ac.valley(word["start"] - 0.10, word["start"] + 0.03)
    return {"t": t, "avail": 0.0, "hard": True, "speech": t}


def _punct(word: dict) -> str:
    w = word["word"].rstrip("\"'”’)]")
    p = w[-1:] if w else ""
    return PUNCT.get(p, "")


def gap(a: dict, b: dict) -> float:
    """Pause between the end of one word's sound and the start of the next
    word itself (a breath in between counts as pause); 0 for run-on words."""
    if a["hard"] or b["hard"]:
        return 0.0
    return max(0.0, b["speech"] - a["t"])


# Used when a recording is too short to measure the reader's own pacing.
DEFAULT_RHYTHM = {
    ".": {"typ": 0.6, "long": 0.8, "max": 1.1},
    ",": {"typ": 0.3, "long": 0.4, "max": 0.6},
    "": {"typ": 0.05, "long": 0.12, "max": 0.35},
}


def rhythm_of(gaps: dict[str, list[float]]) -> dict[str, dict]:
    """The reader's pacing per context, from measured gaps between kept words:
    typ (median) is what a cut gets; long (90th percentile) is the long end
    of normal; max is where a pause stops sounding like reading and starts
    sounding like a mistake."""
    out = {}
    for cls, dflt in DEFAULT_RHYTHM.items():
        g = np.array([x for x in gaps.get(cls, []) if x < 3.0])
        if len(g) < 8:
            out[cls] = dict(dflt, n=int(len(g)))
            continue
        typ, p90, p97 = np.percentile(g, [50, 90, 97])
        out[cls] = {"typ": float(min(typ, MAX_PAUSE)),
                    "long": float(min(p90, MAX_PAUSE)),
                    "max": float(min(max(p97, typ + 0.25), MAX_PAUSE)),
                    "n": int(len(g))}
    return out


def pacing(out: np.ndarray, spans: list[tuple[dict, float, float]], rhythm: dict) -> list[dict]:
    """Pauses in the finished audio that fall outside the reader's own range.

    spans: each kept word, in order, with its start and end in the output.
    This measures the result itself, so it catches odd pauses wherever they
    come from: at a cut or in the original reading.
    """
    ac = Acoustics(out)
    flags = []
    for (w1, s1, e1), (w2, s2, e2) in zip(spans, spans[1:]):
        a = out_point(ac, {"word": w1["word"], "start": s1, "end": min(e1, s2)}, s2, 0)
        b = in_point(ac, {"word": w2["word"], "start": s2, "end": e2}, s1, 0)
        cls = _punct(w1)
        g = gap(a, b)
        if g > rhythm[cls]["max"] + 0.05:
            flags.append({"time": round(a["t"], 2), "gap": round(g, 2),
                          "usual": round(rhythm[cls]["typ"], 2),
                          "words": [w1["word"], w2["word"]]})
    return flags


# ---------- render ----------

def render(audios: list[np.ndarray], words: list[dict], keep: set[int],
           levels: dict[str, int] | None = None) -> tuple[np.ndarray, dict]:
    """The cleaned chapter, and a report of every run and join in it."""
    levels = levels or {}
    rs = runs(words, keep)
    if not rs:
        return np.zeros(0, dtype=np.float32), {"chunks": [], "joins": []}
    acs = {}

    def ac(part):
        if part not in acs:
            acs[part] = Acoustics(audios[part])
        return acs[part]

    def neighbours(i):
        w = words[i]
        prev = words[i - 1] if i > 0 and words[i - 1]["part"] == w["part"] else None
        nxt = words[i + 1] if i + 1 < len(words) and words[i + 1]["part"] == w["part"] else None
        return prev, nxt

    # Measure the reader's pacing on every pair of kept neighbours, and split
    # runs where a pause is too long to sound like reading.
    measured, gaps = {}, {".": [], ",": [], "": []}
    for first, last in rs:
        a_c = ac(words[first]["part"])
        for i in range(first, last):
            a = out_point(a_c, words[i], words[i + 1]["start"], 0)
            b = in_point(a_c, words[i + 1], words[i]["start"], 0)
            measured[i] = gap(a, b)
            gaps[_punct(words[i])].append(measured[i])
    rhythm = rhythm_of(gaps)
    split = []
    for first, last in rs:
        start = first
        for i in range(first, last):
            if measured[i] > rhythm[_punct(words[i])]["max"]:
                split.append((start, i))
                start = i + 1
        split.append((start, last))
    rs = split
    keys = [join_key(words, a[1], b[0]) for a, b in zip(rs, rs[1:])]
    lvl = [min(MAX_LEVEL, max(0, int(levels.get(k, 0)))) for k in keys]

    # Acoustic boundaries of each run.
    sides = []
    for r, (first, last) in enumerate(rs):
        p = words[first]["part"]
        prev, _ = neighbours(first)
        _, nxt = neighbours(last)
        lv_in = lvl[r - 1] if r > 0 else 0
        lv_out = lvl[r] if r < len(lvl) else 0
        a_in = in_point(ac(p), words[first], prev["start"] if prev else None, lv_in)
        a_out = out_point(ac(p), words[last], nxt["start"] if nxt else ac(p).duration, lv_out)
        if a_out["t"] <= a_in["t"]:  # degenerate (a tiny word): keep the timestamps
            a_in = dict(a_in, t=words[first]["start"])
            a_out = dict(a_out, t=max(words[last]["end"], a_in["t"] + 0.05))
        sides.append((a_in, a_out))

    # Decide how much silence each side contributes to each pause.
    head = [0.0] * len(rs)
    tail = [0.0] * len(rs)
    fill = [0.0] * len(rs)  # room tone after run r
    pauses = []
    for r in range(len(rs) - 1):
        (_, A), (B, _) = sides[r], sides[r + 1]
        L = LEVELS[lvl[r]]
        wa, wb = words[rs[r][1]], words[rs[r + 1][0]]
        same = wa["part"] == wb["part"]
        consecutive = same and rs[r + 1][0] == rs[r][1] + 1
        R = rhythm[_punct(wa)]
        if consecutive:
            # One of the reader's own pauses, too long to sound like reading:
            # shorten it to the long end of normal, keeping some emphasis.
            want = max(R["long"], 0.8 * R["max"])
        else:  # a cut: the reader's usual pause here
            want = R["typ"]
        want = min(MAX_PAUSE, want * L["pace"])
        # A breath before the next word is part of the pause. If it alone
        # would make the pause too long, leave it out.
        breath = max(0.0, B["speech"] - B["t"])
        if breath > 0.05 and breath > want - MIN_FILL:
            c = ac(wb["part"])
            t = c.valley(B["speech"] - 0.04, B["speech"] + 0.005)
            B = {"t": t, "avail": 0.0, "hard": True, "speech": t}
            sides[r + 1] = (B, sides[r + 1][1])
            breath = 0.0
        want = max(0.0, want - breath)
        keep_a = A["avail"] if L["keep"] is None else min(A["avail"], L["keep"])
        keep_b = B["avail"] if L["keep"] is None else min(B["avail"], L["keep"])
        # Original silence from both sides, topped up with room tone; there is
        # always a little room tone in the middle so the crossfades meet noise.
        room = max(0.0, want - MIN_FILL)
        t_ = min(keep_a, room / 2)
        h_ = min(keep_b, room - t_)
        t_ = min(keep_a, room - h_)
        tail[r], head[r + 1] = t_, h_
        fill[r] = max(MIN_FILL, want - t_ - h_)
        pauses.append(want + breath)
    head[0] = min(sides[0][0]["avail"], LEAD_IN)
    tail[-1] = min(sides[-1][1]["avail"], END_TAIL)

    # Assemble.
    xf = int(XFADE * RATE)
    out = np.zeros(0, dtype=np.float32)
    chunks, run_pos = [], []
    lead = LEAD_IN - head[0]
    if lead > 0.02:
        out = ac(words[rs[0][0]]["part"]).room_tone(sides[0][0]["t"], lead)
    for r, (first, last) in enumerate(rs):
        p = words[first]["part"]
        A_in, A_out = sides[r]
        # A tight cut fades across the quietest point rather than up to it.
        s = A_in["t"] - head[r]
        if r > 0 and A_in["hard"]:
            s -= LEVELS[lvl[r - 1]]["hard_fade"] / 2
        e = A_out["t"] + tail[r]
        if r < len(rs) - 1 and A_out["hard"]:
            e += LEVELS[lvl[r]]["hard_fade"] / 2
        s, e = max(0.0, s), min(ac(p).duration, e)
        piece = audios[p][int(s * RATE):int(e * RATE)].astype(np.float32)
        fade_in = int(LEVELS[lvl[r - 1]]["hard_fade"] * RATE) if r > 0 and A_in["hard"] else xf
        start_out = len(out) - (min(fade_in, len(out), len(piece)) if len(out) else 0)
        out = _xfade(out, piece, fade_in) if len(out) else piece.copy()
        run_pos.append(start_out / RATE - s)  # output time = source time + offset
        chunks.append({"part": p, "start": round(s, 3), "end": round(e, 3)})
        if r < len(rs) - 1:
            hard_out = A_out["hard"]
            fo = int(LEVELS[lvl[r]]["hard_fade"] * RATE) if hard_out else xf
            fade_next = (int(LEVELS[lvl[r]]["hard_fade"] * RATE) if sides[r + 1][0]["hard"] else xf)
            # The fill also absorbs the length lost to the two crossfades.
            n = fill[r] + (fo + fade_next) / RATE
            tone = ac(p).room_tone(A_out["t"], n)
            out = _xfade(out, tone, fo)
    end_fade = min(len(out), int(0.05 * RATE))
    if end_fade:
        out[-end_fade:] *= np.linspace(1.0, 0.0, end_fade, dtype=np.float32)

    joins = []
    for r in range(len(rs) - 1):
        (_, A), (B, _) = sides[r], sides[r + 1]
        wa, wb = rs[r][1], rs[r + 1][0]
        same = words[wa]["part"] == words[wb]["part"]
        kind = ("pause" if same and wb == wa + 1 else "cut") if same else "part"
        a_end = A["t"] + run_pos[r]
        b_on = B["speech"] + run_pos[r + 1]
        joins.append({
            "key": keys[r], "kind": kind, "level": lvl[r],
            "tight": bool(A["hard"] or B["hard"]),
            "start": round(a_end, 3), "end": round(max(b_on, a_end), 3),
            "pause": round(pauses[r], 2),
            "words": [words[wa]["word"], words[wb]["word"]],
            "before": [w["word"] for w in words[max(rs[r][0], wa - 2):wa + 1]],
            "after": [w["word"] for w in words[wb:min(rs[r + 1][1], wb + 2) + 1]],
            "source": [{"part": words[wa]["part"], "t": round(A["t"], 3)},
                       {"part": words[wb]["part"], "t": round(B["t"], 3)}],
        })
    spans = []
    for r, (first, last) in enumerate(rs):
        for i in range(first, last + 1):
            spans.append((words[i], words[i]["start"] + run_pos[r], words[i]["end"] + run_pos[r]))
    return out, {"chunks": chunks, "joins": joins, "pacing": pacing(out, spans, rhythm),
                 "rhythm": {k or "word": {x: round(v, 2) for x, v in d.items()} for k, d in rhythm.items()}}


def write_wav(samples: np.ndarray, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(out), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(RATE)
        f.writeframes(pcm.tobytes())
