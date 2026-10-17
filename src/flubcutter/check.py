"""Listen back to every join in a render, the way a careful editor would.

The rendered audio around each join (a couple of seconds either side) is
transcribed again and compared with the words that should be there. A
clipped word ending ("conversation" for "conversations") or a leaked piece
of a cut word shows up as a mismatch right at the join, and that join gets
rebuilt more cautiously (see edit.LEVELS).
"""

from pathlib import Path

from flubcutter.align import normalize

# Seconds of output before and after the join to listen to; tried in turn.
WINDOWS = [(2.0, 2.0), (2.6, 1.7), (1.7, 2.6)]


def heard_ok(heard: list[str], before: list[str], after: list[str]) -> bool:
    """Does the last kept word before the join run straight into the first
    one after it, both intact and with nothing in between? (Words further
    from the join are left out: the recogniser hears those a little
    differently from time to time, which says nothing about the cut.)"""
    want = before[-1:] + after[:1]
    n = len(want)
    return any(heard[i:i + n] == want for i in range(len(heard) - n + 1))


def check_joins(output: Path, joins: list[dict], only: set[str] | None = None,
                progress=None) -> dict[str, dict]:
    """{join key: {"ok": True/False, or None if nothing was heard, "heard": text}}
    for the joins in `only` (default all)."""
    from flubcutter.transcribe import SAMPLE_RATE, load_audio, words_in

    todo = [j for j in joins if only is None or j["key"] in only]
    if not todo:
        return {}
    audio = load_audio(output)
    results = {}
    for n, j in enumerate(todo):
        before = normalize(" ".join(j["before"]))
        after = normalize(" ".join(j["after"]))
        result = {"ok": None, "heard": ""}
        # The recogniser occasionally returns nothing for a clip that starts
        # mid-word; nudging the window fixes that. A join only fails if it was
        # heard, and heard wrong, every time.
        for pad in WINDOWS:
            a = max(0, int((j["start"] - pad[0]) * SAMPLE_RATE))
            b = min(len(audio), int((j["end"] + pad[1]) * SAMPLE_RATE))
            heard = words_in(audio[a:b])
            if not heard:
                continue
            ok = heard_ok(normalize(" ".join(heard)), before, after)
            result = {"ok": ok, "heard": " ".join(heard)}
            if ok:
                break
        results[j["key"]] = result
        if progress:
            progress((n + 1) / len(todo))
    return results
