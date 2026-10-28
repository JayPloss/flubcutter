"""Listen back to every join in a render, the way a careful editor would.

The rendered audio around each join (a couple of seconds either side) is
transcribed again and compared with the words that should be there. A
clipped word ending ("conversation" for "conversations") or a leaked piece
of a cut word shows up as a mismatch right at the join, and that join gets
rebuilt more cautiously (see edit.LEVELS).
"""

from pathlib import Path

from flubcutter.align import canon, normalize

# Seconds of output before and after the join to listen to; tried in turn.
WINDOWS = [(2.0, 2.0), (2.6, 1.7), (1.7, 2.6)]


def heard_ok(heard: list[str], before: list[str], after: list[str]) -> bool:
    """Does the last kept word before the join run straight into the first
    one after it, both intact and with nothing in between? (Words further
    from the join are left out: the recogniser hears those a little
    differently from time to time, which says nothing about the cut.)"""
    tidy = lambda w: canon("".join(normalize(w)))
    want = [tidy(w) for w in before[-1:] + after[:1]]
    heard = [tidy(w) for w in heard]
    n = len(want)
    for i in range(len(heard) - n + 1):
        if heard[i:i + n] == want:
            # The start of the take that was cut away, heard again: "told
            # Kreacher to to go" for "told Kreacher | to go". (Unless the
            # reading itself says it twice.)
            again_after = i + n < len(heard) and heard[i + n] == want[-1] \
                and [tidy(w) for w in after[1:2]] != want[-1:]
            again_before = i > 0 and heard[i - 1] == want[0] \
                and [tidy(w) for w in before[-2:-1]] != want[:1]
            return not (again_after or again_before)
    # A compound split at the join can come back closed up: "rabbit" "hole"
    # heard as "rabbithole".
    return any(h == "".join(want) for h in heard)


def check_joins(output: Path, joins: list[dict], only: set[str] | None = None,
                progress=None, book: dict[int, str] | None = None) -> dict[str, dict]:
    """{join key: {"ok": True/False, or None if nothing was heard, "heard": text}}
    for the joins in `only` (default all). book: the book's word for transcript
    words the recogniser got wrong the first time (the listen-back may well
    hear it right)."""
    book = book or {}
    from flubcutter.transcribe import SAMPLE_RATE, load_audio, words_in

    todo = [j for j in joins if only is None or j["key"] in only]
    if not todo:
        return {}
    audio = load_audio(output)
    results = {}
    for n, j in enumerate(todo):
        before = normalize(" ".join(j["before"]))
        after = normalize(" ".join(j["after"]))
        a_i, b_i = (int(x.split(":")[1]) for x in j["key"].split(">"))
        alt = (before[:-1] + normalize(book[a_i]) if a_i in book else before,
               normalize(book[b_i]) + after[1:] if b_i in book else after)
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
            said = normalize(" ".join(heard))
            ok = heard_ok(said, before, after) or heard_ok(said, *alt)
            result = {"ok": ok, "heard": " ".join(heard)}
            if ok:
                break
        results[j["key"]] = result
        if progress:
            progress((n + 1) / len(todo))
    return results
