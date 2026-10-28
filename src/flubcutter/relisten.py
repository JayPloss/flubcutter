"""Step 4: listen to the cleaned chapter for repeats the transcript missed.

Hearing a whole recording, the transcriber often takes a quick stutter
("to to go", "it's it's gone", "a b- basin") for one word, so the aligner
never sees a second take to cut and both stay in. Heard again in the
cleaned audio, the stutter usually comes out as the words it is. Each
repeat that isn't in the book is written back into the recording's
transcript at the times it was heard, and the next alignment finds it as a
restart (or a stray word for the judge) and cuts it.

A repeat is only written back where the transcript has fewer words over
that stretch than were heard, all of them among the words heard: anything
else is the recogniser hearing the same words two ways ("but - well" heard
as "but but"), not a stutter.

A repeat can also be heard where the transcript had it right and the AI
judge kept the first try as "the book word, misheard" ("Mondio of" before
"moaned the elf"). Hearing it come out as a repeat settles that: it's cut.
"""

from flubcutter.align import canon, normalize

LONGEST = 4      # longest phrase looked for said twice over
PARTIAL = 3      # longest broken-off start of a word ("b" before "basin")


def _tokens(word: str) -> list[str]:
    return [canon(t) for t in normalize(word)]


def find_repeats(heard: list[dict], book_text: str) -> list[dict]:
    """Repeats in heard words ({"word", "start", "end"}) that the book doesn't
    have: a phrase said twice ("to to"), said twice with a different last
    word ("moaned the orph, moaned the elf"), or a word broken off and
    started again ("b basin"). Each is {"first", "last"}: heard word indices."""
    book = " " + " ".join(canon(t) for t in normalize(book_text)) + " "
    toks = [" ".join(_tokens(w["word"])) for w in heard]
    in_book = lambda i, j: (" " + " ".join(toks[i:j]) + " ") in book
    out, k = [], 0
    while k < len(toks):
        hit = None
        for n in range(1, LONGEST + 1):
            a, b = toks[k:k + n], toks[k + n:k + 2 * n]
            if len(b) < n or not all(a):
                break
            if a == b or (n >= 3 and a[:-1] == b[:-1]):
                hit = (k, k + 2 * n - 1)
        nxt = toks[k + 1] if k + 1 < len(toks) else ""
        if hit is None and toks[k] and len(toks[k]) <= PARTIAL and " " not in toks[k] \
                and nxt.startswith(toks[k]) and len(nxt) > len(toks[k]):
            hit = (k, k + 1)
        if hit and not in_book(hit[0], hit[1] + 1):
            out.append({"first": hit[0], "last": hit[1]})
            k = hit[1] + 1
        else:
            k += 1
    return out


def to_source(chunks: list[dict], t: float) -> tuple[int, float] | None:
    """(part, time in its recording) for a time in the cleaned audio, from
    the render's chunks; None in a fill between chunks."""
    for c in reversed(chunks):
        if "at" in c and c["at"] <= t <= c["at"] + c["end"] - c["start"]:
            return c["part"], c["start"] + t - c["at"]
    return None


def source_words(heard: list[dict], spot: dict, chunks: list[dict], words: list[dict],
                 across: bool = False) -> tuple[int, float, list[int]] | None:
    """Where a spot is in the recording: (part, offset from cleaned-audio
    time, indices of the chapter words there), or None if it crosses a cut
    (unless `across`: then the words from its start to its end)."""
    h = heard[spot["first"]:spot["last"] + 1]
    a, b = to_source(chunks, h[0]["start"]), to_source(chunks, h[-1]["end"])
    if not a or not b or a[0] != b[0] or b[1] < a[1]:
        return None
    part, off = a[0], a[1] - h[0]["start"]
    if abs((b[1] - h[-1]["end"]) - off) > 0.01 and not across:
        return None  # a cut in between
    mid = lambda w: (w["start"] + w["end"]) / 2
    idx = [i for i, w in enumerate(words) if w["part"] == part and a[1] - 0.05 <= mid(w) <= b[1] + 0.05]
    if not idx or idx != list(range(idx[0], idx[-1] + 1)):
        return None
    return part, off, idx


def patches(heard: list[dict], spots: list[dict], chunks: list[dict],
            words: list[dict], skip: set[int] = frozenset()) -> list[dict]:
    """For each spot, the chapter words to replace: {"first", "count"} (global
    word indices, as in `words`, each with a "part") and the "words" heard
    there, in the recording's time. Spots that cross a cut, or fall on words
    in `skip` (asides, kept as recorded), or don't pass the check in the
    module docstring, are left out."""
    out = []
    for s in spots:
        h = heard[s["first"]:s["last"] + 1]
        where = source_words(heard, s, chunks, words)
        if not where or skip & set(where[2]):
            continue
        part, off, idx = where
        said = [t for i in idx for t in _tokens(words[i]["word"])]
        got = [t for w in h for t in _tokens(w["word"])]
        if len(said) >= len(got) or not set(said) <= set(got):
            continue
        out.append({"first": idx[0], "count": len(idx), "part": part,
                    "words": [{"word": w["word"], "start": round(w["start"] + off, 3),
                               "end": round(w["end"] + off, 3), "prob": w.get("prob", 1.0),
                               "heard_again": True} for w in h]})
    return out


def shift_key(key: str, first: int, count: int, new: int) -> str:
    """A join key ("0:12>0:15") after words first..first+count-1 became `new` words."""
    def move(x: int) -> int:
        if x >= first + count:
            return x + new - count
        return x if x < first else first + min(x - first, new - 1)
    out = []
    for side in key.split(">"):
        p, i = side.split(":")
        out.append(f"{p}:{move(int(i))}")
    return ">".join(out)
