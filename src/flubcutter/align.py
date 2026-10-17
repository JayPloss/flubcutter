"""Step 2: align the transcript against the book text and find where they differ.

Core rule: the reader re-reads a passage until satisfied, so the LAST take of
any repeated passage is the one to keep. The alignment is a standard
edit-distance alignment, but the traceback runs from the end and prefers
matching, so later takes get paired with the book and earlier takes are left
over as "extra" transcript words.

A chapter may be recorded in several parts. Words carry a "part" index and
part-local times; the alignment runs over all parts in order.
"""

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

MATCH, SUB, EXTRA, MISSING = "match", "sub", "extra", "missing"

# A silence between kept words longer than this is worth trimming.
LONG_GAP = 1.0

# Alignments bigger than this many DP cells are split at anchor points.
MAX_CELLS = 12_000_000
ANCHOR_N = 6
PIECE_WORDS = 1500

FILLERS = {"um", "uh", "umm", "uhh", "er", "erm", "ah", "hmm", "mm", "mhm"}


def normalize(text: str) -> list[str]:
    """Lowercase words with punctuation stripped; hyphenated words are split."""
    text = text.lower().replace("’", "'").replace("‘", "'")
    text = re.sub(r"[-—–]", " ", text)
    return [w for w in (re.sub(r"[^a-z0-9']", "", t).strip("'") for t in text.split()) if w]


def _dp_align(book: list[int], spoken: list[int]) -> list[tuple[str, int | None, int | None]]:
    """Edit-distance alignment; ops are (kind, book_index, spoken_index)."""
    n, m = len(book), len(spoken)
    cost = np.empty((n + 1, m + 1), dtype=np.int32)
    steps = np.arange(m + 1, dtype=np.int32)
    cost[0] = steps
    sp = np.asarray(spoken, dtype=np.int64)
    for i in range(1, n + 1):
        prev = cost[i - 1]
        base = np.empty(m + 1, dtype=np.int32)
        base[0] = i
        base[1:] = np.minimum(prev[:-1] + (sp != book[i - 1]), prev[1:] + 1)
        # row[j] = min(base[j], row[j-1] + 1), as a running minimum.
        cost[i] = np.minimum.accumulate(base - steps) + steps

    # Traceback from the end, preferring the diagonal: when a passage was read
    # more than once, this pairs the book with the latest take.
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        c = cost[i, j]
        if i > 0 and j > 0:
            same = book[i - 1] == spoken[j - 1]
            if c == cost[i - 1, j - 1] + (0 if same else 1):
                ops.append((MATCH if same else SUB, i - 1, j - 1))
                i, j = i - 1, j - 1
                continue
        if j > 0 and c == cost[i, j - 1] + 1:
            ops.append((EXTRA, None, j - 1))
            j -= 1
        else:
            ops.append((MISSING, i - 1, None))
            i -= 1
    ops.reverse()
    return ops


def _anchors(book: list[int], spoken: list[int]) -> list[tuple[int, int]]:
    """Split points: n-grams that occur exactly once in both texts.

    A passage that was read twice has all its n-grams twice in the transcript,
    so an anchor can never fall inside a re-read region.
    """
    def grams(seq):
        return [tuple(seq[k:k + ANCHOR_N]) for k in range(len(seq) - ANCHOR_N + 1)]

    bg, sg = grams(book), grams(spoken)
    bc, sc = Counter(bg), Counter(sg)
    book_pos = {g: k for k, g in enumerate(bg) if bc[g] == 1}
    pairs = [(book_pos[g], j) for j, g in enumerate(sg) if sc[g] == 1 and g in book_pos]

    # Keep a run of pairs increasing in both texts (longest increasing subsequence).
    pairs.sort()
    tails, tails_idx, parent = [], [], [-1] * len(pairs)
    import bisect
    for k, (_, j) in enumerate(pairs):
        pos = bisect.bisect_left(tails, j)
        if pos == len(tails):
            tails.append(j)
            tails_idx.append(k)
        else:
            tails[pos] = j
            tails_idx[pos] = k
        parent[k] = tails_idx[pos - 1] if pos else -1
    chain, k = [], tails_idx[-1] if tails_idx else -1
    while k != -1:
        chain.append(pairs[k])
        k = parent[k]
    chain.reverse()

    splits, last = [], 0
    for i, j in chain:
        if i - last >= PIECE_WORDS:
            mid = ANCHOR_N // 2
            splits.append((i + mid, j + mid))
            last = i
    return splits


def align_words(book: list[str], spoken: list[str]) -> list[tuple[str, int | None, int | None]]:
    vocab: dict[str, int] = {}
    b = [vocab.setdefault(w, len(vocab)) for w in book]
    s = [vocab.setdefault(w, len(vocab)) for w in spoken]
    if (len(b) + 1) * (len(s) + 1) <= MAX_CELLS:
        return _dp_align(b, s)

    ops = []
    bounds = [(0, 0)] + _anchors(b, s) + [(len(b), len(s))]
    for (i0, j0), (i1, j1) in zip(bounds, bounds[1:]):
        for kind, bi, sj in _dp_align(b[i0:i1], s[j0:j1]):
            ops.append((kind, None if bi is None else bi + i0, None if sj is None else sj + j0))
    return ops


def _overlap(extra: list[str], text: list[str]) -> float:
    if not extra:
        return 0.0
    window = set(text)
    return sum(w in window for w in extra) / len(extra)


def analyze(words: list[dict], book_text: str) -> dict:
    """words: transcript words (all parts, in order), each with a "part" key."""
    flat, owner = [], []
    for idx, w in enumerate(words):
        for p in normalize(w["word"]):
            flat.append(p)
            owner.append(idx)

    book = normalize(book_text)
    ops = align_words(book, flat)

    events = []
    kept = []
    k = 0
    while k < len(ops):
        kind, b, s = ops[k]
        if kind == EXTRA:
            start = k
            while k < len(ops) and ops[k][0] == EXTRA:
                k += 1
            idxs = sorted({owner[ops[x][2]] for x in range(start, k)})
            # Split a span that crosses a part boundary.
            groups = []
            for i in idxs:
                if groups and words[groups[-1][-1]]["part"] == words[i]["part"]:
                    groups[-1].append(i)
                else:
                    groups.append([i])
            next_book = next((ops[x][1] for x in range(k, len(ops)) if ops[x][1] is not None), len(book))
            for g in groups:
                span = [p for i in g for p in normalize(words[i]["word"])]
                after = book[next_book:next_book + len(span) + 3]
                if all(w in FILLERS for w in span):
                    etype = "filler"
                elif _overlap(span, after) >= 0.6:
                    etype = "restart"
                else:
                    etype = "extra"
                events.append({
                    "type": etype,
                    "part": words[g[0]]["part"],
                    "text": " ".join(words[i]["word"] for i in g),
                    "start": words[g[0]]["start"],
                    "end": words[g[-1]]["end"],
                    "words": g,
                    "book_index": next_book,
                })
            continue
        if kind == SUB:
            w = words[owner[s]]
            events.append({"type": "sub", "part": w["part"], "said": w["word"], "book": book[b],
                           "start": w["start"], "end": w["end"], "words": [owner[s]],
                           "book_index": b})
        elif kind == MISSING:
            events.append({"type": "missing", "book": book[b], "book_index": b})
        if s is not None and (not kept or kept[-1] != owner[s]):
            kept.append(owner[s])
        k += 1

    # Missing words have no time; place them at the next kept word so they sort.
    for e in events:
        if e["type"] == "missing":
            nxt = next((op[2] for op in ops if op[1] is not None and op[1] > e["book_index"]
                        and op[2] is not None), None)
            if nxt is not None:
                w = words[owner[nxt]]
                e.update(part=w["part"], start=w["start"], end=w["start"])

    # Long silences between adjacent kept words in the same part, plus dead
    # air at the start of each part. Silence around a cut goes with the cut.
    for idx, w in enumerate(words):
        first_of_part = idx == 0 or words[idx - 1]["part"] != w["part"]
        if first_of_part and w["start"] > LONG_GAP:
            events.append({"type": "gap", "part": w["part"], "start": 0.0, "end": w["start"],
                           "seconds": round(w["start"], 2)})
    for a, b in zip(kept, kept[1:]):
        if b == a + 1 and words[a]["part"] == words[b]["part"]:
            gap = words[b]["start"] - words[a]["end"]
            if gap > LONG_GAP:
                events.append({"type": "gap", "part": words[a]["part"], "start": words[a]["end"],
                               "end": words[b]["start"], "seconds": round(gap, 2)})

    events.sort(key=lambda e: (e.get("part", 1 << 30), e.get("start", 0.0)))
    for e in events:
        e["id"] = f"{e['type']}-{e.get('part', 'x')}-{e.get('start', 0.0):.2f}-{e.get('book_index', 0)}"

    matched = sum(op[0] == MATCH for op in ops)
    return {
        "book_words": len(book),
        "matched": matched,
        "coverage": round(matched / len(book), 3) if book else 0,
        "kept_words": kept,
        "events": events,
    }


def book_context(book_text: str, book_index: int, before: int = 40, after: int = 40) -> tuple[str, str]:
    """Book text around a normalized-word index, for showing context."""
    raw = book_text.split()
    # normalize() may split hyphenated tokens; map normalized index -> raw index.
    norm_to_raw = []
    for r, tok in enumerate(raw):
        norm_to_raw.extend([r] * len(normalize(tok)))
    r = norm_to_raw[min(book_index, len(norm_to_raw) - 1)] if norm_to_raw else 0
    if book_index >= len(norm_to_raw):
        r = len(raw)
    return " ".join(raw[max(0, r - before):r]), " ".join(raw[r:r + after])


def report(result: dict) -> str:
    lines = [f"Matched {result['matched']}/{result['book_words']} book words "
             f"({result['coverage']:.0%})", ""]
    for e in result["events"]:
        t = f"p{e.get('part', 0)} {e['start']:7.2f}s" if "start" in e else "         -"
        match e["type"]:
            case "restart":
                lines.append(f"{t}  RESTART  cut: \"{e['text']}\"")
            case "filler":
                lines.append(f"{t}  FILLER   cut: \"{e['text']}\"")
            case "extra":
                lines.append(f"{t}  EXTRA    review: \"{e['text']}\"")
            case "sub":
                lines.append(f"{t}  SUB      said \"{e['said']}\", book has \"{e['book']}\"")
            case "missing":
                lines.append(f"{t}  MISSING  \"{e['book']}\" not read")
            case "gap":
                lines.append(f"{t}  GAP      {e['seconds']}s of silence")
    return "\n".join(lines)


def save(result: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
