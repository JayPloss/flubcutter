"""Step 2: align the transcript against the book text and find where they differ.

Core rule: the reader re-reads a passage until satisfied, so the LAST take of
any repeated passage is the one to keep. The alignment is a standard
edit-distance alignment, but the traceback runs from the end and prefers
matching, so later takes get paired with the book and earlier takes are left
over as "extra" transcript words.
"""

import json
import re
from pathlib import Path

MATCH, SUB, EXTRA, MISSING = "match", "sub", "extra", "missing"

# A silence between kept words longer than this is worth flagging (yawn, pause).
LONG_GAP = 1.0


def normalize(text: str) -> list[str]:
    """Lowercase words with punctuation stripped; hyphenated words are split."""
    text = text.lower().replace("’", "'").replace("-", " ").replace("—", " ")
    return [w for w in (re.sub(r"[^a-z0-9']", "", t) for t in text.split()) if w]


def align_words(book: list[str], spoken: list[str]) -> list[tuple[str, int | None, int | None]]:
    """Return ops (kind, book_index, spoken_index), in order."""
    n, m = len(book), len(spoken)
    # cost[i][j] = cheapest alignment of book[:i] with spoken[:j]
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = i
    for j in range(1, m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        bi = book[i - 1]
        row, prev = cost[i], cost[i - 1]
        for j in range(1, m + 1):
            diag = prev[j - 1] + (0 if bi == spoken[j - 1] else 1)
            row[j] = min(diag, row[j - 1] + 1, prev[j] + 1)

    # Traceback from the end, preferring the diagonal: when a passage was read
    # more than once, this pairs the book with the latest take.
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            same = book[i - 1] == spoken[j - 1]
            if cost[i][j] == cost[i - 1][j - 1] + (0 if same else 1):
                ops.append((MATCH if same else SUB, i - 1, j - 1))
                i, j = i - 1, j - 1
                continue
        if j > 0 and cost[i][j] == cost[i][j - 1] + 1:
            ops.append((EXTRA, None, j - 1))
            j -= 1
        else:
            ops.append((MISSING, i - 1, None))
            i -= 1
    ops.reverse()
    return ops


def is_restart(extra: list[str], book_after: list[str]) -> bool:
    """True if an extra span is an earlier attempt at the text that follows it."""
    if not extra:
        return False
    window = set(book_after[: len(extra) + 3])
    return sum(w in window for w in extra) / len(extra) >= 0.6


def analyze(transcript: dict, book_text: str) -> dict:
    words = transcript["words"]
    spoken = [normalize(w["word"]) for w in words]
    # One Whisper token can normalize to several words (e.g. "daisy-chain").
    flat, owner = [], []
    for idx, parts in enumerate(spoken):
        for p in parts:
            flat.append(p)
            owner.append(idx)

    book = normalize(book_text)
    ops = align_words(book, flat)

    events = []
    kept = []  # transcript word indices that line up with the book
    k = 0
    while k < len(ops):
        kind, b, s = ops[k]
        if kind == EXTRA:
            start = k
            while k < len(ops) and ops[k][0] == EXTRA:
                k += 1
            idxs = sorted({owner[ops[x][2]] for x in range(start, k)})
            next_book = next((ops[x][1] for x in range(k, len(ops)) if ops[x][1] is not None), len(book))
            span_words = [flat[ops[x][2]] for x in range(start, k)]
            events.append({
                "type": "restart" if is_restart(span_words, book[next_book:]) else "extra",
                "text": " ".join(words[i]["word"] for i in idxs),
                "start": words[idxs[0]]["start"],
                "end": words[idxs[-1]]["end"],
                "words": idxs,
            })
            continue
        if kind == SUB:
            w = words[owner[s]]
            events.append({"type": "sub", "said": w["word"], "book": book[b],
                           "start": w["start"], "end": w["end"], "words": [owner[s]]})
        elif kind == MISSING:
            events.append({"type": "missing", "book": book[b], "book_index": b})
        if s is not None and (not kept or kept[-1] != owner[s]):
            kept.append(owner[s])
        k += 1

    # Long silences between adjacent kept words (yawns, pauses), plus any dead
    # air before the first word. Silence around a cut span goes with the cut.
    edges = [(None, 0)] if words else []
    edges += [(a, b) for a, b in zip(kept, kept[1:]) if b == a + 1]
    for a, b in edges:
        gap_start = words[a]["end"] if a is not None else 0.0
        gap = words[b]["start"] - gap_start
        if gap > LONG_GAP:
            events.append({"type": "gap", "start": gap_start, "end": words[b]["start"],
                           "seconds": round(gap, 2)})

    events.sort(key=lambda e: e.get("start", float("inf")))
    matched = sum(op[0] == MATCH for op in ops)
    return {
        "audio": transcript["audio"],
        "book_words": len(book),
        "matched": matched,
        "coverage": round(matched / len(book), 3) if book else 0,
        "kept_words": kept,
        "events": events,
    }


def report(result: dict) -> str:
    lines = [f"Matched {result['matched']}/{result['book_words']} book words "
             f"({result['coverage']:.0%})", ""]
    for e in result["events"]:
        t = f"{e['start']:7.2f}s" if "start" in e else "       -"
        match e["type"]:
            case "restart":
                lines.append(f"{t}  RESTART  cut: \"{e['text']}\"")
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
