"""Step 2: align the transcript against the book text and find where they differ.

Core rule: the reader re-reads a passage until satisfied, so the LAST take of
any repeated passage is the one to keep. The alignment is a standard
edit-distance alignment, but the traceback runs from the end and prefers
matching, so later takes get paired with the book and earlier takes are left
over as "extra" transcript words.

A chapter may be recorded in several parts. Words carry a "part" index and
part-local times; the alignment runs over all parts in order.
"""

import difflib
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
# What a yawn, a sigh or a cough is often transcribed as. Lined up against a
# different book word, it is not that word misread: it is extra sound (and
# the book word was likely swallowed with it), so the judge decides it.
INTERJECTIONS = FILLERS | {"yeah", "yes", "oh", "ooh", "aah", "huh", "ha", "okay", "ok", "hm", "mmm", "oof", "ugh", "whoa", "wow"}


def _interjection_for(said: str, book_word: str) -> bool:
    """Whether a spoken word lined up against a book word is an interjection
    standing in for it rather than that word misheard ("yes" for "yet")."""
    toks = normalize(said)
    return bool(toks) and all(t in INTERJECTIONS for t in toks) and book_word not in INTERJECTIONS \
        and difflib.SequenceMatcher(None, "".join(toks), book_word).ratio() < 0.5


def normalize(text: str) -> list[str]:
    """Lowercase words with punctuation stripped; hyphenated words are split."""
    text = text.lower().replace("’", "'").replace("‘", "'")
    text = re.sub(r"[-—–]", " ", text)
    return [w for w in (re.sub(r"[^a-z0-9']", "", t).strip("'") for t in text.split()) if w]


# Where the book itself pauses, after a word: a paragraph end beats a
# sentence end beats the edge of an aside beats a clause end. Used to pace
# the cleaned audio. An aside -- in brackets or between dashes, like this
# one -- is read with a little pause going in and coming out.
BREAK_RANK = {"": 0, ",": 1, "(": 2, ".": 3, "¶": 4}
_SENTENCE_END = ".!?…"
_CLAUSE_END = ",;:-"
_DASH = re.compile(r"(—|–|--+)")


def book_breaks(book_text: str) -> list[str]:
    """For each word of normalize(book_text): "¶" if it ends a paragraph, "."
    a sentence, "(" the edge of an aside (the word before it opens, or its
    last word), "," a clause, else "". Paragraphs are separated by blank
    lines (one per line if the text has no blank lines at all)."""
    sep = r"\n\s*\n" if re.search(r"\n\s*\n", book_text.strip()) else r"\n"
    out = []
    for para in re.split(sep, book_text):
        start = len(out)

        def mark(cls):  # the last word so far pauses at least this much
            if len(out) > start and BREAK_RANK[cls] > BREAK_RANK[out[-1]]:
                out[-1] = cls

        for tok in para.split():
            for piece in _DASH.split(tok):  # "think—" "—yes" "late—but"
                if _DASH.fullmatch(piece):
                    mark("(")
                    continue
                if piece.lstrip("\"'“‘").startswith(("(", "[")):
                    mark("(")
                n = len(normalize(piece))
                tail = re.search(r"[^\w]*$", piece).group()  # "stupid)," -> "),"
                cls = ("." if any(ch in _SENTENCE_END for ch in tail)
                       else "(" if any(ch in ")]" for ch in tail)
                       else "," if any(ch in _CLAUSE_END for ch in tail) else "")
                if n:
                    out.extend([""] * n)
                mark(cls)  # (punctuation standing on its own marks the word before)
        if len(out) > start:
            out[-1] = "¶"
    return out


def canon(word: str) -> str:
    """A normalized word with spelling variants folded together, for matching:
    "labelled" = "labeled", "colour" = "color", "realise" = "realize"."""
    w = re.sub(r"(.)\1+", r"\1", word)
    w = re.sub(r"our(s?)$", r"or\1", w)
    return re.sub(r"is(e|es|ed|ing)$", r"iz\1", w)


def compounds(book_text: str) -> dict[str, list[str]]:
    """Hyphenated book words, closed up: {"bookshelves": ["book", "shelves"]}.
    The recogniser may write "book-shelves" as one word or as two."""
    out = {}
    for tok in re.findall(r"[A-Za-z']+(?:-[A-Za-z']+)+", book_text):
        parts = normalize(tok)
        if len(parts) > 1:
            out[canon("".join(parts))] = parts
    return out


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


def sound(word: str) -> str:
    """A rough key for how a word sounds: "creature" and "Kreacher" are both
    "KR..". Consonants only after the first letter, similar ones folded
    together, doubles closed up."""
    w = re.sub(r"[^a-z]", "", word)
    if not w:
        return ""
    for a, b in (("ch", "x"), ("sh", "x"), ("ph", "f"), ("th", "t"), ("ck", "k"), ("q", "k"),
                 ("c", "k"), ("z", "s"), ("v", "f"), ("d", "t"), ("b", "p"), ("g", "k")):
        w = w.replace(a, b)
    w = w[0] + re.sub(r"[aeiouyhw]", "", w[1:])
    return re.sub(r"(.)\1+", r"\1", w).upper()


def _sounds_like(a: str, b: str) -> bool:
    return len(a) >= 4 and len(b) >= 4 and sound(a)[:2] == sound(b)[:2] and len(sound(a)) >= 2


def _sound_alike_swaps(ops: list[tuple[str, int | None, int | None]], book: list[str],
                       spoken: list[str]) -> list[tuple[str, int | None, int | None]]:
    """A misheard word right before a stutter: book "told Kreacher to go",
    spoken "told creature to to go". Both pairings cost the same, and the
    traceback pairs "Kreacher" with the first "to" and leaves "creature"
    over, so the stutter stays in. Where an extra word sounds like the book
    word its neighbour stands in for, the neighbour doesn't, and the
    neighbour is said again straight after, the two swap: the misheard word
    reads the book word and the stutter is extra. (Without the repeat, the
    two are more likely one word the transcriber split: "Herman only" for
    "Hermione".)"""
    for k in range(len(ops) - 1):
        (k1, b1, s1), (k2, b2, s2) = ops[k], ops[k + 1]
        if k1 == EXTRA and k2 == SUB and s2 == s1 + 1 and spoken[s2] == spoken[min(s2 + 1, len(spoken) - 1)] \
                and s2 + 1 < len(spoken) and _sounds_like(spoken[s1], book[b2]) and not _sounds_like(spoken[s2], book[b2]):
            ops[k], ops[k + 1] = (SUB, b2, s1), (EXTRA, None, s2)
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


def _rescue_strays(ops: list[tuple[str, int | None, int | None]], spoken: list[int],
                   longest: int = 3) -> list[tuple[str, int | None, int | None]]:
    """Last-take-wins can pick a common word out of the middle of extra
    speech: book "fell past it. Well", spoken "fell past it. So I'm making a
    recording for it ... Well" keeps the second "it". A short island of
    matched words with extra speech on both sides goes back to the same
    words right after the previous kept word, when those were left over.
    (A restart is not an island: its words run on into the rest of the take,
    or into a retry of the words that follow.)"""
    kind_of = {s: k for k, (kind, _, s) in enumerate(ops) if s is not None}
    matched = lambda s: s in kind_of and ops[kind_of[s]][0] in (MATCH, SUB)
    s = 0
    while s < len(spoken):
        if not matched(s) or (s > 0 and matched(s - 1)):
            s += 1
            continue
        end = s
        while end + 1 < len(spoken) and matched(end + 1):
            end += 1
        n = end - s + 1
        prev = next((p for p in range(s - 1, -1, -1) if matched(p)), None)
        nxt = next((p for p in range(end + 1, len(spoken)) if matched(p)), None)
        # A take that runs into a retry of what comes next is real reading.
        runs_on = nxt is not None and spoken[end + 1] == spoken[nxt]
        if (n <= longest and not runs_on and s > 0 and end + 1 < len(spoken) and prev is not None
                and prev + n < s and spoken[prev + 1:prev + 1 + n] == spoken[s:end + 1]
                and not any(matched(x) for x in range(prev + 1, prev + 1 + n))
                and all(ops[kind_of[x]][0] == MATCH for x in range(s, end + 1))
                and all(op[0] == EXTRA for op in ops[kind_of[prev] + 1:kind_of[s]])):
            for i in range(n):
                k_new, k_old = kind_of[prev + 1 + i], kind_of[s + i]
                ops[k_new] = (MATCH, ops[k_old][1], prev + 1 + i)
                ops[k_old] = (EXTRA, None, s + i)
        s = end + 1
    return ops


def align_words(book: list[str], spoken: list[str]) -> list[tuple[str, int | None, int | None]]:
    vocab: dict[str, int] = {}
    b = [vocab.setdefault(canon(w), len(vocab)) for w in book]
    s = [vocab.setdefault(canon(w), len(vocab)) for w in spoken]
    if (len(b) + 1) * (len(s) + 1) <= MAX_CELLS:
        return _sound_alike_swaps(_rescue_strays(_dp_align(b, s), s), book, spoken)

    ops = []
    bounds = [(0, 0)] + _anchors(b, s) + [(len(b), len(s))]
    for (i0, j0), (i1, j1) in zip(bounds, bounds[1:]):
        for kind, bi, sj in _dp_align(b[i0:i1], s[j0:j1]):
            ops.append((kind, None if bi is None else bi + i0, None if sj is None else sj + j0))
    return _sound_alike_swaps(_rescue_strays(ops, s), book, spoken)


def _overlap(extra: list[str], text: list[str]) -> float:
    if not extra:
        return 0.0
    window = set(text)
    return sum(w in window for w in extra) / len(extra)


def analyze(words: list[dict], book_text: str) -> dict:
    """words: transcript words (all parts, in order), each with a "part" key."""
    split = compounds(book_text)
    flat, owner = [], []
    for idx, w in enumerate(words):
        for p in normalize(w["word"]):
            for q in split.get(canon(p), [p]):
                flat.append(q)
                owner.append(idx)

    book = normalize(book_text)
    ops = align_words(book, flat)

    events = []
    kept = []
    # Sentence and paragraph ends and asides from the book, on the spoken word that
    # read them (or, if the book word was skipped, the last word before it).
    brk = book_breaks(book_text)
    breaks: dict[int, str] = {}
    last = None
    for kind, b, s in ops:
        if s is not None and kind != EXTRA:
            last = owner[s]
        if b is not None and brk[b] in ("(", ".", "¶") and last is not None:
            if BREAK_RANK[brk[b]] > BREAK_RANK[breaks.get(last, "")]:
                breaks[last] = brk[b]
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
        if kind == SUB and _interjection_for(words[owner[s]]["word"], book[b]):
            w = words[owner[s]]
            events.append({"type": "extra", "part": w["part"], "text": w["word"], "start": w["start"],
                           "end": w["end"], "words": [owner[s]], "book_index": b, "stands_for": book[b]})
            k += 1
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
    book_of = {owner[s]: b for _, b, s in ops if b is not None and s is not None}
    for a, b in zip(kept, kept[1:]):
        if b == a + 1 and words[a]["part"] == words[b]["part"]:
            gap = words[b]["start"] - words[a]["end"]
            if gap > LONG_GAP:
                events.append({"type": "gap", "part": words[a]["part"], "start": words[a]["end"],
                               "end": words[b]["start"], "seconds": round(gap, 2),
                               "between": [a, b], "book_index": book_of.get(b, 0)})

    events.sort(key=lambda e: (e.get("part", 1 << 30), e.get("start", 0.0)))
    for e in events:
        e["id"] = f"{e['type']}-{e.get('part', 'x')}-{e.get('start', 0.0):.2f}-{e.get('book_index', 0)}"

    matched = sum(op[0] == MATCH for op in ops)
    return {
        "book_words": len(book),
        "matched": matched,
        "coverage": round(matched / len(book), 3) if book else 0,
        "kept_words": kept,
        "breaks": {str(i): c for i, c in sorted(breaks.items())},
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
            case "noise":
                lines.append(f"{t}  NOISE    review: {e['sound']}s of sound, no words")
    return "\n".join(lines)


def save(result: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
