import random

from flubcutter import align
from flubcutter.align import analyze, normalize


def words_from(tokens, part_of=lambda k: 0, step=0.4):
    return [{"word": w, "start": k * step, "end": k * step + 0.3, "part": part_of(k)}
            for k, w in enumerate(tokens)]


def events_of(result, *types):
    return [(e["type"], e["text"]) for e in result["events"] if e["type"] in types]


def test_normalize_splits_hyphens_and_strips_punctuation():
    assert normalize("“Oh dear!” waistcoat-pocket, Alice’s") == ["oh", "dear", "waistcoat", "pocket", "alice's"]


def test_last_take_wins_on_exact_repeat():
    book = "so she was considering in her own mind"
    spoken = "so she was considering so she was considering in her own mind".split()
    r = analyze(words_from(spoken), book)
    assert events_of(r, "restart", "extra") == [("restart", "so she was considering")]
    # The kept words are the second take.
    assert r["kept_words"] == list(range(4, 12))


def test_flubbed_word_then_correction():
    book = "very tired of sitting by her sister on the bank and of having nothing to do"
    spoken = ("very tired of sitting by her sister on the bunk "
              "very tired of sitting by her sister on the bank and of having nothing to do").split()
    r = analyze(words_from(spoken), book)
    assert events_of(r, "restart", "extra", "sub") == [
        ("restart", "very tired of sitting by her sister on the bunk")]


def test_aside_is_extra_not_restart():
    book = "a watch out of its waistcoat pocket and looked at it"
    spoken = "a watch out of its waistcoat pocket what's a waistcoat it's like a vest and looked at it".split()
    r = analyze(words_from(spoken), book)
    assert events_of(r, "restart", "extra") == [("extra", "what's a waistcoat it's like a vest")]


def test_filler_is_its_own_type():
    r = analyze(words_from("the white um rabbit".split()), "the white rabbit")
    assert events_of(r, "filler", "extra") == [("filler", "um")]


def test_overlap_across_parts_keeps_later_part():
    book = "one two three four five six seven eight nine ten"
    p1 = "one two three four five six seven".split()
    p2 = "five six seven eight nine ten".split()
    words = words_from(p1, lambda k: 0) + [dict(w, part=1) for w in words_from(p2)]
    r = analyze(words, book)
    restarts = [e for e in r["events"] if e["type"] == "restart"]
    assert [(e["part"], e["text"]) for e in restarts] == [(0, "five six seven")]


def test_long_chapter_uses_anchors_and_finds_every_flub(monkeypatch):
    random.seed(1)
    vocab = [f"w{k}" for k in range(3000)]
    book = [random.choice(vocab) for _ in range(9000)]
    spoken, truth, i = [], [], 0
    while i < len(book):
        if random.random() < 0.004 and i > 10:
            take = book[i:i + random.randint(3, 8)] + ["flubword"]
            spoken += take
            truth.append(("restart", " ".join(take)))
        if random.random() < 0.002:
            ad = "hey kids what do you think happens next".split()
            spoken += ad
            truth.append(("extra", " ".join(ad)))
        spoken.append(book[i])
        i += 1

    calls = []
    real = align._dp_align
    monkeypatch.setattr(align, "_dp_align", lambda b, s: calls.append(1) or real(b, s))
    half = len(spoken) // 2
    r = analyze(words_from(spoken, lambda k: 0 if k < half else 1), " ".join(book))
    assert len(calls) > 1, "expected the alignment to be split at anchors"
    assert r["coverage"] == 1.0
    assert sorted(events_of(r, "restart", "extra")) == sorted(truth)
