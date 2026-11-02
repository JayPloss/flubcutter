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


def test_a_yawn_transcribed_as_yeah_in_place_of_book_words_is_extra():
    # The book says "to the left"; a yawn came out as "Yeah. Yeah." over "to the".
    book = "column two. To the left of this paper sat another"
    spoken = "column two. Yeah. Yeah. left of this paper sat another".split()
    r = analyze(words_from(spoken), book)
    assert [e["text"] for e in r["events"] if e["type"] == "extra"] == ["Yeah.", "Yeah."]
    assert 2 not in r["kept_words"] and 3 not in r["kept_words"]
    assert not [e for e in r["events"] if e["type"] == "sub"]


def test_a_stutter_beside_a_misheard_name_is_cut():
    # "Kreacher" heard as "creature", then "to" said twice.
    book = "and he told Kreacher to go home and never"
    spoken = "and he told creature to to go home and never".split()
    r = analyze(words_from(spoken), book)
    assert 3 in r["kept_words"]
    assert events_of(r, "restart", "extra") == [("restart", "to")]


def test_talk_after_the_last_line_does_not_take_over_its_words():
    book = "harry stared up at the dark roof listening to the pounding of the rain"
    said = "harry stared up at the dark roof listening to the pounding the right that's true it's a half"
    r = analyze(words_from(said.split()), book)
    kept = [said.split()[i] for i in r["kept_words"]]
    assert kept[-3:] == ["pounding", "the", "right"]
    (e,) = [e for e in r["events"] if e["type"] == "extra"]
    assert e["text"] == "that's true it's a half" and e["book_index"] == r["book_words"]


def test_talk_after_a_dropped_last_word_is_all_extra():
    book = "neither ron nor hermione answered him"
    said = "neither ron nor hermione answered end of chapter twelve perfect"
    r = analyze(words_from(said.split()), book)
    assert [said.split()[i] for i in r["kept_words"]][-1] == "answered"
    (e,) = [e for e in r["events"] if e["type"] == "extra"]
    assert e["text"] == "end of chapter twelve perfect"


def test_a_reread_last_line_still_keeps_the_last_take():
    book = "and they all went home to bed"
    said = "and they all went home to bed and they all went home to bed"
    r = analyze(words_from(said.split()), book)
    assert r["kept_words"][-1] == len(said.split()) - 1


def test_a_common_word_in_the_talk_does_not_pass_for_the_book():
    book = "she drained it from the bottom of her glass"
    said = "she drained it from the bottom end of chapter twenty five"
    w = words_from(said.split())
    for x in w[6:]:
        x["start"] += 8
        x["end"] += 8
    r = analyze(w, book)
    assert said.split()[r["kept_words"][-1]] == "bottom"


def test_the_last_word_read_after_a_pause_is_kept():
    book = "and that was the end"
    said = "and that was the end"
    w = words_from(said.split())
    w[4]["start"] += 5
    w[4]["end"] += 5
    r = analyze(w, book)
    assert r["kept_words"][-1] == 4
