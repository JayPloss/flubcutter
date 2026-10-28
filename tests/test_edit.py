import numpy as np

from flubcutter import check, edit

R = edit.RATE


def tone(seconds, amp=0.3, freq=220.0):
    t = np.arange(int(seconds * R)) / R
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def hiss(seconds, amp=0.003, seed=0):
    return (np.random.default_rng(seed).standard_normal(int(seconds * R)) * amp).astype(np.float32)


def recording(layout):
    """Room noise with 'words' (tones) at the given (start, end) times."""
    total = max(e for _, e in layout) + 1.0
    audio = hiss(total)
    for s, e in layout:
        a, b = int(s * R), int(e * R)
        audio[a:b] += tone((b - a) / R + 0.01)[: b - a]
    return audio


def W(word, start, end, part=0):
    return {"word": word, "start": start, "end": end, "part": part}


def test_runs_split_at_cuts_and_parts():
    words = [W("a", 1.0, 1.4), W("b", 1.5, 1.9), W("x", 2.0, 2.4), W("c", 2.5, 2.9),
             W("d", 5.0, 5.4), W("e", 0.5, 0.9, part=1)]
    assert edit.runs(words, {0, 1, 3, 4, 5}) == [(0, 1), (3, 4), (5, 5)]


def test_word_edges_come_from_the_audio_not_the_timestamps():
    audio = recording([(1.0, 1.5), (2.2, 2.6)])
    ac = edit.Acoustics(audio)
    # Timestamps are sloppy: the first word "ends" when the next one starts.
    first, second = W("one", 1.04, 2.24), W("two", 2.24, 2.7)
    end = edit.out_point(ac, first, second["start"], 0)
    start = edit.in_point(ac, second, first["start"], 0)
    assert not end["hard"] and not start["hard"]
    assert abs(end["t"] - 1.5) < 0.03
    assert abs(start["t"] - 2.2) < 0.03
    assert end["avail"] > 0.6 and start["avail"] > 0.6


def test_short_gap_inside_a_word_does_not_end_it():
    # "conversation" + 50 ms closure + "s": the word ends after the s.
    audio = recording([(1.0, 1.5), (1.55, 1.62)])
    ac = edit.Acoustics(audio)
    end = edit.out_point(ac, W("conversations", 1.0, 3.0), ac.duration, 0)
    assert abs(end["t"] - 1.62) < 0.03


def test_no_silence_means_a_tight_cut_at_the_quietest_point():
    audio = recording([(1.0, 2.0)])
    audio[int(1.48 * R):int(1.5 * R)] *= 0.05  # a dip between two run-on words
    ac = edit.Acoustics(audio)
    end = edit.out_point(ac, W("get", 1.0, 1.5), 1.5, 0)
    assert end["hard"] and abs(end["t"] - 1.49) < 0.02


def test_render_cuts_flub_and_paces_the_join_like_the_reader():
    # "a," [flub] "b" ... the flub is cut and the pause is the comma pause.
    audio = recording([(1.0, 1.4), (2.0, 2.5), (3.2, 3.6)])
    words = [W("a,", 1.0, 1.4), W("flub", 2.0, 2.5), W("b", 3.2, 3.6)]
    out, rep = edit.render([audio], words, {0, 2})
    (j,) = rep["joins"]
    assert j["kind"] == "cut" and not j["tight"]
    usual = rep["rhythm"][","]["typ"]
    assert abs((j["end"] - j["start"]) - usual) < 0.05
    # The output has exactly two "words" of sound.
    ac = edit.Acoustics(out)
    loud = ac.db > ac.floor + 15
    assert np.count_nonzero(np.diff(loud.astype(int)) == 1) == 2
    assert abs(len(out) / R - (edit.LEAD_IN + 0.4 + usual + 0.4 + edit.END_TAIL)) < 0.1
    assert rep["pacing"] == []


def test_dead_air_breathes_at_a_sentence_end_but_not_mid_sentence():
    # 0.6 s silence, a flub, 0.7 s silence: the same cut, three contexts.
    audio = recording([(1.0, 1.4), (2.0, 2.5), (3.2, 3.6)])
    pause = {}
    for name, w, brk in [("mid", "a", {}), ("stop", "a.", {}), ("para", "a", {0: "¶"})]:
        words = [W(w, 1.0, 1.4), W("flub", 2.0, 2.5), W("b", 3.2, 3.6)]
        _, rep = edit.render([audio], words, {0, 2}, breaks=brk)
        (j,) = rep["joins"]
        pause[name], rh = j["end"] - j["start"], rep["rhythm"]
        assert rep["pacing"] == []
    assert abs(pause["mid"] - rh["word"]["typ"]) < 0.05
    assert abs(pause["stop"] - rh["."]["long"]) < 0.05
    # Only ~1.25 s of silence to keep, short of the long end at a paragraph.
    assert rh["¶"]["typ"] < pause["para"] <= rh["¶"]["long"]
    assert pause["mid"] < pause["stop"] < pause["para"]


def test_a_run_on_sentence_end_gets_the_usual_pause():
    # "a." ran straight into the flub: there is no dead air to keep.
    audio = recording([(1.0, 2.5), (3.2, 3.6)])
    words = [W("a.", 1.0, 1.4), W("flub", 1.4, 2.5), W("b", 3.2, 3.6)]
    _, rep = edit.render([audio], words, {0, 2})
    (j,) = rep["joins"]
    assert j["tight"] and j["break"] == "."


def test_paragraph_pause_of_the_readers_own_is_left_alone():
    # 1.3 s after a paragraph is reading; after a sentence it is too long.
    audio = recording([(1.0, 1.4), (2.7, 3.1)])
    words = [W("a.", 1.0, 1.4), W("b", 2.7, 3.1)]
    _, para = edit.render([audio], words, {0, 1}, breaks={0: "¶"})
    assert para["joins"] == [] and para["pacing"] == []
    _, stop = edit.render([audio], words, {0, 1})
    assert stop["joins"][0]["kind"] == "pause"


def test_paragraph_rhythm_is_never_shorter_than_sentence_rhythm():
    rh = edit.rhythm_of({".": [0.5] * 20 + [0.9] * 5})
    assert rh["¶"]["typ"] >= rh["."]["long"] and rh["¶"]["max"] >= rh["."]["max"]


def test_book_breaks_mark_sentence_and_paragraph_ends():
    from flubcutter import align
    text = "Oh dear! Oh dear! I shall be late.\n\nSo she ran, \"quick\" -- after it.\n"
    assert len(align.book_breaks(text)) == len(align.normalize(text))
    marks = [(w, c) for w, c in zip(align.normalize(text), align.book_breaks(text)) if c]
    assert marks == [("dear", "."), ("dear", "."), ("late", "¶"), ("ran", ","),
                     ("quick", "("), ("it", "¶")]


def test_book_breaks_mark_both_edges_of_an_aside():
    from flubcutter import align
    text = ("So she was considering in her own mind (as well as she could, for the "
            "day made her feel very stupid), whether it was worth it. I think\u2014\" "
            "(for, you see, she had learnt it) \"\u2014yes, that\u2019s right\u2014but then "
            "a rabbit-hole.")
    words, brk = align.normalize(text), align.book_breaks(text)
    assert len(words) == len(brk)
    assert [(w, c) for w, c in zip(words, brk) if c] == [
        ("mind", "("), ("could", ","), ("stupid", "("), ("it", "."), ("think", "("),
        ("for", ","), ("see", ","), ("it", "("), ("yes", ","), ("right", "("), ("hole", "¶")]


def test_a_cut_into_an_aside_gets_a_little_pause():
    # "mind" ran straight into a flub, then "(as well..." -- before, 0 s;
    # 0.7 s of silence after the flub, so some of it breathes.
    audio = recording([(1.0, 2.5), (3.2, 3.6)])
    words = [W("mind", 1.0, 1.4), W("flub", 1.4, 2.5), W("as", 3.2, 3.6)]
    _, plain = edit.render([audio], words, {0, 2})
    _, aside = edit.render([audio], words, {0, 2}, breaks={0: "("})
    rh = aside["rhythm"]
    assert aside["joins"][0]["break"] == "("
    assert plain["joins"][0]["pause"] < rh["("]["typ"] - 0.05
    assert rh["("]["typ"] - 0.05 <= aside["joins"][0]["pause"] <= rh["("]["long"] + 0.05
    assert rh[","]["typ"] < rh["("]["typ"] <= rh["."]["long"] + 0.3


def test_alignment_puts_breaks_on_the_last_take():
    from flubcutter import align
    words = [W(w, i, i + 0.5) for i, w in enumerate("the end the end. Next one".split())]
    a = align.analyze(words, "The end.\n\nNext one.")
    assert a["breaks"] == {"3": "¶", "5": "¶"}


def test_mistake_length_pause_is_shortened_to_the_readers_range():
    # Mid-sentence words, then a 3.6 s stop: that reads as a mistake.
    audio = recording([(1.0, 1.4), (5.0, 5.4)])
    words = [W("a", 1.0, 1.4), W("b", 5.0, 5.4)]
    out, rep = edit.render([audio], words, {0, 1})
    (j,) = rep["joins"]
    rh = rep["rhythm"]["word"]
    assert j["kind"] == "pause"
    assert rh["long"] - 0.05 <= j["end"] - j["start"] <= rh["max"]
    assert rep["pacing"] == []


def test_pacing_flags_dead_air_in_the_output():
    audio = recording([(1.0, 1.4), (2.6, 3.0)])  # 1.2 s mid-sentence
    spans = [(W("a", 0, 0), 1.0, 1.4), (W("b", 0, 0), 2.6, 3.0)]
    (flag,) = edit.pacing(audio, spans, edit.DEFAULT_RHYTHM)
    assert flag["words"] == ["a", "b"] and abs(flag["gap"] - 1.2) < 0.05
    ok = [(W("a.", 0, 0), 1.0, 1.4), (W("b", 0, 0), 2.6, 3.0)]  # fine after a full stop... nearly
    assert edit.pacing(audio, ok, {**edit.DEFAULT_RHYTHM, ".": {"typ": 0.6, "long": 0.9, "max": 1.3}}) == []


def test_higher_level_keeps_less_original_audio_and_fades_longer():
    audio = recording([(1.0, 1.4), (3.0, 3.4)])
    words = [W("a.", 1.0, 1.4), W("x", 2.0, 2.4), W("b", 3.0, 3.4)]
    _, rep0 = edit.render([audio], words, {0, 2})
    key = rep0["joins"][0]["key"]
    _, rep1 = edit.render([audio], words, {0, 2}, {key: 1})
    assert rep1["joins"][0]["level"] == 1
    # A rebuild never makes the pause tighter (that is what makes a cut sound
    # rushed) nor longer.
    assert rep1["joins"][0]["pause"] == rep0["joins"][0]["pause"]


def test_room_tone_is_quiet_and_the_right_length():
    audio = recording([(1.0, 1.4), (3.0, 3.4)])
    ac = edit.Acoustics(audio)
    rt = ac.room_tone(2.0, 2.5)  # longer than any one quiet stretch: looped
    assert len(rt) == int(2.5 * R)
    assert np.sqrt((rt ** 2).mean()) < 0.01


def test_alignment_keeps_the_last_take_despite_spelling():
    from flubcutter import align
    # Book: British "labelled" and "book-shelves"; the recogniser writes
    # "labeled" and "bookshelves", and two stumbles of "upon" must not
    # stand in for the real "book-shelves".
    spoken = ("cupboards and bookshelves here and there here and there upon upon here and there. "
              "It was labelled orange. It was labeled orange marmalade.").split()
    words = [W(w, i, i + 0.5) for i, w in enumerate(spoken)]
    a = align.analyze(words, "cupboards and book-shelves; here and there. "
                             "It was labelled \"ORANGE MARMALADE\".")
    kept = " ".join(spoken[i] for i in a["kept_words"])
    assert kept == "cupboards and bookshelves here and there. It was labeled orange marmalade."


def test_alignment_does_not_pick_a_stray_word_out_of_an_aside():
    from flubcutter import align
    spoken = "she fell past it. So I'm making a recording for it to test it. Well thought Alice".split()
    words = [W(w, i, i + 0.5) for i, w in enumerate(spoken)]
    a = align.analyze(words, "she fell past it.\n\n'Well!' thought Alice")
    assert [spoken[i] for i in a["kept_words"]] == "she fell past it. Well thought Alice".split()
    # ...but a real restart still keeps the last take.
    spoken = "the cat the cat sat".split()
    a = align.analyze([W(w, i, i + 0.5) for i, w in enumerate(spoken)], "The cat sat.")
    assert a["kept_words"] == [2, 3, 4]


def test_heard_ok():
    assert check.heard_ok("or conversations so she was".split(), ["or", "conversations"], ["so", "she"])
    assert not check.heard_ok("or conversation so she was".split(), ["or", "conversations"], ["so", "she"])
    assert not check.heard_ok("this and but at".split(), ["at", "this"], ["but", "at"])
    assert check.heard_ok("pictures are conversations so she".split(), ["or", "conversations"], ["so", "she"])
    assert check.heard_ok("it was labeled orange".split(), ["was", "labelled"], ["orange"])
    assert check.heard_ok("a large rabbithole under".split(), ["rabbit"], ["hole", "under"])
    # The cut-away take's first word leaks through.
    assert not check.heard_ok("he told Creature to to go home".split(), ["told", "Creature"], ["to", "go"])
    assert not check.heard_ok("the the white rabbit".split(), ["the", "the"][:1], ["white"])
    # ...unless the book says it twice.
    assert check.heard_ok("I I think".split(), ["I"], ["I", "think"])


def test_room_tone_leaves_out_the_soft_start_of_a_word():
    # The original's noise hides the soft start of a word; the denoised
    # master shows it. Room tone must not carry (and loop) that sliver.
    rng = np.random.default_rng(1)
    n = 2 * edit.RATE
    master = (rng.standard_normal(n) * 1e-4).astype(np.float32)
    t = np.arange(int(0.08 * edit.RATE)) / edit.RATE
    onset = int(1.38 * edit.RATE)
    master[onset:onset + len(t)] += (0.003 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    master[int(1.46 * edit.RATE):] += (0.3 * np.sin(2 * np.pi * 220 * np.arange(n - int(1.46 * edit.RATE)) / edit.RATE)).astype(np.float32)
    master[:int(0.5 * edit.RATE)] += (0.3 * np.sin(2 * np.pi * 220 * np.arange(int(0.5 * edit.RATE)) / edit.RATE)).astype(np.float32)
    original = master + (rng.standard_normal(n) * 0.003).astype(np.float32)
    ac = edit.Acoustics(master, original)
    tone = ac.room_tone(0.9, 2.0)  # longer than the gap: it loops
    assert np.sqrt(np.mean(tone ** 2)) < 3e-4


def test_a_tight_cut_fades_out_before_the_cut_word_starts():
    # "slowly," runs into a restarted "for" with only ~70 ms of quiet: the
    # fade of the cut must be over before "for" begins, at every level.
    audio = recording([(1.0, 1.5), (1.57, 1.9), (3.0, 3.4)])
    words = [W("slowly,", 1.0, 1.5), W("for", 1.55, 1.9), W("for", 3.0, 3.4)]
    ac = edit.Acoustics(audio)
    for level, L in enumerate(edit.LEVELS):
        p = edit.out_point(ac, words[0], words[1]["start"], level)
        if p["hard"]:
            assert p["t"] + L["hard_fade"] / 2 <= 1.57, (level, p)


def test_a_tight_cut_leaves_out_a_breath_after_the_word():
    # "slowly," then 60 ms of breath hiss (quiet overall, loud in the high
    # band) then the restarted "for": the cut ends with the voice, not the
    # breath. A word ending in a hiss of its own ("this") keeps it.
    rng = np.random.default_rng(2)
    audio = recording([(1.0, 1.5), (1.58, 1.9), (3.0, 3.4)])
    hiss = np.diff(rng.standard_normal(int(0.06 * R) + 1)).astype(np.float32) * 0.004
    audio[int(1.505 * R):int(1.505 * R) + len(hiss)] += hiss
    ac = edit.Acoustics(audio)
    if (ac.db[ac.k(1.51):ac.k(1.56)] < ac.floor + edit.QUIET_DB).all():  # the setup is as intended
        cut = edit.out_point(ac, W("slowly,", 1.0, 1.5), 1.55, 0)
        assert cut["hard"] and cut["t"] + edit.LEVELS[0]["hard_fade"] / 2 <= 1.515
        own = edit.out_point(ac, W("this", 1.0, 1.5), 1.55, 0)
        assert own["t"] > cut["t"]  # not pulled back to the end of the voice


def test_the_hiss_of_a_word_starting_with_f_is_not_a_breath():
    # "for": 80 ms of "f" hiss (quiet overall, loud in the high band) right
    # before the voice. It belongs to the word: the cut must not drop it.
    rng = np.random.default_rng(3)
    audio = recording([(1.0, 1.4), (3.08, 3.4)])
    f = np.diff(rng.standard_normal(int(0.08 * R) + 1)).astype(np.float32) * 0.01
    audio[int(3.0 * R):int(3.0 * R) + len(f)] += f
    ac = edit.Acoustics(audio)
    p = edit.in_point(ac, W("for", 3.1, 3.4), 2.0, 0)
    assert p["speech"] <= 3.01
    words = [W("slowly,", 1.0, 1.4), W("um", 2.0, 2.4), W("for", 3.1, 3.4)]
    _, rep = edit.render([audio], words, {0, 2})
    assert rep["joins"][0]["source"][1]["t"] <= 3.01


def test_a_cut_noise_takes_the_yawn_out_of_the_pause():
    # "a." then three seconds of yawn running straight on from it, then "b".
    audio = recording([(1.0, 1.4), (4.4, 4.8)])
    audio[int(1.4 * R):int(4.3 * R)] += tone(2.9, amp=0.05, freq=140.0)
    words = [W("a.", 1.0, 1.4), W("b", 4.4, 4.8)]
    kept, _ = edit.render([audio], words, {0, 1})
    out, rep = edit.render([audio], words, {0, 1}, cut_after={0})
    assert len(kept) / R - len(out) / R > 2.0
    (j,) = rep["joins"]
    assert j["kind"] == "pause" and j["pause"] < 1.2
