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
    # "a." [flub] "b" ... the flub is cut and the pause is the full-stop pause.
    audio = recording([(1.0, 1.4), (2.0, 2.5), (3.2, 3.6)])
    words = [W("a.", 1.0, 1.4), W("flub", 2.0, 2.5), W("b", 3.2, 3.6)]
    out, rep = edit.render([audio], words, {0, 2})
    (j,) = rep["joins"]
    assert j["kind"] == "cut" and not j["tight"]
    usual = rep["rhythm"]["."]["typ"]
    assert abs((j["end"] - j["start"]) - usual) < 0.05
    # The output has exactly two "words" of sound.
    ac = edit.Acoustics(out)
    loud = ac.db > ac.floor + 15
    assert np.count_nonzero(np.diff(loud.astype(int)) == 1) == 2
    assert abs(len(out) / R - (edit.LEAD_IN + 0.4 + usual + 0.4 + edit.END_TAIL)) < 0.1
    assert rep["pacing"] == []


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
    assert rep1["joins"][0]["pause"] < rep0["joins"][0]["pause"]  # tighter, never slower


def test_room_tone_is_quiet_and_the_right_length():
    audio = recording([(1.0, 1.4), (3.0, 3.4)])
    ac = edit.Acoustics(audio)
    rt = ac.room_tone(2.0, 2.5)  # longer than any one quiet stretch: looped
    assert len(rt) == int(2.5 * R)
    assert np.sqrt((rt ** 2).mean()) < 0.01


def test_heard_ok():
    assert check.heard_ok("or conversations so she was".split(), ["or", "conversations"], ["so", "she"])
    assert not check.heard_ok("or conversation so she was".split(), ["or", "conversations"], ["so", "she"])
    assert not check.heard_ok("this and but at".split(), ["at", "this"], ["but", "at"])
    assert check.heard_ok("pictures are conversations so she".split(), ["or", "conversations"], ["so", "she"])
