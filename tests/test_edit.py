import numpy as np

from flubcutter import edit


def test_plan_drops_cut_words_and_caps_pauses():
    words = [
        {"word": "a", "start": 5.0, "end": 5.5, "part": 0},   # 5 s of dead air before
        {"word": "b", "start": 5.6, "end": 6.0, "part": 0},
        {"word": "x", "start": 6.0, "end": 6.4, "part": 0},   # cut
        {"word": "b", "start": 6.5, "end": 7.0, "part": 0},
        {"word": "c", "start": 10.0, "end": 10.5, "part": 0},  # 3 s pause before
    ]
    chunks = edit.plan(words, {0, 1, 3, 4}, [12.0])
    assert chunks[0]["start"] == 4.7                      # lead-in trimmed to 0.3 s
    assert chunks[0]["end"] == 6.0                        # stops before the cut word
    assert chunks[1]["start"] == 6.4 and chunks[1]["end"] == 7.6
    assert chunks[2]["start"] == 9.4                      # 3 s pause becomes 1.2 s total


def test_plan_starts_new_chunk_per_part():
    words = [
        {"word": "a", "start": 1.0, "end": 1.5, "part": 0},
        {"word": "b", "start": 0.5, "end": 1.0, "part": 1},
    ]
    chunks = edit.plan(words, {0, 1}, [2.0, 2.0])
    assert [c["part"] for c in chunks] == [0, 1]


def test_snap_moves_cut_to_silence():
    audio = np.ones(edit.RATE, dtype=np.float32) * 0.5
    quiet = int(0.52 * edit.RATE)
    audio[quiet:quiet + 480] = 0.0                       # 10 ms of silence at 0.52 s
    idx = edit.snap(audio, 0.47)
    assert abs(idx / edit.RATE - 0.525) < 0.011


def test_render_length_matches_chunks():
    a = np.zeros(edit.RATE * 4, dtype=np.float32)
    out = edit.render([a], [{"part": 0, "start": 0.0, "end": 1.0}, {"part": 0, "start": 2.0, "end": 3.0}])
    expected = 2 * edit.RATE - int(edit.CROSSFADE * edit.RATE)
    assert abs(len(out) - expected) <= 2 * int(edit.SNAP_RADIUS * edit.RATE)
