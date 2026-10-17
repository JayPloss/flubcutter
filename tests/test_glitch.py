import json

import numpy as np

from flubcutter import edit
from flubcutter.library import Library

R = edit.RATE


def make_chapter(tmp_path):
    """A recording of "one two. two three" with the first "two." a restart."""
    lib = Library(tmp_path)
    book = lib.create_book("Test")
    ch = lib.add_chapter(book["id"], "One", "one two three")
    layout = [("one", 1.0, 1.4), ("two.", 1.6, 2.0), ("two", 3.0, 3.4), ("three", 3.6, 4.0)]
    audio = (np.random.default_rng(0).standard_normal(5 * R) * 0.003).astype(np.float32)
    for _, s, e in layout:
        t = np.arange(int(s * R), int(e * R)) / R
        audio[int(s * R):int(e * R)] += 0.3 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    rec = lib.book_dir(book["id"]) / "audio" / "rec.wav"
    edit.write_wav(audio, rec)
    part = lib.add_part(book["id"], ch["id"], "audio/rec.wav")
    words = [{"word": w, "start": s, "end": e, "prob": 1.0} for w, s, e in layout]
    path = lib.transcript_path(book["id"], ch["id"], part["id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"audio": str(rec), "duration": 5.0, "words": words}))
    lib.run_align(book["id"], ch["id"])
    lib.run_render(book["id"], ch["id"], check=None)
    return lib, book["id"], ch["id"]


def test_glitch_reports_escalate_then_reopen_the_flub(tmp_path):
    lib, b, c = make_chapter(tmp_path)
    (join,) = lib.render_info(b, c)["joins"]
    assert join["words"] == ["one", "two"]

    r1 = lib.report_glitch(b, c, join["end"] + 1.0)
    assert [j["level"] for j in r1["fixed"]] == [1]
    assert r1["replay"] < join["start"]
    r2 = lib.report_glitch(b, c, r1["fixed"][0]["end"] + 1.0)
    assert [j["level"] for j in r2["fixed"]] == [2]
    assert lib.render_info(b, c)["joins"][0]["level"] == 2

    # Still bad at the most careful level: the restart goes back up for review.
    r3 = lib.report_glitch(b, c, r2["fixed"][0]["end"] + 1.0)
    assert r3["fixed"] == [] and r3["maxed"] == [join["key"]]
    (eid,) = r3["reopened"]
    assert eid.startswith("restart")
    assert eid in lib.review(b, c)["flagged"]

    st = lib.joins_state(b, c)
    assert len(st["glitches"]) == 3
    assert (lib.work_dir(b, c) / st["glitches"][0]["clip"]).exists()


def test_glitch_away_from_any_cut_changes_nothing(tmp_path):
    lib, b, c = make_chapter(tmp_path)
    before = lib.render_info(b, c)["rendered"]
    r = lib.report_glitch(b, c, 0.2, window=0.1)
    assert r["joins"] == [] and r["fixed"] == []
    assert lib.render_info(b, c)["rendered"] == before
    assert lib.status(b, lib.chapter(lib.book(b), c))["rendered"]
