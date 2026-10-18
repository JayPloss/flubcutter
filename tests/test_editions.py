import json

import numpy as np

from flubcutter import edit
from flubcutter.library import Library

R = edit.RATE


def make_chapter(tmp_path):
    """"one two" ... a chat with the kids ... "three four"."""
    lib = Library(tmp_path)
    book = lib.create_book("Test")
    ch = lib.add_chapter(book["id"], "One", "One two.\n\nThree four.")
    layout = [("one", 1.0, 1.4), ("two.", 1.6, 2.0),
              ("what", 3.0, 3.3), ("is", 3.4, 3.6), ("that", 3.7, 4.0), ("a", 4.4, 4.5), ("rabbit", 4.6, 5.0),
              ("three", 6.0, 6.4), ("four.", 6.6, 7.0)]
    audio = (np.random.default_rng(0).standard_normal(8 * R) * 0.003).astype(np.float32)
    for _, s, e in layout:
        t = np.arange(int(s * R), int(e * R)) / R
        audio[int(s * R):int(e * R)] += 0.3 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    rec = lib.book_dir(book["id"]) / "audio" / "rec.wav"
    edit.write_wav(audio, rec)
    part = lib.add_part(book["id"], ch["id"], "audio/rec.wav")
    words = [{"word": w, "start": s, "end": e, "prob": 1.0} for w, s, e in layout]
    path = lib.transcript_path(book["id"], ch["id"], part["id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"audio": str(rec), "duration": 8.0, "words": words}))
    a = lib.run_align(book["id"], ch["id"])
    (chat,) = [e for e in a["events"] if e["type"] == "extra"]
    return lib, book["id"], ch["id"], chat["id"]


def seconds(lib, b, c, edition):
    return len(edit.load(lib.output_path(b, c, edition))) / R


def test_an_aside_is_in_extended_only(tmp_path):
    lib, b, c, chat = make_chapter(tmp_path)
    lib.add_ai_verdicts(b, c, {chat: {"kind": "aside", "verdict": "keep", "confidence": 0.9,
                                      "reason": "A question from the kids."}})
    lib.run_render(b, c, check=None)
    ext, plain = lib.render_info(b, c, "extended"), lib.render_info(b, c, "plain")
    assert ext["asides"] == [chat] and plain["asides"] == []
    assert seconds(lib, b, c, "extended") > seconds(lib, b, c, "plain") + 1.5
    st = lib.status(b, lib.chapter(lib.book(b), c))
    assert st["asides"] == 1 and st["plain_seconds"] < st["output_seconds"]


def test_an_aside_cut_on_purpose_is_in_neither(tmp_path):
    lib, b, c, chat = make_chapter(tmp_path)
    lib.add_ai_verdicts(b, c, {chat: {"kind": "aside", "verdict": "keep", "confidence": 0.9, "reason": ""}})
    lib.set_decisions(b, c, {chat: "cut"})
    lib.run_render(b, c, check=None)
    assert seconds(lib, b, c, "extended") == seconds(lib, b, c, "plain")


def test_marking_a_flub_as_an_aside_keeps_it_in_extended(tmp_path):
    lib, b, c, chat = make_chapter(tmp_path)
    lib.add_ai_verdicts(b, c, {chat: {"kind": "flub", "verdict": "cut", "confidence": 0.6, "reason": ""}})
    review = lib.review(b, c)
    event = next(e for e in lib.alignment(b, c)["events"] if e["id"] == chat)
    assert lib.suggestion(event, review)["decision"] == "cut"
    lib.set_kinds(b, c, {chat: "aside"})
    review = lib.review(b, c)
    assert lib.kind(event, review) == "aside"
    assert lib.suggestion(event, review)["decision"] == "keep"
    lib.run_render(b, c, check=None)
    assert lib.render_info(b, c, "extended")["asides"] == [chat]
    lib.set_kinds(b, c, {chat: None})  # back to the AI's call
    assert lib.kind(event, lib.review(b, c)) == "flub"


def test_an_aside_is_kept_as_recorded():
    # Inside a chat a 2.5 s "pause" may be a child talking too quietly to be
    # transcribed: it must not be shortened, nor flagged, nor set the rhythm.
    from test_edit import W, recording
    audio = recording([(1.0, 1.4), (2.0, 2.4), (4.9, 5.3), (6.0, 6.4)])
    words = [W("end.", 1.0, 1.4), W("what's", 2.0, 2.4), W("that?", 4.9, 5.3), W("Then", 6.0, 6.4)]
    out, rep = edit.render([audio], words, {0, 1, 2, 3}, intact={1})
    assert [j["words"] for j in rep["joins"]] == []
    assert rep["pacing"] == []
    _, trimmed = edit.render([audio], words, {0, 1, 2, 3})
    assert [j["words"] for j in trimmed["joins"]] == [["what's", "that?"]]


def test_chapter_status_from_unsure_to_ready_to_out_of_date(tmp_path):
    import os, time
    lib, b, c, chat = make_chapter(tmp_path)
    st = lambda: lib.status(b, lib.chapter(lib.book(b), c))
    lib.add_ai_verdicts(b, c, {chat: {"kind": "aside", "verdict": "keep", "confidence": 0.5, "reason": ""}})
    assert st()["stage"] == "needs you" and st()["needs_you"] == 1
    lib.add_ai_verdicts(b, c, {chat: {"kind": "aside", "verdict": "keep", "confidence": 0.9, "reason": ""}})
    assert st()["stage"] == "not rendered" and st()["needs_you"] == 0
    lib.run_render(b, c, check=None)
    assert st()["stage"] == "ready"
    time.sleep(0.01)
    lib.set_decisions(b, c, {chat: "cut"})
    assert st()["stage"] == "out of date"


def test_a_bad_cut_is_rebuilt_more_carefully_and_logged(tmp_path):
    lib, b, c, chat = make_chapter(tmp_path)
    lib.set_decisions(b, c, {chat: "cut"})
    lib.run_render(b, c, check=None)
    (j,) = [j for j in lib.render_info(b, c, "extended")["joins"] if j["kind"] == "cut"]
    lib.rate_join(b, c, j["key"], "good")
    assert lib.joins_state(b, c)["levels"] == {}
    lib.rate_join(b, c, j["key"], "bad")
    st = lib.joins_state(b, c)
    assert st["levels"] == {j["key"]: 1}
    assert st["verdicts"][j["key"]]["verdict"] == "bad" and len(st["verdict_log"]) == 2
    lib.run_render(b, c, check=None)
    (j2,) = [x for x in lib.render_info(b, c, "extended")["joins"] if x["key"] == j["key"]]
    assert j2["level"] == 1  # comes up again: its verdict was for level 0
