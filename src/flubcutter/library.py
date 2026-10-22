"""The library: books -> chapters -> recording parts, stored on disk.

    <library>/<book-id>/
        book.json                   title, chapter order, parts per chapter
        text/<chapter-id>.txt       the chapter's book text
        audio/                      raw recordings (any format ffmpeg reads)
        work/<chapter-id>/          transcripts, alignment, review, peaks
        output/plain/<chapter-id>.wav     the cleaned chapter: just the story
        output/extended/<chapter-id>.wav  ... with the asides that were kept
"""

import json
import os
import re
import shutil
import threading
import time
from pathlib import Path

from flubcutter import align, edit, enhance, judge
from flubcutter.check import check_joins

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".webm", ".caf", ".aiff", ".aif", ".mp4"}
BROWSER_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".webm", ".mp4"}
CANDIDATES = ("restart", "filler", "extra", "noise")
JUDGED = ("extra", "noise")  # candidates the AI judge decides
# A pause between two read words with this much sound in it, but no words
# (a yawn, a cough, a laugh, a chair), is a candidate of its own: "noise".
NOISE_MIN = 0.8
# Below this the AI's call is a guess, so the edit waits for a person;
# above it the suggestion is accepted unless someone says otherwise.
SURE = 0.8

# Every chapter is rendered twice. Plain: just the story, every flub and every
# aside (unscripted talk: a conversation with the kids, an explanation, a
# joke) taken out. Extended: the story with the asides that were kept.
EDITIONS = ("extended", "plain")
KINDS = ("flub", "aside")

_lock = threading.RLock()
_render_lock = threading.Lock()
GLITCH_WINDOW = 3.0  # seconds before the press that a glitch report covers


def default_root() -> Path:
    return Path(os.environ.get("FLUBCUTTER_LIBRARY", Path.home() / "Flubcutter"))


def slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:48] or "untitled"


def _read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


class Library:
    def __init__(self, root: Path | None = None):
        self.root = Path(root or default_root()).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    # ---------- books ----------

    def book_dir(self, book_id: str) -> Path:
        d = (self.root / book_id).resolve()
        if d.parent != self.root:
            raise ValueError("bad book id")
        return d

    def book_ids(self) -> list[str]:
        books = [d.name for d in self.root.iterdir() if (d / "book.json").exists()]
        return sorted(books, key=lambda b: (self.book(b).get("order", 0), b))

    def book(self, book_id: str) -> dict:
        data = _read_json(self.book_dir(book_id) / "book.json")
        if data is None:
            raise KeyError(book_id)
        return data

    def save_book(self, book: dict) -> None:
        with _lock:
            _write_json(self.book_dir(book["id"]) / "book.json", book)

    def create_book(self, title: str, author: str = "") -> dict:
        with _lock:
            base = slugify(title)
            book_id, n = base, 2
            while (self.root / book_id).exists():
                book_id, n = f"{base}-{n}", n + 1
            d = self.root / book_id
            for sub in ("text", "audio", "work", "output"):
                (d / sub).mkdir(parents=True, exist_ok=True)
            book = {"id": book_id, "title": title, "author": author,
                    "order": len(self.book_ids()), "created": time.time(), "chapters": []}
            self.save_book(book)
            return book

    def delete_book(self, book_id: str) -> None:
        """Remove the book's metadata and working files. Audio is moved aside, never deleted."""
        with _lock:
            d = self.book_dir(book_id)
            audio = d / "audio"
            if audio.exists() and any(audio.iterdir()):
                keep = self.root / "_removed" / f"{book_id}-{int(time.time())}"
                keep.parent.mkdir(exist_ok=True)
                shutil.move(str(audio), str(keep))
            shutil.rmtree(d)

    def update_book(self, book_id: str, **fields) -> dict:
        with _lock:
            book = self.book(book_id)
            for k in ("title", "author"):
                if k in fields and fields[k] is not None:
                    book[k] = fields[k]
            self.save_book(book)
            return book

    # ---------- chapters ----------

    def chapter(self, book: dict, chapter_id: str) -> dict:
        for ch in book["chapters"]:
            if ch["id"] == chapter_id:
                return ch
        raise KeyError(chapter_id)

    def text_path(self, book_id: str, chapter_id: str) -> Path:
        return self.book_dir(book_id) / "text" / f"{chapter_id}.txt"

    def work_dir(self, book_id: str, chapter_id: str) -> Path:
        return self.book_dir(book_id) / "work" / chapter_id

    def output_path(self, book_id: str, chapter_id: str, edition: str | None = "extended") -> Path:
        """The cleaned chapter in one edition (see EDITIONS). None: where a
        render from before editions existed went."""
        out = self.book_dir(book_id) / "output"
        return (out / edition if edition else out) / f"{chapter_id}.wav"

    def render_file(self, book_id: str, chapter_id: str, edition: str = "extended") -> Path:
        name = "render.json" if edition == "extended" else f"render.{edition}.json"
        return self.work_dir(book_id, chapter_id) / name

    def _remove_outputs(self, book_id: str, chapter_id: str) -> None:
        for edition in (*EDITIONS, None):
            self.output_path(book_id, chapter_id, edition).unlink(missing_ok=True)
        for edition in EDITIONS:
            self.render_file(book_id, chapter_id, edition).unlink(missing_ok=True)

    def add_chapter(self, book_id: str, title: str, text: str = "") -> dict:
        with _lock:
            book = self.book(book_id)
            used = {c["id"] for c in book["chapters"]}
            n = len(book["chapters"]) + 1
            while f"ch{n:02d}" in used:
                n += 1
            ch = {"id": f"ch{n:02d}", "title": title, "parts": []}
            book["chapters"].append(ch)
            self.save_book(book)
            self.text_path(book_id, ch["id"]).write_text(text)
            return ch

    def update_chapter(self, book_id: str, chapter_id: str, title=None, text=None) -> None:
        with _lock:
            book = self.book(book_id)
            ch = self.chapter(book, chapter_id)
            if title is not None:
                ch["title"] = title
                self.save_book(book)
            if text is not None:
                p = self.text_path(book_id, chapter_id)
                if not p.exists() or p.read_text() != text:
                    p.write_text(text)
                    self.invalidate(book_id, chapter_id)

    def delete_chapter(self, book_id: str, chapter_id: str) -> None:
        with _lock:
            book = self.book(book_id)
            book["chapters"] = [c for c in book["chapters"] if c["id"] != chapter_id]
            self.save_book(book)
            self.text_path(book_id, chapter_id).unlink(missing_ok=True)
            self._remove_outputs(book_id, chapter_id)
            shutil.rmtree(self.work_dir(book_id, chapter_id), ignore_errors=True)

    def move_chapter(self, book_id: str, chapter_id: str, delta: int) -> None:
        with _lock:
            book = self.book(book_id)
            chs = book["chapters"]
            i = next(k for k, c in enumerate(chs) if c["id"] == chapter_id)
            j = max(0, min(len(chs) - 1, i + delta))
            chs.insert(j, chs.pop(i))
            self.save_book(book)

    def chapter_text(self, book_id: str, chapter_id: str) -> str:
        p = self.text_path(book_id, chapter_id)
        return p.read_text() if p.exists() else ""

    # ---------- audio + parts ----------

    def audio_files(self, book_id: str) -> list[dict]:
        d = self.book_dir(book_id) / "audio"
        book = self.book(book_id)
        used = {p["audio"]: c["id"] for c in book["chapters"] for p in c["parts"]}
        files = []
        for f in sorted(d.rglob("*")) if d.exists() else []:
            if f.is_file() and f.suffix.lower() in AUDIO_EXTS:
                rel = str(f.relative_to(self.book_dir(book_id)))
                files.append({"path": rel, "name": str(f.relative_to(d)),
                              "size": f.stat().st_size, "chapter": used.get(rel)})
        return files

    def add_part(self, book_id: str, chapter_id: str, audio_rel: str) -> dict:
        with _lock:
            path = (self.book_dir(book_id) / audio_rel).resolve()
            if self.book_dir(book_id) not in path.parents or not path.is_file():
                raise ValueError("audio file not in this book's folder")
            book = self.book(book_id)
            ch = self.chapter(book, chapter_id)
            used = {p["id"] for p in ch["parts"]}
            n = 1
            while f"p{n}" in used:
                n += 1
            part = {"id": f"p{n}", "audio": audio_rel}
            ch["parts"].append(part)
            self.save_book(book)
            self.invalidate(book_id, chapter_id)
            return part

    def remove_part(self, book_id: str, chapter_id: str, part_id: str) -> None:
        with _lock:
            book = self.book(book_id)
            ch = self.chapter(book, chapter_id)
            ch["parts"] = [p for p in ch["parts"] if p["id"] != part_id]
            self.save_book(book)
            wd = self.work_dir(book_id, chapter_id)
            for f in wd.glob(f"{part_id}.*"):
                f.unlink()
            self.invalidate(book_id, chapter_id)

    def move_part(self, book_id: str, chapter_id: str, part_id: str, delta: int) -> None:
        with _lock:
            book = self.book(book_id)
            parts = self.chapter(book, chapter_id)["parts"]
            i = next(k for k, p in enumerate(parts) if p["id"] == part_id)
            j = max(0, min(len(parts) - 1, i + delta))
            parts.insert(j, parts.pop(i))
            self.save_book(book)
            self.invalidate(book_id, chapter_id)

    def part_audio(self, book_id: str, part: dict) -> Path:
        return self.book_dir(book_id) / part["audio"]

    def playable_audio(self, book_id: str, chapter_id: str, part: dict) -> Path:
        """The part's audio in a format the browser can play (cached copy if needed)."""
        import subprocess

        src = self.part_audio(book_id, part)
        if src.suffix.lower() in BROWSER_EXTS:
            return src
        out = self.work_dir(book_id, chapter_id) / f"{part['id']}.preview.wav"
        if not out.exists() or out.stat().st_mtime < src.stat().st_mtime:
            out.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src),
                            "-ac", "1", "-ar", "24000", str(out)], check=True)
        return out

    def transcript_path(self, book_id: str, chapter_id: str, part_id: str) -> Path:
        return self.work_dir(book_id, chapter_id) / f"{part_id}.transcript.json"

    def peaks_path(self, book_id: str, chapter_id: str, part_id: str) -> Path:
        return self.work_dir(book_id, chapter_id) / f"{part_id}.peaks.json"

    def invalidate(self, book_id: str, chapter_id: str) -> None:
        """Text or parts changed: the alignment (and render) are out of date."""
        wd = self.work_dir(book_id, chapter_id)
        (wd / "alignment.json").unlink(missing_ok=True)
        self._remove_outputs(book_id, chapter_id)
        # Join levels name transcript words, which may have changed; the
        # glitch log stays as history.
        st = _read_json(wd / "joins.json")
        if st and st.get("levels"):
            st["levels"] = {}
            _write_json(wd / "joins.json", st)

    # ---------- pipeline state ----------

    def transcripts(self, book_id: str, ch: dict) -> list[dict | None]:
        return [_read_json(self.transcript_path(book_id, ch["id"], p["id"])) for p in ch["parts"]]

    def chapter_words(self, book_id: str, ch: dict) -> list[dict]:
        words = []
        for k, t in enumerate(self.transcripts(book_id, ch)):
            for w in (t or {}).get("words", []):
                words.append(dict(w, part=k))
        return words

    def alignment(self, book_id: str, chapter_id: str) -> dict | None:
        return _read_json(self.work_dir(book_id, chapter_id) / "alignment.json")

    def review(self, book_id: str, chapter_id: str) -> dict:
        r = _read_json(self.work_dir(book_id, chapter_id) / "review.json", {})
        r.setdefault("decisions", {})
        r.setdefault("ai", {})
        return r

    def save_review(self, book_id: str, chapter_id: str, review: dict) -> None:
        with _lock:
            _write_json(self.work_dir(book_id, chapter_id) / "review.json", review)

    def set_decisions(self, book_id: str, chapter_id: str, changes: dict[str, str | None]) -> dict:
        with _lock:
            r = self.review(book_id, chapter_id)
            for eid, d in changes.items():
                r.get("flagged", {}).pop(eid, None)
                if d in ("cut", "keep"):
                    r["decisions"][eid] = d
                else:
                    r["decisions"].pop(eid, None)
            self.save_review(book_id, chapter_id, r)
            return r

    def add_ai_verdicts(self, book_id: str, chapter_id: str, verdicts: dict[str, dict]) -> None:
        with _lock:
            r = self.review(book_id, chapter_id)
            r["ai"].update(verdicts)
            self.save_review(book_id, chapter_id, r)

    def set_kinds(self, book_id: str, chapter_id: str, changes: dict[str, str | None]) -> dict:
        """Say what extra speech is: "aside" or "flub" (None: back to the AI's call)."""
        with _lock:
            r = self.review(book_id, chapter_id)
            kinds = r.setdefault("kinds", {})
            for eid, k in changes.items():
                if k in KINDS:
                    kinds[eid] = k
                else:
                    kinds.pop(eid, None)
            self.save_review(book_id, chapter_id, r)
            return r

    @staticmethod
    def kind(event: dict, review: dict) -> str | None:
        """"aside" for unscripted talk that was meant, "flub" for a mistake;
        None for extra speech nobody has judged yet, and for non-candidates."""
        if event["type"] in ("restart", "filler"):
            return "flub"
        if event["type"] not in JUDGED:
            return None
        return review.get("kinds", {}).get(event["id"]) or review["ai"].get(event["id"], {}).get("kind")

    @classmethod
    def suggestion(cls, event: dict, review: dict) -> dict:
        """What the tool proposes for a candidate, before any human decision.
        For an aside, "keep" means: in the Extended edition."""
        if event["type"] == "restart":
            return {"decision": "cut", "by": "rule", "reason": "Earlier take of a re-read passage."}
        if event["type"] == "filler":
            return {"decision": "cut", "by": "rule", "reason": "Filler sound."}
        ai = review["ai"].get(event["id"])
        kind = cls.kind(event, review)
        if ai and kind == ai.get("kind", kind):
            return {"decision": ai["verdict"], "by": "ai", "reason": ai.get("reason", ""),
                    "confidence": ai.get("confidence"), "kind": ai.get("kind")}
        if kind == "aside":
            return {"decision": "keep", "by": "rule", "reason": "An aside: in the Extended edition."}
        if kind == "flub":
            return {"decision": "cut", "by": "rule", "reason": "Marked as a flub."}
        if event["type"] == "noise":
            return {"decision": "cut", "by": "rule", "reason": "Sound with no words: a yawn, a cough, a noise."}
        return {"decision": "keep", "by": "default", "reason": "Not judged yet; kept to be safe."}

    @classmethod
    def needs_you(cls, event: dict, review: dict) -> bool:
        """Whether a candidate waits for a person: nobody has decided it and
        the suggestion is a guess (nothing judged it, the AI isn't sure), or a
        cut of it kept glitching. Everything else is accepted as suggested."""
        if event["type"] not in CANDIDATES or event["id"] in review["decisions"]:
            return False
        if review.get("flagged", {}).get(event["id"]):
            return True
        s = cls.suggestion(event, review)
        return s["by"] == "default" or (s["by"] == "ai" and (s.get("confidence") or 0) < SURE)

    def effective_decisions(self, alignment: dict, review: dict) -> dict[str, str]:
        out = {}
        for e in alignment["events"]:
            if e["type"] in CANDIDATES:
                out[e["id"]] = review["decisions"].get(e["id"]) or self.suggestion(e, review)["decision"]
        return out

    def edition_decisions(self, alignment: dict, review: dict, edition: str) -> dict[str, str]:
        """The decisions one edition is rendered with: Plain cuts every aside."""
        out = self.effective_decisions(alignment, review)
        if edition == "plain":
            for e in alignment["events"]:
                if self.kind(e, review) == "aside":
                    out[e["id"]] = "cut"
        return out

    def status(self, book_id: str, ch: dict) -> dict:
        text = self.chapter_text(book_id, ch["id"]).strip()
        parts = len(ch["parts"])
        transcribed = sum(t is not None for t in self.transcripts(book_id, ch))
        a = self.alignment(book_id, ch["id"])
        review = self.review(book_id, ch["id"])
        cands = [e for e in a["events"] if e["type"] in CANDIDATES] if a else []
        extras = [e for e in cands if e["type"] in JUDGED]
        reviewed = sum(e["id"] in review["decisions"] for e in cands)
        waiting = sum(self.needs_you(e, review) for e in cands)
        judged = sum(e["id"] in review["ai"] or e["id"] in review["decisions"] for e in extras)
        out = self.output_path(book_id, ch["id"])
        render = _read_json(self.render_file(book_id, ch["id"])) if out.exists() else None
        plain = self.render_info(book_id, ch["id"], "plain")
        review_file = self.work_dir(book_id, ch["id"]) / "review.json"
        rendered = out.exists() and (not review_file.exists()
                                     or out.stat().st_mtime >= review_file.stat().st_mtime)

        if not text:
            stage = "needs text"
        elif not parts:
            stage = "needs audio"
        elif transcribed < parts:
            stage = "to transcribe"
        elif not a:
            stage = "to align"
        elif judged < len(extras) and judge.available():
            stage = "to judge"
        elif waiting:
            stage = "needs you"
        elif not out.exists():
            stage = "not rendered"
        elif not rendered:
            stage = "out of date"  # decisions changed since the render
        else:
            stage = "ready"
        return {
            "stage": stage,
            "has_text": bool(text),
            "words": len(text.split()),
            "parts": parts,
            "transcribed": transcribed,
            "aligned": a is not None,
            "coverage": a["coverage"] if a else None,
            "candidates": len(cands),
            "extras": len(extras),
            "judged": judged,
            "reviewed": reviewed,
            "needs_you": waiting,
            "rendered": rendered,
            "input_seconds": render["input_seconds"] if render else None,
            "output_seconds": render["output_seconds"] if render else None,
            "plain_seconds": plain["output_seconds"] if plain else None,
            "asides": sum(self.kind(e, review) == "aside" for e in cands),
        }

    # ---------- pipeline steps ----------

    def run_transcribe(self, book_id: str, chapter_id: str, part_id: str, progress=None) -> None:
        from flubcutter.transcribe import save, transcribe

        book = self.book(book_id)
        part = next(p for p in self.chapter(book, chapter_id)["parts"] if p["id"] == part_id)
        result = transcribe(self.part_audio(book_id, part), progress=progress)
        save(result, self.transcript_path(book_id, chapter_id, part_id))
        self.invalidate(book_id, chapter_id)

    def run_align(self, book_id: str, chapter_id: str) -> dict:
        book = self.book(book_id)
        ch = self.chapter(book, chapter_id)
        words = self.chapter_words(book_id, ch)
        result = align.analyze(words, self.chapter_text(book_id, chapter_id))
        result["parts"] = [p["id"] for p in ch["parts"]]
        self.find_noises(book_id, ch, result)
        align.save(result, self.work_dir(book_id, chapter_id) / "alignment.json")
        return result

    def find_noises(self, book_id: str, ch: dict, a: dict) -> None:
        """Turn each long pause that isn't quiet into a "noise" candidate.

        Silence is judged on the original recording, as for the cuts. The
        edges of the pause are left out: they belong to the words' sounds."""
        acs = {}
        for e in a["events"]:
            if e["type"] != "gap" or "between" not in e:
                continue
            p = e["part"]
            if p not in acs:
                acs[p] = edit.Acoustics(edit.load(self.part_audio(book_id, ch["parts"][p])))
            ac = acs[p]
            lo, hi = ac.k(e["start"] + 0.15), ac.k(e["end"] - 0.15)
            if hi <= lo:
                continue
            sound = float((ac.db[lo:hi] > ac.floor + edit.QUIET_DB).sum()) * edit.HOP
            if sound >= NOISE_MIN:
                e.update(type="noise", sound=round(sound, 2), words=[], text="(sound, no words)",
                         id=f"noise-{p}-{e['start']:.2f}-{e['book_index']}")
        a["noise_checked"] = True

    def run_render(self, book_id: str, chapter_id: str, check: str | set | None = "all",
                   progress=None) -> Path:
        """Render both editions of the chapter, then listen back to the joins
        (`check`: "all", a set of join keys, or None) and rebuild any that come
        out wrong. Returns the Extended edition's path."""
        with _render_lock:
            book = self.book(book_id)
            ch = self.chapter(book, chapter_id)
            a = self.alignment(book_id, chapter_id)
            if a is None:
                raise RuntimeError("chapter is not aligned yet")
            if "breaks" not in a:  # aligned before sentence/paragraph ends were kept
                a = self.run_align(book_id, chapter_id)
            breaks = {int(i): c for i, c in a["breaks"].items()}
            subs = {e["words"][0]: e["book"] for e in a["events"] if e["type"] == "sub"}
            words = self.chapter_words(book_id, ch)
            review = self.review(book_id, chapter_id)
            transcripts = self.transcripts(book_id, ch)
            originals = [edit.load(self.part_audio(book_id, p)) for p in ch["parts"]]
            durations = [len(x) / edit.RATE for x in originals]
            audios = []
            for k, p in enumerate(ch["parts"]):
                step = (lambda f, k=k: progress(0.05 * (k + f) / len(ch["parts"]), "enhancing audio")) \
                    if progress else None
                audios.append(self.master_audio(book_id, p, step))
            self.output_path(book_id, chapter_id, None).unlink(missing_ok=True)  # pre-editions render

            # Joins the check had to rebuild last time start where they ended up.
            # Levels belong to joins, not editions: a join in both is built alike.
            user = self.joins_state(book_id, chapter_id)["levels"]
            auto = dict(_read_json(self.render_file(book_id, chapter_id), {}).get("auto_levels", {}))
            levels = lambda: {k: max(user.get(k, 0), auto.get(k, 0)) for k in {*user, *auto}}
            checked: dict[str, dict] = {}
            keys, kept = set(), {}
            for n, edition in enumerate(EDITIONS):
                span = lambda f, step, n=n: progress((0.05 + 0.95 * (n + f) / len(EDITIONS)),
                                                     f"{step} ({edition})") if progress else None
                decisions = self.edition_decisions(a, review, edition)
                keep = edit.keep_set(a, decisions)
                # Sound with no words: cut, the pause is rebuilt; kept, it
                # stays exactly as recorded.
                noises = [e for e in a["events"] if e["type"] == "noise"]
                cut_after = {e["between"][0] for e in noises if decisions.get(e["id"]) == "cut"}
                # An aside is kept as it was recorded, and set off from the
                # reading like a paragraph.
                asides = [e for e in a["events"] if self.kind(e, review) == "aside" and e["words"]
                          and set(e["words"]) <= keep]
                intact = {i for e in asides for i in e["words"][:-1]}
                intact |= {e["between"][0] for e in noises if decisions.get(e["id"]) != "cut"}
                marks = dict(breaks)
                for e in asides:
                    marks[e["words"][0] - 1] = marks[e["words"][-1]] = "¶"
                out = self.output_path(book_id, chapter_id, edition)
                render_file = self.render_file(book_id, chapter_id, edition)
                same = next((e for e, k in kept.items() if k == (keep, cut_after, intact)), None)
                kept[edition] = (keep, cut_after, intact)
                if same:  # no asides kept: the editions are one and the same
                    out.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(self.output_path(book_id, chapter_id, same), out)
                    shutil.copyfile(self.render_file(book_id, chapter_id, same), render_file)
                    continue
                only = None if check == "all" else set(check or ())
                for attempt in range(edit.MAX_LEVEL + 1):
                    span(0.4 * attempt / (edit.MAX_LEVEL + 1), "rendering")
                    samples, report = edit.render(audios, words, keep, levels(), marks, originals, intact, cut_after)
                    samples = enhance.loudness(samples)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    edit.write_wav(samples, out)
                    if check is None:
                        break
                    if attempt == 0 and checked:
                        # A join the other edition listened to is the same join.
                        new = {j["key"] for j in report["joins"]} - checked.keys()
                        only = new if only is None else new & only
                    if only is not None and not only:
                        break
                    cur = levels()
                    step = (lambda f: span(0.4 + 0.6 * f, "listening back to the cuts")) if progress else None
                    results = check_joins(out, report["joins"], only, step, subs)
                    for k, r in results.items():
                        if k in checked and checked[k]["ok"] is False:
                            r["was"] = checked[k].get("was", checked[k]["heard"])
                        checked[k] = r
                    retry = {k for k, r in results.items()
                             if r["ok"] is False and cur.get(k, 0) < edit.MAX_LEVEL}
                    if not retry:
                        break
                    for k in retry:
                        auto[k] = cur.get(k, 0) + 1
                    only = retry
                keys |= {j["key"] for j in report["joins"]}
                for j in report["joins"]:
                    if j["key"] in checked:
                        j["check"] = checked[j["key"]]
                _write_json(render_file, {
                    **report, "edition": edition, "input_seconds": round(sum(durations), 2),
                    "output_seconds": round(len(samples) / edit.RATE, 2),
                    "rendered": time.time(), "transcripts": [t is not None for t in transcripts],
                    "asides": [e["id"] for e in asides],
                })
            # Glitch fixes and listen-back levels live with the Extended render.
            ext = self.render_file(book_id, chapter_id)
            _write_json(ext, {**_read_json(ext, {}),
                              "auto_levels": {k: v for k, v in auto.items() if k in keys}})
            return self.output_path(book_id, chapter_id)

    def master_audio(self, book_id: str, part: dict, progress=None):
        """The part's recording, cleaned up (see enhance), made once and cached."""
        return enhance.master(self.part_audio(book_id, part), self.book_dir(book_id) / "work" / "masters",
                              progress)

    def render_info(self, book_id: str, chapter_id: str, edition: str = "extended") -> dict | None:
        if not self.output_path(book_id, chapter_id, edition).exists():
            return None
        return _read_json(self.render_file(book_id, chapter_id, edition))

    def joins_state(self, book_id: str, chapter_id: str) -> dict:
        """Per-join caution levels set by glitch reports, and the report log.
        Kept apart from review.json: a glitch report isn't a review decision."""
        st = _read_json(self.work_dir(book_id, chapter_id) / "joins.json", {})
        st.setdefault("levels", {})
        st.setdefault("glitches", [])
        st.setdefault("verdicts", {})  # join key -> the listener's latest call on it
        st.setdefault("verdict_log", [])
        return st

    def rate_join(self, book_id: str, chapter_id: str, key: str, verdict: str,
                  edition: str = "extended", passive: bool = False) -> dict:
        """The listener's call on one cut, heard in the listen-through:
        "good" (said, or `passive`: heard through without objecting), or "bad", which also rebuilds that cut more cautiously (the
        caller renders). Every call is logged with the cut as it was built,
        so the defaults can be tuned from them; a clip of each bad one is
        kept with the glitch reports."""
        if verdict not in ("good", "bad"):
            raise ValueError("verdict is good or bad")
        info = self.render_info(book_id, chapter_id, edition) or {}
        join = next((j for j in info.get("joins", []) if j["key"] == key), None)
        if join is None:
            raise RuntimeError("no such cut in the current render")
        out = {"key": key, "verdict": verdict}
        if verdict == "bad":
            out.update(self.report_glitch(book_id, chapter_id, join["end"], edition=edition,
                                          keys=[key], render=False))
        with _lock:
            st = self.joins_state(book_id, chapter_id)
            # passive: heard right through without an objection.
            entry = {"verdict": verdict, "level": join.get("level", 0), "edition": edition,
                     "passive": passive, "at": time.time()}
            st["verdicts"][key] = entry
            st["verdict_log"].append(dict(entry, join=join))
            _write_json(self.work_dir(book_id, chapter_id) / "joins.json", st)
        return out

    def report_glitch(self, book_id: str, chapter_id: str, t: float,
                      window: float = GLITCH_WINDOW, edition: str = "extended",
                      keys: list[str] | None = None, render: bool = True) -> dict:
        """The listener heard a glitch at output time t (or up to `window`
        seconds before it). Rebuild the joins in that stretch more cautiously
        and re-render. A join that is already as cautious as it gets probably
        has its cut in the wrong place, so its flubs go back up for review.

        Each report is logged with a copy of the audio around it
        (work/<chapter>/glitches/), so reports can be studied later to improve
        the defaults.
        """
        info = self.render_info(book_id, chapter_id, edition)
        if not info:
            raise RuntimeError("chapter is not rendered yet")
        lo, hi = t - window, t + 0.3
        if keys is not None:  # the listener said exactly which cut
            hit = [j for j in info.get("joins", []) if j["key"] in keys]
        else:
            hit = [j for j in info.get("joins", []) if j["end"] >= lo and j["start"] <= hi]
        # Nearest first: the glitch is most likely just before the press.
        hit.sort(key=lambda j: abs((j["start"] + j["end"]) / 2 - (t - 0.6)))
        wd = self.work_dir(book_id, chapter_id)
        gdir = wd / "glitches"
        gdir.mkdir(parents=True, exist_ok=True)
        n = len(list(gdir.glob("*.wav"))) + 1
        clip = edit.load(self.output_path(book_id, chapter_id, edition))
        a, b = max(0, int((t - 5) * edit.RATE)), int((t + 1) * edit.RATE)
        edit.write_wav(clip[a:b], gdir / f"{n:03d}.wav")

        raised, maxed, reopened = [], [], []
        with _lock:
            st = self.joins_state(book_id, chapter_id)
            for j in hit:
                level = max(st["levels"].get(j["key"], 0), j.get("level", 0))
                if level < edit.MAX_LEVEL:
                    st["levels"][j["key"]] = level + 1
                    raised.append(j["key"])
                else:
                    maxed.append(j["key"])
            if maxed:
                review = self.review(book_id, chapter_id)
                for j in hit:
                    if j["key"] in maxed:
                        reopened += self._reopen_between(book_id, chapter_id, j, review)
                if reopened:
                    self.save_review(book_id, chapter_id, review)
            st["glitches"].append({
                "n": n, "clip": f"glitches/{n:03d}.wav", "time": round(t, 2), "edition": edition,
                "reported": time.time(), "reopened": reopened,
                "joins": [dict(j, new_level=st["levels"].get(j["key"], 0)) for j in hit],
            })
            _write_json(wd / "joins.json", st)
        if raised and render:
            # No listen-back check here: the listener is about to hear the fix
            # themselves, and it should play straight away.
            self.run_render(book_id, chapter_id, check=None)
        info = self.render_info(book_id, chapter_id, edition) or {}
        fixed = [j for j in info.get("joins", []) if j["key"] in raised]
        return {"joins": hit, "fixed": fixed, "maxed": maxed, "reopened": reopened,
                "render": info,
                "replay": max(0.0, min((j["start"] for j in fixed), default=t) - 2.5)}

    def _reopen_between(self, book_id: str, chapter_id: str, join: dict, review: dict) -> list[str]:
        """Put the flubs that were cut at this join back up for review."""
        a = self.alignment(book_id, chapter_id) or {"events": []}
        A, B = join["source"]
        out = []
        for e in a["events"]:
            if e["type"] not in CANDIDATES:
                continue
            if A["part"] == B["part"]:
                inside = e["part"] == A["part"] and A["t"] - 0.5 <= e["start"] and e["end"] <= B["t"] + 0.5
            else:  # cut at the end of one recording or the start of the next
                inside = ((e["part"] == A["part"] and e["start"] >= A["t"] - 0.5)
                          or (e["part"] == B["part"] and e["end"] <= B["t"] + 0.5))
            if inside:
                review["decisions"].pop(e["id"], None)
                review.setdefault("flagged", {})[e["id"]] = "A glitch was reported at this cut three times."
                out.append(e["id"])
        return out

    def peaks(self, book_id: str, chapter_id: str, part_id: str) -> dict:
        """Waveform peaks (100 per second), cached next to the transcript."""
        book = self.book(book_id)
        part = next(x for x in self.chapter(book, chapter_id)["parts"] if x["id"] == part_id)
        return _peaks(self.part_audio(book_id, part), self.peaks_path(book_id, chapter_id, part_id))

    def output_peaks(self, book_id: str, chapter_id: str, edition: str = "extended") -> dict:
        name = "output.peaks.json" if edition == "extended" else f"output.{edition}.peaks.json"
        return _peaks(self.output_path(book_id, chapter_id, edition), self.work_dir(book_id, chapter_id) / name)


def _peaks(src: Path, cache: Path) -> dict:
    cached = _read_json(cache)
    if cached and cached.get("mtime") == src.stat().st_mtime:
        return cached
    import numpy as np

    rate, per = 8000, 80
    x = edit.load(src, rate=rate)
    n = len(x) // per
    peaks = np.abs(x[: n * per]).reshape(n, per).max(axis=1) if n else np.zeros(0)
    top = float(peaks.max()) if len(peaks) else 1.0
    data = {"mtime": src.stat().st_mtime, "per_second": rate // per,
            "duration": len(x) / rate, "max": round(top, 4),
            "peaks": [round(float(v), 3) for v in peaks]}
    _write_json(cache, data)
    return data
