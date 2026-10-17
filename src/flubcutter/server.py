"""Local web app: progress dashboard, book setup, and chapter review."""

import shutil
from pathlib import Path

from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from flubcutter import importers, judge
from flubcutter.align import book_context
from flubcutter.jobs import Jobs
from flubcutter.library import AUDIO_EXTS, CANDIDATES, Library

STATIC = Path(__file__).parent / "static"


def create_app(lib: Library) -> FastAPI:
    app = FastAPI(title="flubcutter")
    jobs = Jobs(lib)

    def book_or_404(b):
        try:
            return lib.book(b)
        except (KeyError, ValueError):
            raise HTTPException(404, "no such book")

    def chapter_or_404(b, c):
        book = book_or_404(b)
        try:
            return book, lib.chapter(book, c)
        except KeyError:
            raise HTTPException(404, "no such chapter")

    def part_or_404(b, c, p):
        book, ch = chapter_or_404(b, c)
        for part in ch["parts"]:
            if part["id"] == p:
                return book, ch, part
        raise HTTPException(404, "no such part")

    # ---------- library + dashboard ----------

    @app.get("/api/library")
    def library():
        books = []
        for b in lib.book_ids():
            book = lib.book(b)
            chapters = []
            for ch in book["chapters"]:
                st = lib.status(b, ch)
                st["busy"] = jobs.busy(b, ch["id"])
                chapters.append({"id": ch["id"], "title": ch["title"], "status": st})
            books.append({"id": b, "title": book["title"], "author": book.get("author", ""),
                          "chapters": chapters})
        return {"root": str(lib.root), "books": books, "judge": judge.available()}

    @app.get("/api/jobs")
    def list_jobs():
        return jobs.list()

    @app.post("/api/jobs")
    def submit_job(kind: str = Body(...), book: str = Body(...), chapter: str = Body(...),
                   part: str | None = Body(None)):
        chapter_or_404(book, chapter)
        try:
            return jobs.submit(kind, book, chapter, part)
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.post("/api/books/{b}/process")
    def process_book(b: str):
        """Queue every chapter that has text and audio and still needs machine work."""
        book = book_or_404(b)
        queued = []
        for ch in book["chapters"]:
            st = lib.status(b, ch)
            if st["stage"] in ("to transcribe", "to align", "to judge"):
                queued.append(jobs.submit("process", b, ch["id"]))
        return {"queued": len(queued)}

    # ---------- books ----------

    @app.post("/api/books")
    def create_book(title: str = Body(...), author: str = Body("")):
        if not title.strip():
            raise HTTPException(400, "title required")
        return lib.create_book(title.strip(), author.strip())

    @app.get("/api/books/{b}")
    def get_book(b: str):
        book = book_or_404(b)
        chapters = []
        for ch in book["chapters"]:
            st = lib.status(b, ch)
            st["busy"] = jobs.busy(b, ch["id"])
            chapters.append({**ch, "status": st})
        return {**book, "chapters": chapters, "audio": lib.audio_files(b),
                "audio_dir": str(lib.book_dir(b) / "audio")}

    @app.patch("/api/books/{b}")
    def patch_book(b: str, title: str | None = Body(None), author: str | None = Body(None)):
        book_or_404(b)
        return lib.update_book(b, title=title, author=author)

    @app.delete("/api/books/{b}")
    def delete_book(b: str):
        book_or_404(b)
        lib.delete_book(b)
        return {"ok": True}

    @app.post("/api/books/{b}/import")
    async def import_text(b: str, file: UploadFile = File(...)):
        """Add chapters from an .epub or a .txt file."""
        book_or_404(b)
        data = await file.read()
        name = (file.filename or "").lower()
        try:
            if name.endswith(".epub"):
                chapters = importers.epub_chapters(data)
            else:
                chapters = importers.text_chapters(data.decode("utf-8", errors="replace"))
        except Exception as e:
            raise HTTPException(400, f"couldn't read that file: {e}")
        for ch in chapters:
            lib.add_chapter(b, ch["title"], ch["text"])
        return {"added": len(chapters)}

    @app.post("/api/books/{b}/audio")
    async def upload_audio(b: str, files: list[UploadFile] = File(...)):
        book_or_404(b)
        dest = lib.book_dir(b) / "audio"
        dest.mkdir(exist_ok=True)
        saved = []
        for f in files:
            name = Path(f.filename or "recording").name
            if Path(name).suffix.lower() not in AUDIO_EXTS:
                raise HTTPException(400, f"{name}: not an audio file I recognize")
            target = dest / name
            with target.open("wb") as out:
                shutil.copyfileobj(f.file, out)
            saved.append(name)
        return {"saved": saved}

    # ---------- chapters ----------

    @app.post("/api/books/{b}/chapters")
    def add_chapter(b: str, title: str = Body(...), text: str = Body("")):
        book_or_404(b)
        return lib.add_chapter(b, title, text)

    @app.patch("/api/books/{b}/chapters/{c}")
    def patch_chapter(b: str, c: str, title: str | None = Body(None), text: str | None = Body(None)):
        chapter_or_404(b, c)
        lib.update_chapter(b, c, title=title, text=text)
        return {"ok": True}

    @app.delete("/api/books/{b}/chapters/{c}")
    def delete_chapter(b: str, c: str):
        chapter_or_404(b, c)
        lib.delete_chapter(b, c)
        return {"ok": True}

    @app.post("/api/books/{b}/chapters/{c}/move")
    def move_chapter(b: str, c: str, delta: int = Body(..., embed=True)):
        chapter_or_404(b, c)
        lib.move_chapter(b, c, delta)
        return {"ok": True}

    @app.post("/api/books/{b}/chapters/{c}/parts")
    def add_part(b: str, c: str, audio: str = Body(..., embed=True)):
        chapter_or_404(b, c)
        try:
            return lib.add_part(b, c, audio)
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.delete("/api/books/{b}/chapters/{c}/parts/{p}")
    def remove_part(b: str, c: str, p: str):
        part_or_404(b, c, p)
        lib.remove_part(b, c, p)
        return {"ok": True}

    @app.post("/api/books/{b}/chapters/{c}/parts/{p}/move")
    def move_part(b: str, c: str, p: str, delta: int = Body(..., embed=True)):
        part_or_404(b, c, p)
        lib.move_part(b, c, p, delta)
        return {"ok": True}

    @app.get("/api/books/{b}/chapters/{c}")
    def chapter_detail(b: str, c: str):
        book, ch = chapter_or_404(b, c)
        text = lib.chapter_text(b, c)
        transcripts = lib.transcripts(b, ch)
        a = lib.alignment(b, c)
        review = lib.review(b, c)
        words = lib.chapter_words(b, ch)
        events = []
        if a:
            kept = set(a["kept_words"])
            for e in a["events"]:
                e = dict(e)
                if e["type"] in CANDIDATES:
                    e["candidate"] = True
                    e["suggestion"] = lib.suggestion(e, review)
                    e["decision"] = review["decisions"].get(e["id"])
                    before, after = book_context(text, e["book_index"], 25, 25)
                    e["context"] = {"book_before": before, "book_after": after}
                events.append(e)
            for w_idx, w in enumerate(words):
                w["kept"] = w_idx in kept
        parts = []
        for k, p in enumerate(ch["parts"]):
            t = transcripts[k]
            parts.append({**p, "index": k, "transcribed": t is not None,
                          "duration": t["duration"] if t else None,
                          "words": [dict(w, i=i) for i, w in enumerate(words) if w["part"] == k]})
        render = None
        rp = lib.work_dir(b, c) / "render.json"
        if rp.exists():
            import json
            render = json.loads(rp.read_text())
            render.pop("chunks", None)
        return {"book": {"id": b, "title": book["title"]}, "chapter": {"id": c, "title": ch["title"]},
                "text": text, "parts": parts, "events": events,
                "status": {**lib.status(b, ch), "busy": jobs.busy(b, c)},
                "coverage": a["coverage"] if a else None, "render": render}

    @app.put("/api/books/{b}/chapters/{c}/decisions")
    def put_decisions(b: str, c: str, changes: dict[str, str | None] = Body(...)):
        chapter_or_404(b, c)
        r = lib.set_decisions(b, c, changes)
        return {"decisions": r["decisions"]}

    @app.get("/api/books/{b}/chapters/{c}/parts/{p}/peaks")
    def peaks(b: str, c: str, p: str):
        part_or_404(b, c, p)
        return JSONResponse(lib.peaks(b, c, p))

    @app.get("/api/books/{b}/chapters/{c}/parts/{p}/audio")
    def part_audio(b: str, c: str, p: str):
        _, _, part = part_or_404(b, c, p)
        return FileResponse(lib.part_audio(b, part))

    @app.get("/api/books/{b}/chapters/{c}/output")
    def output_audio(b: str, c: str):
        chapter_or_404(b, c)
        out = lib.output_path(b, c)
        if not out.exists():
            raise HTTPException(404, "not rendered yet")
        return FileResponse(out, filename=f"{b}-{c}.wav")

    # ---------- the app ----------

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
