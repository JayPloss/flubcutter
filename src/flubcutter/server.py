"""Local web app: progress dashboard, book setup, and chapter review."""

import shutil
from pathlib import Path

from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from flubcutter import __version__, importers, judge
from flubcutter.align import book_context
from flubcutter.jobs import RENDERABLE, Jobs, apart
from flubcutter.library import AUDIO_EXTS, CANDIDATES, EDITIONS, Library

STATIC = Path(__file__).parent / "static"


def create_app(lib: Library) -> FastAPI:
    app = FastAPI(title="Flubcutter", version=__version__)
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
                st["job"] = jobs.current(b, ch["id"])
                st["busy"] = st["job"] is not None
                chapters.append({"id": ch["id"], "title": ch["title"], "status": st})
            books.append({"id": b, "title": book["title"], "author": book.get("author", ""),
                          "chapters": chapters})
        return {"root": str(lib.root), "books": books, "judge": judge.available(),
                "version": __version__}

    @app.get("/api/jobs")
    def list_jobs():
        out = jobs.list()
        for j in out:
            try:
                j["title"] = lib.chapter(lib.book(j["book"]), j["chapter"])["title"]
            except (KeyError, ValueError):
                j["title"] = j["chapter"]
        return out

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
        """Queue every chapter that still needs machine work: processing (which
        renders by itself, with best guesses for anything still waiting for a
        person) or just a render."""
        book = book_or_404(b)
        queued = []
        for ch in book["chapters"]:
            st = lib.status(b, ch)
            if jobs.busy(b, ch["id"]):
                continue
            if st["stage"] in ("to transcribe", "to align", "to judge"):
                queued.append(jobs.submit("process", b, ch["id"]))
            elif st["stage"] in RENDERABLE and not st["rendered"]:
                queued.append(jobs.submit("render", b, ch["id"]))
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
            st["job"] = jobs.current(b, ch["id"])
            st["busy"] = st["job"] is not None
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
                    e["kind"] = lib.kind(e, review)
                    e["ai_kind"] = review["ai"].get(e["id"], {}).get("kind")
                    e["decision"] = review["decisions"].get(e["id"])
                    e["flag"] = review.get("flagged", {}).get(e["id"])
                    e["needs_you"] = lib.needs_you(e, review)
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
        render = lib.render_info(b, c)
        if render:
            render.pop("chunks", None)
        plain = lib.render_info(b, c, "plain")
        if render and plain:
            render["plain"] = {k: plain[k] for k in ("output_seconds", "pacing", "joins") if k in plain}
        if render:
            render["verdicts"] = lib.joins_state(b, c)["verdicts"]
        return {"book": {"id": b, "title": book["title"]}, "chapter": {"id": c, "title": ch["title"]},
                "text": text, "parts": parts, "events": events,
                "status": {**lib.status(b, ch), "busy": jobs.busy(b, c), "job": jobs.current(b, c)},
                "coverage": a["coverage"] if a else None, "render": render}

    @app.put("/api/books/{b}/chapters/{c}/decisions")
    def put_decisions(b: str, c: str, changes: dict[str, str | None] = Body(...)):
        chapter_or_404(b, c)
        r = lib.set_decisions(b, c, changes)
        return {"decisions": r["decisions"], "render": jobs.render_if_stale(b, c)}

    @app.put("/api/books/{b}/chapters/{c}/kinds")
    def put_kinds(b: str, c: str, changes: dict[str, str | None] = Body(...)):
        """Mark extra speech as an "aside" or a "flub" (null: the AI's call)."""
        chapter_or_404(b, c)
        r = lib.set_kinds(b, c, changes)
        return {"kinds": r.get("kinds", {}), "render": jobs.render_if_stale(b, c)}

    @app.get("/api/books/{b}/chapters/{c}/parts/{p}/peaks")
    def peaks(b: str, c: str, p: str):
        part_or_404(b, c, p)
        return JSONResponse(lib.peaks(b, c, p))

    @app.get("/api/books/{b}/chapters/{c}/parts/{p}/audio")
    def part_audio(b: str, c: str, p: str):
        _, _, part = part_or_404(b, c, p)
        return FileResponse(lib.playable_audio(b, c, part))

    def edition_or_404(edition: str) -> str:
        if edition not in EDITIONS:
            raise HTTPException(404, f"no such edition: {edition}")
        return edition

    @app.get("/api/books/{b}/chapters/{c}/output")
    def output_audio(b: str, c: str, edition: str = "extended"):
        chapter_or_404(b, c)
        out = lib.output_path(b, c, edition_or_404(edition))
        if not out.exists():
            raise HTTPException(404, "not rendered yet")
        return FileResponse(out, filename=f"{b}-{c}-{edition}.wav")

    @app.get("/api/books/{b}/chapters/{c}/output/peaks")
    def output_peaks(b: str, c: str, edition: str = "extended"):
        chapter_or_404(b, c)
        if not lib.output_path(b, c, edition_or_404(edition)).exists():
            raise HTTPException(404, "not rendered yet")
        return JSONResponse(lib.output_peaks(b, c, edition))

    @app.post("/api/books/{b}/chapters/{c}/joins/verdict")
    def join_verdict(b: str, c: str, body: dict = Body(...)):
        """Good or bad, for one cut. A bad one is rebuilt in the background
        (the listener keeps listening) and comes up again once it is."""
        chapter_or_404(b, c)
        try:
            r = lib.rate_join(b, c, str(body["key"]), str(body["verdict"]),
                              edition_or_404(body.get("edition", "extended")),
                              passive=bool(body.get("passive")))
        except (RuntimeError, ValueError) as e:
            raise HTTPException(409, str(e))
        r.pop("render", None)
        if r["verdict"] == "bad" and r.get("joins"):
            r["job"] = jobs.submit("render", b, c)
        return r

    @app.post("/api/books/{b}/chapters/{c}/glitch")
    def glitch(b: str, c: str, body: dict = Body(...)):
        chapter_or_404(b, c)
        try:
            # Rebuilding the spot loads audio and a speech model: in a process of its own.
            r = apart(lib, "report_glitch", b, c, float(body["time"]),
                      edition=edition_or_404(body.get("edition", "extended")))
        except RuntimeError as e:
            raise HTTPException(409, str(e))
        r["render"].pop("chunks", None)
        return r

    # ---------- the app ----------

    @app.get("/")
    def index():
        page = (STATIC / "index.html").read_text().replace("{{version}}", __version__)
        return HTMLResponse(page, headers={"Cache-Control": "no-cache"})

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    # The app's scripts import each other, so a browser holding a cached copy
    # of one and a fresh copy of another gets a blank page: always revalidate
    # (the ETag makes that cheap).
    @app.middleware("http")
    async def revalidate_static(request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    return app
