"""A single background worker that runs pipeline steps one at a time.

Transcription is CPU-heavy, so running steps one after another keeps the
machine responsive and makes progress easy to report.
"""

import itertools
import queue
import threading
import time
import traceback

from flubcutter.library import Library

KINDS = ("transcribe", "align", "judge", "render", "process")
# Stages where a render can run: the machine work is done.
RENDERABLE = ("needs you", "not rendered", "out of date", "ready")


class Jobs:
    def __init__(self, lib: Library):
        self.lib = lib
        self._q: queue.Queue = queue.Queue()
        self._jobs: dict[int, dict] = {}
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        threading.Thread(target=self._worker, daemon=True).start()

    def submit(self, kind: str, book: str, chapter: str, part: str | None = None) -> dict:
        if kind not in KINDS:
            raise ValueError(f"unknown job kind {kind}")
        with self._lock:
            for j in self._jobs.values():
                # A running job may have started before the latest change,
                # so only a queued one makes a new one unnecessary.
                if (j["kind"], j["book"], j["chapter"], j["part"]) == (kind, book, chapter, part) \
                        and j["state"] == "queued":
                    return j
            job = {"id": next(self._ids), "kind": kind, "book": book, "chapter": chapter,
                   "part": part, "state": "queued", "progress": 0.0, "step": "",
                   "error": None, "created": time.time(), "finished": None}
            self._jobs[job["id"]] = job
        self._q.put(job["id"])
        return job

    def list(self) -> list[dict]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j["id"])
            # Drop finished jobs older than ten minutes.
            now = time.time()
            for j in jobs:
                if j["finished"] and now - j["finished"] > 600:
                    self._jobs.pop(j["id"], None)
            return [dict(j) for j in jobs]

    def busy(self, book: str, chapter: str) -> bool:
        return self.current(book, chapter) is not None

    def current(self, book: str, chapter: str) -> dict | None:
        """The chapter's running job, or else its first queued one."""
        with self._lock:
            mine = [j for j in self._jobs.values() if j["book"] == book and j["chapter"] == chapter
                    and j["state"] in ("queued", "running")]
        mine.sort(key=lambda j: (j["state"] != "running", j["id"]))
        return {k: mine[0][k] for k in ("id", "kind", "state", "step", "progress")} if mine else None

    def render_if_stale(self, book: str, chapter: str) -> dict | None:
        """Queue a render whenever the cleaned audio is missing or older than
        the decisions. Edits still waiting for a person render with the best
        guess, so the household player always has the latest version."""
        st = self.lib.status(book, self.lib.chapter(self.lib.book(book), chapter))
        if st["stage"] in RENDERABLE and not st["rendered"]:
            return self.submit("render", book, chapter)
        return None

    def _worker(self):
        while True:
            job = self._jobs.get(self._q.get())
            if job is None:
                continue
            job["state"] = "running"
            try:
                self._run(job)
                job["state"], job["progress"] = "done", 1.0
            except Exception as exc:  # report, don't kill the worker
                job["state"] = "error"
                job["error"] = str(exc) or exc.__class__.__name__
                traceback.print_exc()
            job["finished"] = time.time()

    def _run(self, job):
        lib, b, c = self.lib, job["book"], job["chapter"]

        def progress(frac, step=None):
            job["progress"] = round(frac, 3)
            if step:
                job["step"] = step

        if job["kind"] == "transcribe":
            lib.run_transcribe(b, c, job["part"], progress=progress)
        elif job["kind"] == "align":
            lib.run_align(b, c)
        elif job["kind"] == "judge":
            self._judge(b, c, progress)
        elif job["kind"] == "render":
            lib.run_render(b, c, progress=progress)
        elif job["kind"] == "process":
            self._process(b, c, progress)

    def _judge(self, b, c, progress):
        from flubcutter import judge

        a = self.lib.alignment(b, c)
        if a is None:
            raise RuntimeError("align the chapter first")
        review = self.lib.review(b, c)
        todo = [e for e in a["events"] if e["type"] in ("extra", "noise")
                and e["id"] not in review["ai"] and e["id"] not in review["decisions"]]
        if not todo:
            return
        ch = self.lib.chapter(self.lib.book(b), c)
        verdicts = judge.judge(todo, self.lib.chapter_words(b, ch),
                               self.lib.chapter_text(b, c), progress=progress)
        self.lib.add_ai_verdicts(b, c, verdicts)

    def _process(self, b, c, progress):
        """Everything that needs no human: transcribe missing parts, align, judge."""
        from flubcutter import judge

        lib = self.lib
        ch = lib.chapter(lib.book(b), c)
        if not lib.chapter_text(b, c).strip():
            raise RuntimeError("chapter has no book text")
        if not ch["parts"]:
            raise RuntimeError("chapter has no audio parts")
        todo = [p for p, t in zip(ch["parts"], lib.transcripts(b, ch)) if t is None]
        for k, p in enumerate(todo):
            def part_progress(f, k=k):
                progress(0.85 * (k + f) / len(todo),
                         "transcribing" + (f" part {k + 1} of {len(todo)}" if len(todo) > 1 else ""))
            part_progress(0.0)
            lib.run_transcribe(b, c, p["id"], progress=part_progress)
        progress(0.86, "aligning")
        a = lib.alignment(b, c)
        if a is None or not a.get("noise_checked"):  # aligned before noises were found
            lib.run_align(b, c)
        if judge.available():
            progress(0.9, "AI judging")
            self._judge(b, c, lambda f: progress(0.9 + 0.1 * f, "AI judging"))
        progress(1.0, "done")
        # Nothing needs a person: go straight on to the cleaned audio.
        self.render_if_stale(b, c)
