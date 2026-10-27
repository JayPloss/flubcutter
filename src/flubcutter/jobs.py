"""A single background worker that runs pipeline steps one at a time.

Transcription is CPU-heavy, so running steps one after another keeps the
machine responsive and makes progress easy to report.

Each step runs in a fresh process of its own. A step holds hours of audio
and two speech models; Python and the ONNX runtime keep memory they have
used, so in the app's own process every chapter left gigabytes behind. When
the step's process ends, all of it goes back to the system.
"""

import itertools
import multiprocessing
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
                self._run_apart(job)
                job["state"], job["progress"] = "done", 1.0
                if job["kind"] == "process":
                    # Nothing needs a person: go straight on to the cleaned audio.
                    self.render_if_stale(job["book"], job["chapter"])
            except Exception as exc:  # report, don't kill the worker
                job["state"] = "error"
                job["error"] = str(exc) or exc.__class__.__name__
            job["finished"] = time.time()

    def _run_apart(self, job):
        def progress(frac, step):
            job["progress"] = frac
            if step:
                job["step"] = step

        apart(self.lib, "job", dict(job), progress=progress)


def apart(lib: Library, what: str, *args, progress=None, **kwargs):
    """Run a job (`what` "job") or a Library method in a process of its own
    and return its result, passing progress on as it comes."""
    ctx = multiprocessing.get_context("spawn")
    recv, send = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_child, args=(str(lib.root), what, args, kwargs, send), daemon=True)
    proc.start()
    send.close()
    result = error = None
    while True:
        try:
            msg = recv.recv()
        except EOFError:  # the process is gone
            break
        if msg[0] == "progress":
            if progress:
                progress(msg[1], msg[2])
        elif msg[0] == "result":
            result = msg[1]
        elif msg[0] == "error":
            error = msg[1:]
    proc.join()
    recv.close()
    if error is not None:
        raise error[0](error[1])
    if proc.exitcode != 0:
        raise RuntimeError(f"the worker process stopped (exit code {proc.exitcode})")
    return result


def _child(root: str, what: str, args: tuple, kwargs: dict, conn):
    try:
        def progress(frac, step=None):
            conn.send(("progress", round(frac, 3), step))

        lib = Library(root)
        if what == "job":
            result = run(lib, args[0], progress)
        else:
            result = getattr(lib, what)(*args, **kwargs)
        conn.send(("result", result))
    except Exception as exc:
        traceback.print_exc()
        kind = exc.__class__ if exc.__class__ in (RuntimeError, ValueError, KeyError) else RuntimeError
        conn.send(("error", kind, str(exc) or exc.__class__.__name__))
    finally:
        conn.close()


def run(lib: Library, job: dict, progress) -> None:
    b, c = job["book"], job["chapter"]
    if job["kind"] == "transcribe":
        lib.run_transcribe(b, c, job["part"], progress=progress)
    elif job["kind"] == "align":
        lib.run_align(b, c)
    elif job["kind"] == "judge":
        _judge(lib, b, c, progress)
    elif job["kind"] == "render":
        lib.run_render(b, c, progress=progress)
    elif job["kind"] == "process":
        _process(lib, b, c, progress)


def _judge(lib, b, c, progress):
    from flubcutter import judge

    a = lib.alignment(b, c)
    if a is None:
        raise RuntimeError("align the chapter first")
    review = lib.review(b, c)
    todo = [e for e in a["events"] if e["type"] in ("extra", "noise")
            and e["id"] not in review["ai"] and e["id"] not in review["decisions"]]
    if not todo:
        return
    ch = lib.chapter(lib.book(b), c)
    verdicts = judge.judge(todo, lib.chapter_words(b, ch),
                           lib.chapter_text(b, c), progress=progress)
    lib.add_ai_verdicts(b, c, verdicts)


def _process(lib, b, c, progress):
    """Everything that needs no human: transcribe missing parts, align, judge."""
    from flubcutter import judge

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
        _judge(lib, b, c, lambda f: progress(0.9 + 0.1 * f, "AI judging"))
    progress(1.0, "done")
