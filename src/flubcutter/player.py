"""The household player: listen to the finished chapters on any device at home.

Read-only on the library: it lists each book's rendered chapters (both
editions), streams them with Range support so seeking works on phones, and
keeps listening state (resume position per chapter and edition, played) in
~/.local/share/flubcutter-player/state.json. Chapters are transcoded once to
AAC in the background (a 45-minute WAV is ~250 MB; the AAC is ~20 MB), and the
WAV is served until that copy exists.

Run: flubcutter player [--port 8650] [--host 0.0.0.0]
"""
from __future__ import annotations

import json
import queue
import mimetypes
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from flubcutter.library import EDITIONS, Library

STATIC = Path(__file__).resolve().parent / "static" / "player"
STATE = Path.home() / ".local/share/flubcutter-player/state.json"
CACHE = Path.home() / ".cache/flubcutter-player"
LOCK = threading.Lock()
REMOTE_CMDS = ("toggle", "next", "prev")
LISTENERS: list = []  # one queue per Story Shelf window open on this machine


def remote(cmd: str) -> int:
    """Send a remote button press to every local window; returns how many got it."""
    with LOCK:
        for q in LISTENERS:
            q.put(cmd)
        return len(LISTENERS)

WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen".split())}
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60}


def _number(word: str) -> int | None:
    """'Twenty-eight' -> 28, '7' -> 7."""
    w = word.lower()
    if w.isdigit():
        return int(w)
    if w in WORDS:
        return WORDS[w]
    head, _, tail = w.partition("-")
    if head in TENS:
        return TENS[head] + (WORDS.get(tail, 0) if tail else 0)
    return None


def chapter_numbers(text: str) -> list[int]:
    """Book chapter numbers a chapter's text covers, from its 'Chapter N: Title.' headings."""
    found = re.findall(r"^Chapter ([\w-]+):", text, re.M)
    return [n for n in (_number(w) for w in found) if n is not None]


_durations: dict[tuple[str, float], float] = {}


def duration(path: Path) -> float:
    key = (str(path), path.stat().st_mtime)
    if key not in _durations:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                             capture_output=True, text=True).stdout.strip()
        _durations[key] = float(out or 0)
    return _durations[key]


def cached(src: Path, book: str, chapter: str, edition: str) -> Path:
    """Where the AAC copy of a rendered chapter lives; the name carries the WAV's mtime."""
    return CACHE / book / edition / f"{chapter}.{int(src.stat().st_mtime)}.m4a"


def library(lib: Library) -> list[dict]:
    books = []
    for b in lib.book_ids():
        book = lib.book(b)
        chapters = []
        for i, ch in enumerate(book["chapters"]):
            p = lib.text_path(b, ch["id"])
            nums = chapter_numbers(p.read_text()) if p.exists() else []
            editions = {}
            for ed in EDITIONS:
                out = lib.output_path(b, ch["id"], ed)
                if out.exists():
                    editions[ed] = {"duration": duration(out), "at": out.stat().st_mtime}
            chapters.append({"id": ch["id"], "title": ch["title"], "index": i + 1,
                             "nums": nums, "editions": editions})
        books.append({"id": b, "title": book["title"], "author": book.get("author", ""),
                      "order": book.get("order", 0), "chapters": chapters})
    return sorted(books, key=lambda x: x["order"])


def transcode_all(lib: Library) -> None:
    """Make the AAC copies, one at a time, and drop copies of older renders."""
    for b in lib.book_ids():
        for ch in lib.book(b)["chapters"]:
            for ed in EDITIONS:
                src = lib.output_path(b, ch["id"], ed)
                if not src.exists():
                    continue
                dst = cached(src, b, ch["id"], ed)
                if dst.exists():
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                tmp = dst.with_suffix(".part.m4a")
                r = subprocess.run(["nice", "-n", "19", "ionice", "-c3", "ffmpeg", "-v", "error", "-y", "-i", str(src), "-ac", "1", "-c:a", "aac",
                                    "-b:a", "80k", "-movflags", "+faststart", str(tmp)])
                if r.returncode == 0:
                    tmp.replace(dst)
                    for old in dst.parent.glob(f"{ch['id']}.*.m4a"):
                        if old != dst:
                            old.unlink(missing_ok=True)


def load_state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except (OSError, ValueError):
        return {"chapters": {}, "books": {}, "last": None, "speed": 1}


def make_handler(lib: Library):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _file(self, path: Path, cache=False):
            if not path.is_file():
                return self.send_error(404)
            size = path.stat().st_size
            ctype = "audio/mp4" if path.suffix == ".m4a" else mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            start, end = 0, size - 1
            rng = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
            if rng and (rng[1] or rng[2]):
                if rng[1]:
                    start = int(rng[1])
                    end = int(rng[2]) if rng[2] else end
                else:
                    start = size - int(rng[2])
                end = min(end, size - 1)
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Cache-Control", "max-age=86400" if cache else "no-cache")
            self.end_headers()
            with path.open("rb") as f:  # stream in 256 KB pieces, never the whole file
                f.seek(start)
                left = end - start + 1
                try:
                    while left > 0 and (chunk := f.read(min(262144, left))):
                        self.wfile.write(chunk)
                        left -= len(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def do_GET(self):
            p = self.path.split("?")[0]
            if p == "/":
                return self._file(STATIC / "index.html")
            if p.startswith("/static/") and ".." not in p:
                return self._file(STATIC / p.removeprefix("/static/"))
            if p == "/api/library":
                return self._json({"books": library(lib)})
            if p == "/api/state":
                with LOCK:
                    return self._json(load_state())
            if p == "/api/remote/events":
                return self._remote_events()
            if m := re.fullmatch(r"/audio/([\w-]+)/(ch\d+)/(\w+)", p):
                b, c, ed = m.groups()
                if ed not in EDITIONS or b not in lib.book_ids():
                    return self.send_error(404)
                src = lib.output_path(b, c, ed)
                if not src.exists():
                    return self.send_error(404)
                aac = cached(src, b, c, ed)
                return self._file(aac if aac.exists() else src, cache=aac.exists())
            self.send_error(404)

        def _local(self):
            return self.client_address[0] in ("127.0.0.1", "::1", "::ffff:127.0.0.1")

        def _remote_events(self):
            # Only windows on this machine follow the remote, not phones on the Wi-Fi.
            if not self._local():
                return self.send_error(403)
            q = queue.Queue()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            with LOCK:
                LISTENERS.append(q)
            try:
                self.wfile.write(b": hi\n\n")
                self.wfile.flush()
                while True:
                    try:
                        msg = f"data: {q.get(timeout=20)}\n\n"
                    except queue.Empty:
                        msg = ": ping\n\n"  # notices a closed window
                    self.wfile.write(msg.encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                with LOCK:
                    LISTENERS.remove(q)

        def do_POST(self):
            if self.path.startswith("/api/remote/"):
                cmd = self.path.removeprefix("/api/remote/")
                if not self._local() or cmd not in REMOTE_CMDS:
                    return self.send_error(404)
                n = remote(cmd)
                return self._json({"windows": n}, 200 if n else 409)
            if self.path != "/api/state":
                return self.send_error(404)
            patch = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            with LOCK:
                state = load_state()
                for key in ("chapters", "books"):
                    for k, v in (patch.get(key) or {}).items():
                        state.setdefault(key, {}).setdefault(k, {}).update(v)
                for k in ("last", "speed"):
                    if k in patch:
                        state[k] = patch[k]
                STATE.parent.mkdir(parents=True, exist_ok=True)
                tmp = STATE.with_suffix(".tmp")
                tmp.write_text(json.dumps(state))
                tmp.replace(STATE)
            self._json({"ok": True})

    return Handler


def serve(lib: Library, host: str = "0.0.0.0", port: int = 8650) -> None:
    def keep_transcoding():
        while True:
            try:
                transcode_all(lib)
            except Exception as e:  # a bad file must not stop the player
                print(f"player: transcode failed: {e}", flush=True)
            time.sleep(120)

    threading.Thread(target=keep_transcoding, daemon=True).start()
    srv = ThreadingHTTPServer((host, port), make_handler(lib))
    print(f"flubcutter player: http://{host}:{port}/", flush=True)
    srv.serve_forever()
