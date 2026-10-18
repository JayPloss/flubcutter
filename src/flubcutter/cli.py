import argparse
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _running(url: str) -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen(url + "api/jobs", timeout=1):
            return True
    except OSError:
        return False


def _open(url: str, app: bool = False) -> None:
    import subprocess
    import webbrowser

    if app and shutil.which("omarchy-launch-webapp"):
        subprocess.Popen(["omarchy-launch-webapp", url])
    else:
        webbrowser.open(url)


def _install_desktop() -> None:
    """Write a launcher entry and icon so Flubcutter shows up in the app menu."""
    home = Path.home()
    icon_dir = home / ".local/share/icons/hicolor/scalable/apps"
    icon_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).parent / "assets" / "flubcutter.svg", icon_dir / "flubcutter.svg")

    # Desktop launches don't get the interactive shell's PATH, so the
    # launcher script sets it up (mise shims provide uv and claude).
    script = home / ".local/bin/flubcutter-app"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        "#!/bin/bash\n"
        "# Open Flubcutter in its own window (starts the server if needed).\n"
        'export PATH="$HOME/.local/share/mise/shims:$HOME/.local/bin:$PATH"\n'
        f'cd "{REPO}" || exit 1\n'
        'exec uv run --quiet flubcutter serve --app "$@" >>"$HOME/.cache/flubcutter.log" 2>&1\n'
    )
    script.chmod(0o755)

    entry = home / ".local/share/applications/flubcutter.desktop"
    entry.write_text(
        "[Desktop Entry]\n"
        "Version=1.0\n"
        "Name=Flubcutter\n"
        "Comment=Clean up read-aloud audiobook recordings\n"
        f"Exec={script}\n"
        "Terminal=false\n"
        "Type=Application\n"
        "Icon=flubcutter\n"
        "Categories=AudioVideo;Audio;\n"
        "StartupNotify=true\n"
    )
    print(f"installed {entry}\n          {script}\n          {icon_dir / 'flubcutter.svg'}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="flubcutter")
    parser.add_argument("--library", type=Path, help="library folder (default: ~/Flubcutter)")
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("serve", help="run the web app")
    s.add_argument("--port", type=int, default=8642)
    s.add_argument("--open", action="store_true", help="open it in the browser")
    s.add_argument("--app", action="store_true",
                   help="open it in its own window; reuse the server if it is already running")

    sub.add_parser("install-desktop", help="add Flubcutter to the desktop app launcher")

    d = sub.add_parser("demo", help="add a sample book (Alice) to the library")
    d.add_argument("audio", type=Path, nargs="?", help="a recording of samples/alice-ch1-opening.txt")

    t = sub.add_parser("transcribe", help="one file: verbatim transcript with word timestamps")
    t.add_argument("audio", type=Path)
    t.add_argument("--out", type=Path, help="default: work/<audio name>.transcript.json")

    a = sub.add_parser("align", help="one file: compare a transcript against the book text")
    a.add_argument("transcript", type=Path)
    a.add_argument("book", type=Path, help="plain-text file of the passage that was read")
    a.add_argument("--out", type=Path, help="default: work/<audio name>.alignment.json")

    c = sub.add_parser("cut", help="one file: render cleaned audio from transcript + alignment")
    c.add_argument("transcript", type=Path)
    c.add_argument("alignment", type=Path)
    c.add_argument("--cut-extras", action="store_true",
                   help="also cut unexplained extra speech (default: keep it)")
    c.add_argument("--out", type=Path, help="default: output/<audio name>.clean.wav")

    args = parser.parse_args()

    if args.command == "serve":
        import threading

        url = f"http://127.0.0.1:{args.port}/"
        if args.app and _running(url):
            _open(url, app=True)
            return

        import uvicorn

        from flubcutter.library import Library
        from flubcutter.server import create_app

        lib = Library(args.library)
        print(f"flubcutter: library {lib.root}\n            {url}", flush=True)
        if args.open or args.app:
            threading.Timer(1.5, _open, [url, args.app]).start()
        uvicorn.run(create_app(lib), host="127.0.0.1", port=args.port, log_level="warning")

    elif args.command == "install-desktop":
        _install_desktop()

    elif args.command == "demo":
        from flubcutter.library import Library

        lib = Library(args.library)
        book = lib.create_book("Alice's Adventures in Wonderland (sample)", "Lewis Carroll")
        text = (REPO / "samples" / "alice-ch1-opening.txt").read_text()
        ch = lib.add_chapter(book["id"], "Down the Rabbit-Hole (opening)", text)
        if args.audio:
            dest = lib.book_dir(book["id"]) / "audio" / args.audio.name
            shutil.copy2(args.audio, dest)
            lib.add_part(book["id"], ch["id"], f"audio/{args.audio.name}")
        print(f"added {book['title']} -> {lib.book_dir(book['id'])}")

    elif args.command == "transcribe":
        from flubcutter.transcribe import save, transcribe

        out = args.out or Path("work") / f"{args.audio.stem}.transcript.json"
        result = transcribe(args.audio)
        save(result, out)
        print(f"{len(result['words'])} words, {result['duration']}s -> {out}")

    elif args.command == "align":
        from flubcutter.align import analyze, report, save

        transcript = json.loads(args.transcript.read_text())
        words = [dict(w, part=0) for w in transcript["words"]]
        result = analyze(words, args.book.read_text())
        stem = Path(transcript["audio"]).stem
        out = args.out or Path("work") / f"{stem}.alignment.json"
        save(result, out)
        print(report(result))
        print(f"\n-> {out}")

    elif args.command == "cut":
        from flubcutter import edit

        transcript = json.loads(args.transcript.read_text())
        alignment = json.loads(args.alignment.read_text())
        words = [dict(w, part=0) for w in transcript["words"]]
        decisions = {e["id"]: "cut" if args.cut_extras or e["type"] != "extra" else "keep"
                     for e in alignment["events"]}
        audio = edit.load(Path(transcript["audio"]))
        before = len(audio) / edit.RATE
        breaks = {int(i): c for i, c in alignment.get("breaks", {}).items()}
        cleaned, rep = edit.render([audio], words, edit.keep_set(alignment, decisions), breaks=breaks)
        out = args.out or Path("output") / f"{Path(transcript['audio']).stem}.clean.wav"
        edit.write_wav(cleaned, out)
        after = len(cleaned) / edit.RATE
        tight = sum(j["tight"] for j in rep["joins"])
        print(f"{len(rep['joins'])} joins ({tight} tight), {before:.1f}s -> {after:.1f}s "
              f"({before - after:.1f}s removed) -> {out}")
