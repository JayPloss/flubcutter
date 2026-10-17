import argparse
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(prog="flubcutter")
    parser.add_argument("--library", type=Path, help="library folder (default: ~/Flubcutter)")
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("serve", help="run the web app")
    s.add_argument("--port", type=int, default=8642)
    s.add_argument("--open", action="store_true", help="open it in the browser")

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
        import webbrowser

        import uvicorn

        from flubcutter.library import Library
        from flubcutter.server import create_app

        lib = Library(args.library)
        url = f"http://127.0.0.1:{args.port}/"
        print(f"flubcutter: library {lib.root}\n            {url}")
        if args.open:
            threading.Timer(1.0, webbrowser.open, [url]).start()
        uvicorn.run(create_app(lib), host="127.0.0.1", port=args.port, log_level="warning")

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
        chunks = edit.plan(words, edit.keep_set(alignment, decisions), [before])
        cleaned = edit.render([audio], chunks)
        out = args.out or Path("output") / f"{Path(transcript['audio']).stem}.clean.wav"
        edit.write_wav(cleaned, out)
        after = len(cleaned) / edit.RATE
        print(f"{len(chunks)} chunks, {before:.1f}s -> {after:.1f}s "
              f"({before - after:.1f}s removed) -> {out}")
