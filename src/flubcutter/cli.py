import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="flubcutter")
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("transcribe", help="verbatim transcript with word timestamps")
    t.add_argument("audio", type=Path)
    t.add_argument("--model", default="large-v3-turbo")
    t.add_argument("--out", type=Path, help="default: work/<audio name>.transcript.json")

    a = sub.add_parser("align", help="compare a transcript against the book text")
    a.add_argument("transcript", type=Path)
    a.add_argument("book", type=Path, help="plain-text file of the passage that was read")
    a.add_argument("--out", type=Path, help="default: work/<audio name>.alignment.json")

    args = parser.parse_args()

    if args.command == "transcribe":
        from flubcutter.transcribe import save, transcribe

        out = args.out or Path("work") / f"{args.audio.stem}.transcript.json"
        result = transcribe(args.audio, args.model)
        save(result, out)
        print(f"{len(result['words'])} words, {result['duration']}s -> {out}")

    elif args.command == "align":
        import json

        from flubcutter.align import analyze, report, save

        transcript = json.loads(args.transcript.read_text())
        result = analyze(transcript, args.book.read_text())
        stem = Path(transcript["audio"]).stem
        out = args.out or Path("work") / f"{stem}.alignment.json"
        save(result, out)
        print(report(result))
        print(f"\n-> {out}")
