import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="flubcutter")
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("transcribe", help="verbatim transcript with word timestamps")
    t.add_argument("audio", type=Path)
    t.add_argument("--model", default="large-v3-turbo")
    t.add_argument("--out", type=Path, help="default: work/<audio name>.transcript.json")

    args = parser.parse_args()

    if args.command == "transcribe":
        from flubcutter.transcribe import save, transcribe

        out = args.out or Path("work") / f"{args.audio.stem}.transcript.json"
        result = transcribe(args.audio, args.model)
        save(result, out)
        print(f"{len(result['words'])} words, {result['duration']}s -> {out}")
