# flubcutter

Automatically clean up home-recorded read-aloud audiobooks by comparing the
recording against the book's actual text.

You read a book aloud (to your kids, say) and record it on a phone or tablet.
The result is charming but full of flubs: misread words, restarts, yawns, long
pauses. flubcutter uses the book text as ground truth to find and cut those,
while **keeping** the good unscripted moments, like asides, jokes and
conversations with the listeners.

## How it works

1. **Transcribe**: [NVIDIA Parakeet TDT 0.6B](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)
   (via [onnx-asr](https://github.com/istupakov/onnx-asr)) on the CPU, at about
   10× real time, with word timestamps. Unlike Whisper, a transducer model
   doesn't "tidy up" speech, so repeated phrases and false starts stay in the
   transcript, where we need them.
2. **Align**: the transcript is aligned word by word against the book text.
   **The last take wins:** when you re-read a passage until you're happy, the
   final reading is paired with the book and every earlier attempt becomes a
   *restart* to cut. This also works across recording parts, for example when
   you pick up after a break by re-reading the last paragraph.
3. **Judge**: speech that isn't in the book and isn't a re-read ("extra
   speech") goes to Claude, which decides *cut* (stumbles, "where was I",
   lead-ins) or *keep* (jokes, explanations, chats with the kids). This step
   runs through the `claude` CLI, so it uses your existing Claude login.
4. **Review**: in the web app you see the waveform with every flub
   highlighted, hear each one with pre-roll and post-roll ("play with cut" vs
   "play original"), and approve or reject it, mostly with the keyboard.
5. **Render**: the cuts are applied with room-tone pauses capped at 1.2 s,
   cut points snapped to the quietest nearby 10 ms, and short crossfades.

## Using it

Requirements: Linux or macOS, `ffmpeg`, [`uv`](https://docs.astral.sh/uv/), and
optionally the [`claude` CLI](https://docs.claude.com/en/docs/claude-code) for
the AI judge. The speech model (~650 MB) downloads on first use.

```sh
uv run flubcutter serve --open     # web app at http://127.0.0.1:8642
uv run flubcutter install-desktop  # add "Flubcutter" to the desktop app launcher
uv run flubcutter demo recordings/my-reading.wav   # sample book (Alice, ch. 1 opening)
```

The library lives in `~/Flubcutter`, or set `FLUBCUTTER_LIBRARY` /
`--library`:

```
~/Flubcutter/<book>/
    book.json            chapters and their recording parts, in order
    text/<chapter>.txt   book text per chapter
    audio/               your recordings (drop files here or upload in the app)
    work/<chapter>/      transcripts, alignment, review decisions
    output/<chapter>.wav cleaned chapters
```

In the app:

1. **New book**, then **Import EPUB / text** to create the chapters from an
   ebook file (or add them one by one and paste the text).
2. Add recordings, then assign one or more to each chapter as *parts*, in
   reading order.
3. **Process all** transcribes, aligns and AI-judges every chapter in the
   background.
4. Open a chapter to review. Keys: `P`/`Enter` play with cut, `O` play the
   original, `X` cut, `K` keep, `←`/`→` previous/next, `Space` play/pause,
   `Backspace` undo. With auto-advance on, each decision jumps to the next
   flub and plays it.
5. **Render** the chapter and listen to the result.

There are also single-file commands for quick experiments:
`flubcutter transcribe`, `align` and `cut`.

## Copyright note

This repo contains **code only**. Book texts and recordings are copyrighted or
personal and are never committed; `.gitignore` excludes them, and the library
lives outside the repo. `samples/` holds a short public-domain passage
(*Alice's Adventures in Wonderland*) for testing.

## Development

```sh
uv run pytest
```

The web app is plain ES modules in `src/flubcutter/static/` with no build
step; [wavesurfer.js](https://wavesurfer.xyz) 7.12 is vendored in
`static/vendor/`.

## License

MIT. See [LICENSE](LICENSE). This covers the code only, not any book text or
recordings.
