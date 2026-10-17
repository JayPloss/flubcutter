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
5. **Render**: every cut is placed by measuring the audio rather than
   trusting the word timestamps (which are only good to ~80 ms):
   - the real start and end of each kept word come from a 5 ms loudness
     envelope, in both the full band and the hiss band (so soft endings like
     the "s" in "conversations" aren't clipped), against the recording's own
     noise floor;
   - cuts go in silence only. Where you ran straight from a kept word into a
     flub with no gap, the cut is *tight*: it fades across the quietest point,
     and the app marks it so you know where to listen;
   - **pacing is measured, not guessed.** From every pair of neighbouring
     words you kept, Flubcutter learns how long *you* pause after a full
     stop, after a comma and between other words (run-on words count as no
     pause). A cut gets your usual pause for that spot, counting a breath
     as part of it; a pause of your own that is long enough to sound like a
     mistake is shortened to the long end of your normal range. Dead air is
     the most obvious tell of an edit, so the finished audio is then scored
     on its own: any pause outside your range is flagged and marked in the
     player. Pauses are made of the original silence plus real room tone
     from nearby, with equal-power crossfades;
   - **listen-back check**: after rendering, a few seconds around every cut
     are transcribed again. If the words meeting at a cut don't come out
     intact (a clipped word end, a leaked bit of a flub), that cut is rebuilt
     more cautiously and checked again.
6. **Glitch reports**: play the cleaned chapter in the app and press **G**
   (or the *Glitch* button) right after anything that doesn't sound right.
   The cuts in the last 3 seconds are rebuilt more cautiously (wider safety
   margins, more room tone, longer fades, a slightly tighter pause; never a
   longer one) and the spot replays at once. A cut
   reported three times puts its flub back up for review: the cut itself is
   probably in the wrong place. Every report is logged, with a clip of the
   audio, in `work/<chapter>/joins.json` and `work/<chapter>/glitches/`, so
   the defaults can be tuned from real examples.

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
    work/<chapter>/      transcripts, alignment, review decisions, render
                         report, glitch reports (joins.json, glitches/)
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
5. **Render** the chapter and listen to the result in the *Cleaned chapter*
   player. Cuts are marked under the waveform (amber = tight, green =
   rebuilt, purple = a pause outside your pacing); `N` jumps to just before the next cut. Hear a blip? Press `G`.

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
