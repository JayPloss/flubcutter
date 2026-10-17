# flubcutter

Automatically clean up home-recorded read-aloud audiobooks by comparing the
recording against the book's actual text.

## The idea

You read a book aloud (to your kids, say) and record it on a phone or tablet.
The result is charming but full of flubs: misread words, restarts, yawns, long
pauses. flubcutter uses the book text as ground truth to find and cut those,
while **keeping** the good unscripted moments, like asides, jokes and
conversations with the listeners.

## Pipeline (planned)

1. **Transcribe**: speech-to-text with word-level timestamps.
2. **Align**: match the transcript to the book text and find where they differ.
3. **Classify**: an AI model labels each mismatch:
   - *flub*: misread word, false start or repeated phrase → cut the bad take
   - *noise*: yawn, cough, long silence → cut or shorten
   - *ad lib*: a real aside or conversation → keep
4. **Edit**: make the cuts with short crossfades. Optionally loudness-normalize
   and lightly denoise.
5. **Review**: write an edit list so a person can check or override decisions
   before the final render.

## Copyright note

This repo contains **code only**. Book texts and recordings are copyrighted or
personal and are never committed; `.gitignore` excludes them. Put your own
material in the local `books/` and `recordings/` folders.

## Status

Early planning.
