"""Step 2b: an AI judge for "extra" speech, i.e. words in the recording that
are not in the book and are not an earlier take of a re-read passage.

Each one is one of two kinds of audio, and Claude decides which, given the
surrounding book text and transcript:

- a *flub*: a mistake to cut (a stray word, a mumble, "where was I");
- an *aside*: unscripted talk that was meant (a conversation with the kids,
  an explanation, a joke). Asides are left out of the Plain edition of the
  audiobook (just the story) and kept in the Extended one, unless they are
  not worth hearing again.

Runs through the `claude` CLI in headless mode, so it uses the machine's
existing Claude login and needs no API key.
"""

import json
import shutil
import subprocess

from flubcutter.align import book_context

MODEL = "sonnet"
BATCH = 30

SYSTEM = """You are the editor of a family audiobook: a parent recorded themselves \
reading a book aloud to their kids, and the recording is being cleaned up \
automatically. Each item you get is speech that is NOT in the book text and is NOT \
an obvious re-read of the next passage. For each, decide its KIND and a VERDICT.

KIND is "flub" or "aside":
- flub: a mistake in the reading. Misreadings or stumbles, stray or partial \
words, mumbles, self-corrections ("wait", "where was I", "let me read that \
again"), reading-mechanics chatter ("next page", "okay", "the screen went off"), \
empty lead-ins and sign-offs ("alright", "okay so"), coughs or yawns transcribed \
as words, fragments that only make sense as a mistake.
- aside: unscripted talk that was meant, a break from the reading rather than \
a slip in it. A conversation with someone in the room (a child's questions and \
the answers, a back-and-forth, even about everyday things), explanations of \
words ("a waistcoat is like a vest"), jokes, reactions to the story, the reader \
talking to the listeners or about the recording. Several turns, a second \
speaker, or a question and an answer mean an aside. The "Alright, back to \
work" that closes an aside belongs to it.

VERDICT is "cut" or "keep":
- a flub: "cut", unless it is plainly not a mistake after all.
- an aside: "keep" (it goes in the Extended edition of the audiobook; the \
Plain edition leaves all asides out anyway), unless it is empty and nobody \
would want to hear it again.

When unsure of the kind, choose "aside" for anything that sounds deliberate \
and "flub" for anything short and mechanical. Give a short reason."""

SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "kind": {"type": "string", "enum": ["flub", "aside"]},
                    "verdict": {"type": "string", "enum": ["cut", "keep"]},
                    "confidence": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["id", "kind", "verdict", "confidence", "reason"],
            },
        }
    },
    "required": ["verdicts"],
}


def available() -> bool:
    return shutil.which("claude") is not None


def _items(events: list[dict], words: list[dict], book_text: str) -> list[dict]:
    items = []
    for e in events:
        before_book, after_book = book_context(book_text, e["book_index"], 30, 30)
        first, last = e["words"][0], e["words"][-1]
        said_before = " ".join(w["word"] for w in words[max(0, first - 20):first])
        said_after = " ".join(w["word"] for w in words[last + 1:last + 21])
        items.append({
            "id": e["id"],
            "extra_speech": e["text"],
            "seconds": round(e["end"] - e["start"], 1),
            "book_text_before": before_book,
            "book_text_after": after_book,
            "recording_before": said_before,
            "recording_after": said_after,
        })
    return items


def judge(events: list[dict], words: list[dict], book_text: str, progress=None) -> dict[str, dict]:
    """Return {event_id: {kind, verdict, confidence, reason}}."""
    if not available():
        raise RuntimeError("the `claude` CLI is not installed, so the AI judge can't run")
    items = _items(events, words, book_text)
    out: dict[str, dict] = {}
    for k in range(0, len(items), BATCH):
        batch = items[k:k + BATCH]
        prompt = ("Judge each item. Return one verdict per id.\n\n"
                  + json.dumps(batch, indent=1, ensure_ascii=False))
        proc = subprocess.run(
            ["claude", "-p", "--model", MODEL, "--output-format", "json",
             "--json-schema", json.dumps(SCHEMA), "--system-prompt", SYSTEM,
             "--tools", "", "--no-session-persistence"],
            input=prompt, capture_output=True, text=True, timeout=600,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"claude CLI failed: {(proc.stderr or proc.stdout)[-500:]}")
        envelope = json.loads(proc.stdout)
        result = envelope.get("structured_output") or envelope.get("result")
        if isinstance(result, str):
            result = json.loads(result[result.find("{"):result.rfind("}") + 1])
        ids = {i["id"] for i in batch}
        for v in result["verdicts"]:
            if v["id"] in ids:
                out[v["id"]] = {"kind": v.get("kind", "flub"), "verdict": v["verdict"],
                                "confidence": v.get("confidence"),
                                "reason": v.get("reason", "")}
        if progress:
            progress(min((k + BATCH) / len(items), 1.0))
    return out
