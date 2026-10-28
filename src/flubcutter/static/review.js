import WaveSurfer from "/static/vendor/wavesurfer/wavesurfer.esm.js";
import RegionsPlugin from "/static/vendor/wavesurfer/plugins/regions.esm.js";
import TimelinePlugin from "/static/vendor/wavesurfer/plugins/timeline.esm.js";
import MinimapPlugin from "/static/vendor/wavesurfer/plugins/minimap.esm.js";
import { api, fmtTime, guard, h, MACHINE, pollJobs, setCrumbs, statusOf, statusTag, submitJob, toast } from "./app.js";

const TYPE_LABEL = { restart: "Restart", filler: "Filler", extra: "Extra speech", noise: "Sound, no words" };
const judged = (e) => e.type === "extra" || e.type === "noise";
// Extra speech is either a flub or an aside: unscripted talk that was meant
// (a chat with the kids, a joke). Asides are in the Extended edition only.
const typeLabel = (e) => (e.kind === "aside" ? "Aside" : TYPE_LABEL[e.type] || e.type);
const isAside = (e) => e.kind === "aside";
const EDITION_LABEL = { plain: "Plain", extended: "Extended" };

let S = null; // state of the open review screen

function pref(key, dflt) {
  try { const v = localStorage.getItem("fc." + key); return v === null ? dflt : JSON.parse(v); } catch { return dflt; }
}
function setPref(key, v) {
  try { localStorage.setItem("fc." + key, JSON.stringify(v)); } catch {}
}
function css(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export function teardownReview() {
  if (!S) return;
  if (S.pass) { cancelAnimationFrame(S.pass.raf); S.pass.audio.pause(); URL.revokeObjectURL(S.pass.url); S.pass = null; }
  document.removeEventListener("keydown", S.onKey);
  S.player?.stop();
  S.ws?.destroy();
  S.audio?.pause();
  S.ows?.destroy();
  S.oaudio?.pause();
  S = null;
}

const effective = (e) => e.decision || e.suggestion.decision;

export async function renderReview(view, bookId, chapterId) {
  const data = await api(`/api/books/${bookId}/chapters/${chapterId}`);
  setCrumbs([
    { label: "Library", href: "#/" },
    { label: data.book.title, href: `#/book/${bookId}` },
    { label: data.chapter.title },
  ]);
  S = {
    bookId, chapterId, data, view,
    part: 0, sel: null, ws: null, regions: null, audio: null, player: null,
    filter: pref("filter", "todo"),
    edition: pref("edition", "extended"),
    pre: pref("pre", 3), post: pref("post", 3),
    zoom: pref("zoom2", 12), autoplay: pref("autoplay", true),
    onKey: null,
  };
  // What needs you comes first, the least certain first; the rest (accepted
  // as suggested unless you change them) follow in time order.
  const sure = (e) => (e.flag ? -1 : e.suggestion.confidence ?? 0);
  S.cands = data.events.filter((e) => e.candidate)
    .sort((a, b) => (b.needs_you - a.needs_you) || (a.needs_you ? sure(a) - sure(b) : 0) || a.part - b.part || a.start - b.start);
  S.notes = data.events.filter((e) => e.type === "sub" || e.type === "missing");
  S.byWord = new Map();
  for (const e of S.cands) for (const w of e.words) S.byWord.set(w, e);

  build();

  const first = S.cands.find((e) => !e.decision) || S.cands[0];
  if (first) select(first, { play: false });

  // Called when background jobs change: rebuild only if this chapter changed.
  return refreshReview;
}

async function refreshReview() {
  const { view, bookId, chapterId } = S || {};
  {
    if (!S) return;
    if (S.pass) { S.dirty = true; return; }  // not in the middle of a listen-through
    const fresh = await api(`/api/books/${bookId}/chapters/${chapterId}`);
    const sig = (d) => JSON.stringify([d.status.stage, d.status.busy, d.status.judged, d.events.length, d.parts.map((p) => p.transcribed), d.render?.rendered]);
    if (sig(fresh) !== sig(S.data)) {
      const keepSel = S.sel?.id;
      teardownReview();
      await renderReview(view, bookId, chapterId);
      const again = S?.cands.find((e) => e.id === keepSel);
      if (again) select(again, { play: false });
    }
  };
}

// ---------- layout ----------

function build() {
  const { data } = S;
  const st = data.status;
  const left = h("div", { class: "stack" });
  const right = h("div", { class: "panel cands", id: "cands" });

  const header = h("div", { class: "row" },
    h("div", {},
      h("h1", {}, data.chapter.title),
      h("div", { class: "row small muted", style: "margin-top:4px" },
        data.coverage !== null ? h("span", {}, `${Math.round(data.coverage * 100)}% of book text matched`) : null,
        S.notes.length ? h("span", {}, `${S.notes.length} word-level differences`) : null)),
    h("span", { class: "spacer" }),
    h("div", { id: "progress" }),
    MACHINE.includes(st.stage) && !st.busy
      ? h("button", { class: "primary", onclick: () => guard(() => submitJob("process", S.bookId, S.chapterId)) }, "Process chapter")
      : null);

  if (!st.aligned) {
    S.view.replaceChildren(h("div", { class: "stack" }, header,
      h("div", { class: "panel empty" },
        st.busy ? "Working on this chapter… this page updates when it's ready."
          : !data.parts.length ? h("span", {}, "This chapter has no recordings yet. ", h("a", { href: `#/book/${S.bookId}` }, "Add one on the book page"), ".")
            : "Not processed yet. Press “Process chapter” to transcribe and align it.")));
    renderProgress();
    return;
  }

  // Waveform panel
  const tabs = data.parts.length > 1 ? h("div", { class: "tabs" },
    data.parts.map((p, k) => h("button", { class: k === S.part ? "on" : "", onclick: () => showPart(k) },
      `Part ${k + 1}`, !p.transcribed ? " (not transcribed)" : ""))) : null;

  const zoom = h("input", { type: "range", min: 5, max: 400, value: S.zoom, style: "width:140px",
    oninput: () => { S.zoom = +zoom.value; setPref("zoom2", S.zoom); S.ws?.zoom(S.zoom); } });
  const pre = h("input", { type: "number", min: 0, max: 30, step: 0.5, value: S.pre,
    onchange: () => { S.pre = +pre.value; setPref("pre", S.pre); } });
  const post = h("input", { type: "number", min: 0, max: 30, step: 0.5, value: S.post,
    onchange: () => { S.post = +post.value; setPref("post", S.post); } });
  const auto = h("input", { type: "checkbox", checked: S.autoplay,
    onchange: () => { S.autoplay = auto.checked; setPref("autoplay", S.autoplay); } });

  left.append(
    h("section", { class: "panel" },
      tabs ? h("div", { style: "padding:8px 12px 0" }, tabs) : null,
      h("div", { class: "wavebox" }, h("div", { id: "wave" }), h("div", { id: "minimap" }),
        h("div", { id: "ticks", class: "ticks", title: "Every flub in this part. Click one to jump to it." })),
      h("div", { class: "transport" },
        h("button", { id: "playbtn", class: "primary", onclick: togglePlay }, "▶ Play"),
        h("span", { id: "time", class: "time mono small" }, "0:00.0"),
        h("span", { class: "spacer" }),
        h("label", { class: "inline" }, "Zoom", zoom),
        h("button", { class: "ghost small", onclick: () => { const px = Math.max(1, Math.floor(document.getElementById("wave").clientWidth / (S.ws?.getDuration() || 1))); zoom.value = px; S.zoom = px; S.ws?.zoom(px); } }, "Fit"),
        h("label", { class: "inline" }, "Pre-roll", pre, "s"),
        h("label", { class: "inline" }, "Post-roll", post, "s"))),
    h("section", { class: "panel", id: "detail" }),
    h("section", { class: "panel" },
      h("div", { class: "panel-head" }, h("h2", {}, "Transcript"),
        h("span", { class: "small muted" }, "Click a word to jump there. Struck-through words get cut; highlighted words are flagged.")),
      h("div", { class: "panel-body transcript", id: "transcript" })),
  );

  right.append(
    h("div", { class: "panel-head" },
      h("h2", {}, "Flubs"),
      h("span", { class: "spacer" }),
      h("label", { class: "inline", title: "After a decision, jump to the next flub and play it" }, auto, "auto-advance")),
    h("div", { class: "row", style: "padding:8px 14px;border-bottom:1px solid var(--line)" },
      ["todo", "all", "notes"].map((f) => h("button", { class: "small " + (S.filter === f ? "primary" : "ghost"), "data-filter": f,
        onclick: () => { S.filter = f; setPref("filter", f); renderList(); } },
        f === "todo" ? `Needs you (${S.cands.filter(waiting).length})` : f === "all" ? "All" : "Word diffs"))),
    h("div", { id: "candlist" }),
    h("div", { style: "padding:12px 14px" },
      h("button", { onclick: acceptAll, title: "Take the suggested decision for every flub you haven't decided" }, "Accept all suggestions")));

  // The listen-through comes first; the recording, transcript and edit
  // list are only needed when something needs you, so they fold away.
  const need = S.cands.filter(waiting).length;
  const bench = h("details", { class: "bench", id: "bench", open: need > 0 || !data.render,
    ontoggle: () => { if (bench.open && !S.ws) showPart(S.part); } },
    h("summary", {}, h("b", {}, "Edit decisions"), " ",
      h("span", { class: "muted small" }, need ? `${need} need${need > 1 ? "" : "s"} you` : `${S.cands.length} edits, all settled · the recording, transcript and every edit`)),
    h("div", { class: "review" }, left, right));
  S.view.replaceChildren(h("div", { class: "stack" }, header, renderPanel(), bench));

  S.onKey = onKey;
  document.addEventListener("keydown", S.onKey);
  renderProgress();
  renderList();
  renderDetail();
  if (bench.open) showPart(S.part);
}

function renderPanel(playOpts = {}) {
  const { data } = S;
  S.ows?.destroy();
  S.ows = null;
  S.oaudio?.pause();
  const full = data.render;
  // The Plain edition differs only in its audio, joins and pacing.
  const r = full && S.edition === "plain" && full.plain ? { ...full, ...full.plain } : full;
  const out = `/api/books/${S.bookId}/chapters/${S.chapterId}/output?edition=${S.edition}`;
  const joins = r?.joins || [];
  const asides = full?.asides?.length || 0;
  const tight = joins.filter((j) => j.tight).length;
  const rebuilt = joins.filter((j) => j.level > 0).length;
  const panel = h("section", { class: "panel", id: "renderpanel" },
    h("div", { class: "panel-head" },
      h("h2", {}, "Cleaned chapter"),
      full?.plain ? h("span", { class: "seg", title: "Plain: just the story. Extended: the story with the asides you kept." },
        ...["plain", "extended"].map((ed) => h("button", { class: "small" + (S.edition === ed ? " on" : ""),
          onclick: () => setEdition(ed) },
          `${EDITION_LABEL[ed]} ${fmtTime(ed === "plain" ? full.plain.output_seconds : full.output_seconds)}`))) : null,
      full?.plain && S.edition === "extended" ? h("span", { class: "small muted" },
        asides ? `${asides} aside${asides > 1 ? "s" : ""} kept` : "no asides kept") : null,
      r ? h("span", { class: "small muted" }, `${fmtTime(r.input_seconds)} → ${fmtTime(r.output_seconds)} (${fmtTime(r.input_seconds - r.output_seconds)} removed)`) : null,
      joins.length ? h("span", { class: "small muted", title: "Tight cuts had no silence to cut in. Rebuilt joins were fixed after a glitch report or a failed listen-back check." },
        `${joins.length} cuts · ${tight} tight · ${rebuilt} rebuilt`) : null,
      r?.rhythm ? paceBadge(r) : null,
      h("span", { class: "spacer" }),
      r ? h("a", { class: "btn", href: out, download: true }, `Download ${full?.plain ? EDITION_LABEL[S.edition] + " " : ""}WAV`) : null,
      data.status.job?.kind === "render" ? statusTag(data.status, { book: S.bookId, chapter: S.chapterId })
        : data.status.stage === "out of date" ? h("span", { class: "status t-todo", title: "Your edits changed since this render; this is the previous version" }, h("span", { class: "label" }, "Out of date")) : null,
      h("button", { class: ["not rendered", "out of date"].includes(data.status.stage) && !data.status.busy ? "primary" : "", disabled: data.status.busy,
        onclick: () => guard(async () => { await submitJob("render", S.bookId, S.chapterId); S.data.status.job = { kind: "render", state: "queued", progress: 0 }; S.data.status.busy = true; renderProgress(); document.getElementById("renderpanel")?.replaceWith(renderPanel()); }) },
        r ? "Re-render" : "Render")));
  if (!r) {
    panel.append(h("div", { class: "panel-body small muted" },
      data.status.busy ? "Making the cleaned audio. It plays here when it's ready."
        : "Render to make the cleaned audio. Edits you haven't decided use the suggestion."));
    return panel;
  }
  panel.append(
    h("div", { class: "wavebox" }, h("div", { id: "outwave" }),
      h("div", { id: "outticks", class: "ticks", title: "Every cut in the cleaned chapter. Click one to hear it." })),
    h("div", { class: "listen" },
      h("button", { id: "passbtn", class: "primary big", onclick: () => (S.pass ? stopPass() : startPass()) }, passLabel()),
      h("div", { id: "passinfo", class: "small" }, passIntro()),
      h("span", { class: "spacer" }),
      h("span", { class: "keys small muted" }, h("kbd", {}, "B"), " bad (no answer means good) ", h("kbd", {}, "← →"), " previous / next cut ", h("kbd", {}, "Space"), " pause ", h("kbd", {}, "Esc"), " stop")),
    h("div", { class: "transport" },
      h("button", { id: "outplay", class: "ghost", onclick: toggleOut }, "▶ Play whole chapter"),
      h("button", { class: "ghost", title: "Back 5 seconds", onclick: () => outSeek(-5) }, "↺ 5s"),
      h("button", { class: "ghost", title: "Jump to just before the next cut and play (N)", onclick: nextJoin }, "Next cut ▸"),
      h("span", { id: "outtime", class: "time mono small" }, "0:00.0"),
      h("span", { class: "spacer" }),
      h("span", { class: "small muted" }, "Playing the whole chapter, B marks the cut you just heard.")));
  const tips = recordingTips(joins);
  if (tips.length) panel.append(h("details", { class: "tips" },
    h("summary", { class: "small" }, h("b", {}, `Recording tips (${tips.length})`), h("span", { class: "muted" }, " · what in the raw recording made a cut hard")),
    h("ul", { class: "small" }, tips.map((t) => h("li", {},
      h("a", { href: "#", onclick: (ev) => { ev.preventDefault(); S.pass ? passJump(t.join) : playFrom(t.join.start - 2.5); } }, fmtTime(t.join.start)),
      ` “${t.join.words.join(" | ")}”: ${t.text}`,
      t.accept ? h("button", { class: "small ghost", style: "margin-left:8px", onclick: () => rateCut("good", t.join) }, "Accept as is") : null)))));
  setTimeout(() => showOutput(playOpts), 0);
  return panel;
}

// What the raw recording could do better, from the cuts that were hard to
// make. Some spots can only be re-recorded, or accepted.
const MAX_LEVEL = 2;
function recordingTips(joins) {
  const out = [];
  for (const j of joins) {
    const v = verdictOf(j);
    if (v === "bad" && (j.level || 0) >= MAX_LEVEL) {
      out.push({ join: j, accept: true,
        text: `still sounds wrong after the most careful rebuild. Re-record from “${j.before.join(" ")}”, or accept it.` });
    } else if (j.tight && v !== "good") {
      out.push({ join: j, text: `you went straight from “${j.words[0]}” into the next take with no pause, so the cut had no silence to go in. When you restart, stop for a beat first.` });
    }
  }
  return out;
}

// ---------- listen-through: each cut with 5 s either side, G good / B bad ----------

const AROUND = 5;   // seconds heard before and after each cut
const FADE = 0.35;  // fade between one stretch and the next

// Your call on a cut, if it is for the cut as it is built now (a rebuilt cut
// comes up again).
function verdictOf(j) {
  const v = S.data.render?.verdicts?.[j.key];
  return v && v.level === (j.level || 0) ? v.verdict : null;
}
// Worth an ear: a cut with no silence to hide in, or one you called bad
// before it was rebuilt. A pause shortened inside one take, or a cut made in
// a silence, has never sounded wrong to you, so those are left out (they are
// still in "Listen to all the cuts again").
function risky(j) {
  const v = S.data.render?.verdicts?.[j.key];
  return (j.kind !== "pause" && j.tight) || (v && v.verdict === "bad");
}
const toHear = () => (shown()?.joins || []).filter((j) => risky(j) && verdictOf(j) !== "good").sort((a, b) => a.start - b.start);

function stretches(joins, dur) {
  const out = [];
  for (const j of joins) {
    const a = Math.max(0, j.start - AROUND), b = Math.min(dur, j.end + AROUND);
    const last = out[out.length - 1];
    if (last && a <= last.end) { last.end = Math.max(last.end, b); last.joins.push(j); }
    else out.push({ start: a, end: b, joins: [j] });
  }
  return out;
}

function passLabel() {
  const n = toHear().length;
  return S.pass ? "■ Stop" : n ? `▶ Listen to the cuts (${n})` : "▶ Listen to all the cuts again";
}
function passIntro() {
  const all = shown()?.joins || [], n = toHear().length;
  const secs = stretches(toHear(), Infinity).reduce((a, s) => a + s.end - s.start, 0);
  if (!all.length) return "No cuts in this chapter.";
  return n ? `${n} of ${all.length} cuts to hear, about ${Math.max(1, Math.round(secs / 60))} min; the rest are pauses and cuts in a silence. Press B if one sounds wrong.`
    : `Nothing left to hear among the ${all.length} cuts.`;
}

async function startPass() {
  const r = shown();
  if (!r?.joins?.length) return;
  let joins = toHear();
  if (!joins.length) joins = [...r.joins].sort((a, b) => a.start - b.start);
  S.ows?.pause(); S.player?.stop(); S.ws?.pause();
  const btn = document.getElementById("passbtn");
  if (btn) { btn.disabled = true; btn.textContent = "Loading…"; }
  // Play from a copy of the file: fixes re-render it while you listen.
  const ed = S.data.render?.plain ? S.edition : "extended";
  const blob = await (await fetch(`/api/books/${S.bookId}/chapters/${S.chapterId}/output?edition=${ed}&t=${S.data.render?.rendered || 0}`)).blob();
  if (!S) return;
  const audio = new Audio(URL.createObjectURL(blob));
  await new Promise((ok) => audio.addEventListener("loadedmetadata", ok, { once: true }));
  S.pass = { audio, url: audio.src, segs: stretches(joins, r.output_seconds || Infinity), i: 0, cur: null, good: 0, bad: 0, raf: 0, edition: ed,
    rated: new Set(), jumped: false };
  if (btn) { btn.disabled = false; btn.textContent = passLabel(); }
  playStretch(0);
}

function playStretch(i, from = null) {
  const P = S?.pass;
  if (!P) return;
  if (i >= P.segs.length) { heardOut(P.cur); return stopPass(true); }
  P.i = i;
  const seg = P.segs[i];
  P.audio.currentTime = from ?? seg.start;
  P.audio.volume = 0;
  P.audio.play();
  cancelAnimationFrame(P.raf);
  const tick = () => {
    if (!S?.pass || S.pass !== P) return;
    const t = P.audio.currentTime;
    // Fade in at the start of a stretch, out at its end (not at the very
    // start or end of the chapter).
    const fin = seg.start > 0 ? Math.min(1, (t - (from ?? seg.start)) / FADE) : 1;
    const fout = Math.min(1, (seg.end - t) / FADE);
    P.audio.volume = Math.max(0, Math.min(fin, fout));
    // The cut you're hearing: the last one reached, else the next one.
    const j = [...seg.joins].reverse().find((x) => x.start <= t) || seg.joins[0];
    if (j !== P.cur) { if (P.cur && !P.jumped) heardOut(P.cur); P.jumped = false; P.cur = j; showPassInfo(); }
    S.ows?.setTime(t);
    if (t >= seg.end || P.audio.ended) { heardOut(P.cur); P.cur = null; return playStretch(i + 1); }
    P.raf = requestAnimationFrame(tick);
  };
  P.raf = requestAnimationFrame(tick);
}

// A cut heard right through without a B is good: no need to say so.
function heardOut(j) {
  const P = S?.pass;
  if (!P || !j || P.rated.has(j.key)) return;
  rateCut("good", j, true);
}

// Jump the listen-through to a cut (arrow keys, or a click on the waveform
// or a cut mark). The cut you jump away from isn't counted as heard.
function passJump(j) {
  const P = S?.pass;
  if (!P || !j) return;
  let i = P.segs.findIndex((g) => g.joins.some((x) => x.key === j.key));
  if (i < 0) {  // a cut not in this pass (already marked good): hear it anyway
    P.segs.push({ start: Math.max(0, j.start - AROUND), end: j.end + AROUND, joins: [j] });
    P.segs.sort((a, b) => a.start - b.start);
    i = P.segs.findIndex((g) => g.joins[0].key === j.key);
  }
  P.jumped = true;
  P.cur = j;
  showPassInfo();
  playStretch(i, Math.max(P.segs[i].start, j.start - AROUND));
}
function passStep(dir) {
  const all = [...(shown()?.joins || [])].sort((a, b) => a.start - b.start);
  const k = all.findIndex((x) => x.key === S.pass?.cur?.key);
  const j = all[Math.max(0, Math.min(all.length - 1, (k < 0 ? 0 : k + dir)))];
  if (j) passJump(j);
}
function passSeek(t) {
  const all = [...(shown()?.joins || [])].sort((a, b) => a.start - b.start);
  passJump(all.find((x) => x.end + 1 >= t) || all[all.length - 1]);
}

function showPassInfo() {
  const P = S?.pass, el = document.getElementById("passinfo");
  if (!el || !P?.cur) return;
  const all = [...(shown()?.joins || [])].sort((a, b) => a.start - b.start);
  const n = all.findIndex((x) => x.key === P.cur.key) + 1;
  const v = verdictOf(P.cur);
  el.replaceChildren(h("b", {}, `Cut ${n} of ${all.length}`), ` at ${fmtTime(P.cur.start)}: “${P.cur.words[0]} | ${P.cur.words[1]}”`,
    v ? h("span", { class: `tag ${v === "good" ? "keep" : "cut"}`, style: "margin-left:8px" }, v === "good" ? "good" : "bad: fixing") : null,
    h("span", { class: "muted" }, ` · ${P.good} good, ${P.bad} bad so far`));
}

function stopPass(finished = false) {
  const P = S?.pass;
  if (!P) return;
  cancelAnimationFrame(P.raf);
  P.audio.pause();
  URL.revokeObjectURL(P.url);
  S.pass = null;
  toast(finished ? `Done: ${P.good} good, ${P.bad} bad.${P.bad ? " The bad ones are being rebuilt; they come up again next time." : ""}` : "Stopped.");
  document.getElementById("renderpanel")?.replaceWith(renderPanel());
  if (S.dirty) { S.dirty = false; refreshReview(); }
}

// G or B. In a listen-through, for the cut being heard; while playing the
// whole chapter, for the last cut before the playhead (within 4 s).
async function rateCut(verdict, which = null, passive = false) {
  let j = which || S.pass?.cur;
  if (!j) {
    const t = S.ows?.getCurrentTime() ?? 0;
    j = (shown()?.joins || []).filter((x) => x.start <= t && x.start >= t - 4).sort((a, b) => b.start - a.start)[0];
    if (!j) return toast("No cut in the last 4 seconds.");
  }
  const P = S.pass;
  if (P) {
    if (P.rated.has(j.key) && verdict === "good") return;
    if (P.rated.has(j.key) && verdictOf(j) === "good") P.good -= 1;  // changed your mind
    P.rated.add(j.key);
    P[verdict] += 1;
  }
  S.data.render.verdicts = { ...(S.data.render.verdicts || {}), [j.key]: { verdict, level: j.level || 0 } };
  showPassInfo(); drawJoins();
  if (!passive) toast(verdict === "good" ? `Good: “${j.words.join(" | ")}”` : `Bad: “${j.words.join(" | ")}”. Rebuilding it in the background; keep listening.`);
  await guard(async () => {
    await api(`/api/books/${S.bookId}/chapters/${S.chapterId}/joins/verdict`, { json: { key: j.key, verdict, passive, edition: P?.edition || S.edition } });
    if (verdict === "bad") pollJobs();
  });
}

function setEdition(ed) {
  S.edition = ed;
  setPref("edition", ed);
  document.getElementById("renderpanel")?.replaceWith(renderPanel());
}

// The render report for the edition being played.
function shown() {
  const r = S.data.render;
  return r && S.edition === "plain" && r.plain ? { ...r, ...r.plain } : r;
}

function paceBadge(r) {
  const rh = r.rhythm, n = (r.pacing || []).length;
  const tip = `Your pacing, measured from this recording: about `
    + (rh["¶"] ? `${rh["¶"].typ.toFixed(2)} s at the end of a paragraph, ` : "")
    + `${rh["."].typ.toFixed(2)} s after a full stop, `
    + (rh["("] ? `${rh["("].typ.toFixed(2)} s going into or out of an aside, ` : "")
    + `${rh[","].typ.toFixed(2)} s after a comma, ${rh.word.typ.toFixed(2)} s between other words. `
    + `Every pause in the cleaned audio is checked against that range.`;
  return h("span", { class: `tag ${n ? "pending" : "keep"}`, title: tip },
    n ? `${n} pause${n > 1 ? "s" : ""} longer than you read` : "pacing ✓");
}

// ---------- cleaned-chapter player ----------

async function showOutput({ seek = null, play = false } = {}) {
  if (!S || !document.getElementById("outwave")) return;
  S.ows?.destroy();
  S.oaudio?.pause();
  const base = `/api/books/${S.bookId}/chapters/${S.chapterId}/output`;
  const stamp = S.data.render?.rendered || 0;
  const ed = S.data.render?.plain ? S.edition : "extended";
  const peaks = await api(`${base}/peaks?edition=${ed}&t=${stamp}`);
  if (!S || !document.getElementById("outwave")) return;
  S.oaudio = new Audio();
  S.oaudio.preload = "auto";
  S.oaudio.src = `${base}?edition=${ed}&t=${stamp}`;
  S.oregions = RegionsPlugin.create();
  S.ows = WaveSurfer.create({
    container: "#outwave", media: S.oaudio,
    peaks: [peaks.peaks], duration: peaks.duration,
    height: 90, normalize: true,
    waveColor: css("--wave"), progressColor: css("--wave-progress"),
    cursorColor: css("--accent"), cursorWidth: 2,
    minPxPerSec: 1, autoScroll: true, dragToSeek: true,
    plugins: [S.oregions],
  });
  S.ows.once("decode", () => {
    drawJoins();
    if (seek !== null) S.ows.setTime(seek);
    if (play) { S.active = "out"; S.ows.play(); }
  });
  S.ows.on("timeupdate", (t) => {
    const el = document.getElementById("outtime");
    if (el) el.textContent = `${fmtTime(t)} / ${fmtTime(peaks.duration)}`;
  });
  S.ows.on("interaction", (t) => { if (S?.pass) { S.ows.pause(); passSeek(t); } });
  S.ows.on("play", () => { S.active = "out"; S.player?.stop(); S.ws?.pause(); setOutBtn(true); });
  S.ows.on("pause", () => setOutBtn(false));
}

function joinState(j) {
  const v = verdictOf(j);
  if (v) return v === "good" ? "good" : "bad";
  if (j.check && j.check.ok === false) return "bad";
  if (j.level > 0) return "fixed";
  return j.tight ? "tight" : "clean";
}
const JOIN_LABEL = { good: "you marked it good", bad: "marked bad: being rebuilt", fixed: "rebuilt", tight: "tight cut (no silence)", clean: "clean" };

function drawJoins() {
  const joins = (S && shown()?.joins) || [];
  const dur = S?.ows?.getDuration();
  if (!dur) return;
  S.oregions.clearRegions();
  const color = { good: css("--keep-bg"), bad: css("--cut-bg"), fixed: css("--aside-bg"), tight: css("--pending-bg"), clean: css("--gap-bg") };
  for (const j of joins) {
    S.oregions.addRegion({ start: j.start, end: Math.max(j.end, j.start + 0.05), color: color[joinState(j)], drag: false, resize: false });
  }
  const el = document.getElementById("outticks");
  const pace = (shown()?.pacing || []).map((p) => h("span", {
    class: "tick pace", style: `left:${(p.time / dur) * 100}%`,
    title: `${fmtTime(p.time)} “${p.words[0]} … ${p.words[1]}”: ${p.gap.toFixed(2)} s pause, you usually take ${p.usual.toFixed(2)} s here`,
    onclick: () => playFrom(p.time - 2.5),
  }));
  el?.replaceChildren(...pace, ...joins.map((j) => {
    const st = joinState(j);
    return h("span", {
      class: `tick join-${st}`, style: `left:${(j.start / dur) * 100}%`,
      title: `${fmtTime(j.start)} “${j.words[0]} | ${j.words[1]}” — ${JOIN_LABEL[st]}${j.level ? ` (level ${j.level})` : ""}`,
      onclick: () => (S.pass ? passJump(j) : playFrom(j.start - 2.5)),
    });
  }));
}

function setOutBtn(playing) {
  const b = document.getElementById("outplay");
  if (b) b.textContent = playing ? "❚❚ Pause" : "▶ Play";
}
function toggleOut() { if (S?.ows) { S.active = "out"; S.ows.playPause(); } }
function outSeek(d) { if (S?.ows) S.ows.setTime(Math.max(0, S.ows.getCurrentTime() + d)); }
function playFrom(t) { if (!S?.ows) return; S.ows.setTime(Math.max(0, t)); S.active = "out"; S.ows.play(); }
function nextJoin() {
  const t = S?.ows?.getCurrentTime() ?? 0;
  const j = (shown()?.joins || []).find((x) => x.start - 2.5 > t + 0.1);
  if (j) playFrom(j.start - 2.5); else toast("No more cuts after this point");
}

async function reportGlitch() {
  if (!S?.ows) return;
  const t = S.ows.getCurrentTime();
  S.ows.pause();
  const btn = document.getElementById("glitchbtn");
  if (btn) btn.disabled = true;
  const near = (shown()?.joins || []).filter((j) => j.end >= t - 3 && j.start <= t + 0.2);
  S.oregions.addRegion({ start: Math.max(0, t - 3), end: t, color: "rgba(229,72,77,0.35)", drag: false, resize: false,
    content: h("span", {}, near.length ? "fixing…" : "glitch") });
  toast(near.length
    ? `Glitch at ${fmtTime(t)}: rebuilding the cut “${near[near.length - 1].words.join(" | ")}”. It replays by itself in a few seconds.`
    : `Glitch at ${fmtTime(t)}: there's no cut near here, so it's in the original recording. Logging it.`);
  await guard(async () => {
    const edition = S.data.render?.plain ? S.edition : "extended";
    const r = await api(`/api/books/${S.bookId}/chapters/${S.chapterId}/glitch`, { json: { time: t, edition } });
    if (edition === "plain") S.data.render.plain = { ...S.data.render.plain, ...r.render };
    else S.data.render = { ...r.render, plain: S.data.render?.plain };
    if (r.reopened.length) {
      toast("That cut still glitched at the most careful setting, so its flub is back in “To review”. Listen to it there.");
      const view = S.view, b = S.bookId, c = S.chapterId;
      teardownReview();
      await renderReview(view, b, c);
      return;
    }
    if (!r.fixed.length) {
      toast(r.maxed.length
        ? "That cut is already at the most careful setting. Logged it, with the audio, for a closer look."
        : "No cut within 3 s of there, so that sound is in the original recording. Logged it.");
      drawJoins();
      return;
    }
    const j = r.fixed[0];
    toast(`Fixed the cut “${j.words[0]} | ${j.words[1]}”. Replaying it now; press G again if you still hear it.`);
    document.getElementById("renderpanel")?.replaceWith(renderPanel({ seek: r.replay, play: true }));
  });
  if (btn) btn.disabled = false;
}

// Where this chapter is, from raw recording to finished audio: each step
// done (✓), happening now (with its progress), waiting for you, or to come.
function renderProgress() {
  const el = document.getElementById("progress");
  if (!el || !S) return;
  const st = S.data.status, job = st.job;
  const need = S.cands.filter(waiting).length;
  const running = (kinds) => job && kinds.includes(job.kind) ? job : null;
  const machineJob = running(["process", "transcribe", "align", "judge"]);
  const renderJob = running(["render"]);
  const steps = [
    ["Recordings", st.parts ? "done" : "todo", st.parts ? `${st.parts} part${st.parts > 1 ? "s" : ""}` : "none yet"],
    ["Flubs found", machineJob ? "busy" : MACHINE.includes(st.stage) || !st.aligned ? "todo" : "done",
      machineJob ? null : st.aligned ? `${S.cands.length} edit${S.cands.length === 1 ? "" : "s"}` : "not yet", machineJob],
    ["Your review", !st.aligned ? "todo" : need ? "action" : "done", !st.aligned ? "" : need ? `${need} need${need > 1 ? "" : "s"} you` : "nothing needs you"],
    ["Cleaned audio", renderJob ? "busy" : st.stage === "ready" ? "done" : st.stage === "out of date" ? "action" : "todo",
      renderJob ? null : st.stage === "ready" ? statusOf(st).hint : st.stage === "out of date" ? "out of date" : "not rendered", renderJob],
  ];
  el.replaceChildren(h("ol", { class: "steps" }, steps.map(([name, state, note, j]) =>
    h("li", { class: `step s-${state}` },
      h("span", { class: "dot" }, state === "done" ? "✓" : ""),
      h("span", {}, h("b", {}, name), " ",
        j ? statusTag({ job: j }, { book: S.bookId, chapter: S.chapterId }) : h("span", { class: "muted" }, note))))));
}

// After a decision: the cleaned audio is out of date, and once nothing needs
// you the server renders it by itself (resp.render is that job).
function afterChange(resp) {
  const st = S.data.status;
  st.rendered = false;
  st.needs_you = S.cands.filter(waiting).length;
  st.stage = st.needs_you ? "needs you" : S.data.render ? "out of date" : "not rendered";
  if (resp?.render) { st.job = resp.render; st.busy = true; }
  renderProgress();
  pollJobs();
}

// ---------- waveform ----------

async function showPart(k) {
  S.part = k;
  S.player?.stop();
  S.ws?.destroy();
  S.audio?.pause();
  document.querySelectorAll(".tabs button").forEach((b, i) => b.classList.toggle("on", i === k));
  const part = S.data.parts[k];
  const base = `/api/books/${S.bookId}/chapters/${S.chapterId}/parts/${part.id}`;
  const peaks = await api(`${base}/peaks`);
  if (!S || S.part !== k) return;

  S.audio = new Audio();
  S.audio.preload = "auto";
  S.audio.src = `${base}/audio`;
  S.regions = RegionsPlugin.create();
  S.ws = WaveSurfer.create({
    container: "#wave",
    media: S.audio,
    peaks: [peaks.peaks],
    duration: peaks.duration,
    height: 170,
    normalize: true,
    waveColor: css("--wave"),
    progressColor: css("--wave-progress"),
    cursorColor: css("--accent"),
    cursorWidth: 2,
    minPxPerSec: S.zoom,
    autoScroll: true,
    autoCenter: false,
    dragToSeek: true,
    plugins: [
      S.regions,
      TimelinePlugin.create({ height: 18, style: { fontSize: "11px", color: css("--muted") } }),
      MinimapPlugin.create({ container: "#minimap", height: 32, waveColor: css("--wave"),
        progressColor: css("--wave-progress"), overlayColor: "rgba(127,127,127,0.18)" }),
    ],
  });
  S.player = new SegmentPlayer(S.audio);
  S.ws.once("decode", () => drawRegions());
  S.ws.on("timeupdate", (t) => {
    const el = document.getElementById("time");
    if (el) el.textContent = `${fmtTime(t)} / ${fmtTime(peaks.duration)}`;
    highlightWord(t);
  });
  S.ws.on("play", () => { S.active = "src"; S.ows?.pause(); setPlayBtn(true); });
  S.ws.on("pause", () => setPlayBtn(false));
  S.ws.on("interaction", () => S.player.stop());
  S.regions.on("region-clicked", (region, e) => {
    e.stopPropagation();
    const ev = S.cands.find((c) => c.id === region.id);
    if (ev) select(ev, { play: false, zoom: false });
  });
  renderTranscript();
}

function regionStyle(e) {
  if (!e.candidate) return { color: css("--gap-bg") };
  const eff = effective(e);
  if (!e.decision) return { color: css(isAside(e) ? "--aside-bg" : "--pending-bg"), label: (eff === "cut" ? "✂ " : "") + typeLabel(e) };
  if (isAside(e)) return { color: css(eff === "cut" ? "--cut-bg" : "--aside-bg"), label: eff === "cut" ? "✂ aside removed" : "aside · Extended" };
  return { color: css(eff === "cut" ? "--cut-bg" : "--keep-bg"), label: eff === "cut" ? "✂ cut" : "keep" };
}

function drawRegions() {
  if (!S?.regions) return;
  S.regions.clearRegions();
  for (const e of S.data.events) {
    if (e.part !== S.part || !(e.candidate || e.type === "gap")) continue;
    const st = regionStyle(e);
    const content = st.label ? h("span", {}, st.label) : undefined;
    S.regions.addRegion({ id: e.id, start: e.start, end: Math.max(e.end, e.start + 0.05), color: st.color,
      drag: false, resize: false, content });
  }
  if (S.sel && S.sel.part === S.part) zoomTo(S.sel);
  renderTicks();
}

function updateRegion(e) {
  renderTicks();
  const r = S.regions?.getRegions().find((x) => x.id === e.id);
  if (!r) return;
  const st = regionStyle(e);
  r.setOptions({ color: st.color, content: st.label ? h("span", {}, st.label) : undefined });
}

// A strip under the minimap with one tick per flub, so you can see where
// they cluster in a long chapter and jump straight to one.
function renderTicks() {
  const el = document.getElementById("ticks");
  const dur = S?.ws?.getDuration();
  if (!el || !dur) return;
  el.replaceChildren(...S.cands.filter((e) => e.part === S.part).map((e) => {
    const state = e.decision ? effective(e) : "pending";
    return h("span", {
      class: `tick ${state}${e === S.sel ? " sel" : ""}`,
      style: `left:${(e.start / dur) * 100}%`,
      title: `${fmtTime(e.start)} ${TYPE_LABEL[e.type] || e.type}: “${e.text}”`,
      onclick: () => select(e),
    });
  }));
}

function zoomTo(e) {
  if (!S.ws) return;
  const visible = document.getElementById("wave").clientWidth / S.zoom;
  const mid = (e.start + e.end) / 2;
  S.ws.setScrollTime(Math.max(0, mid - visible / 2));
}

// ---------- playback ----------

class SegmentPlayer {
  constructor(media) { this.media = media; this.segs = null; this.raf = 0; }
  play(segs) {
    this.stop();
    this.segs = segs.filter(([a, b]) => b > a);
    if (!this.segs.length) return;
    this.k = 0;
    this.media.currentTime = this.segs[0][0];
    this.media.play();
    const tick = () => {
      if (!this.segs) return;
      const t = this.media.currentTime;
      if (t >= this.segs[this.k][1] - 0.01) {
        this.k += 1;
        if (this.k >= this.segs.length) { this.media.pause(); this.segs = null; return; }
        this.media.currentTime = this.segs[this.k][0];
      }
      this.raf = requestAnimationFrame(tick);
    };
    this.raf = requestAnimationFrame(tick);
  }
  stop() { this.segs = null; cancelAnimationFrame(this.raf); }
}

function setPlayBtn(playing) {
  const b = document.getElementById("playbtn");
  if (b) b.textContent = playing ? "❚❚ Pause" : "▶ Play";
}

function togglePlay() {
  if (!S?.ws) return;
  S.active = "src";
  S.ows?.pause();
  S.player.stop();
  S.ws.playPause();
}

function playCut(e) {
  if (!S.ws || !e) return;
  if (e.part !== S.part) return;
  const dur = S.ws.getDuration();
  S.player.play([[Math.max(0, e.start - S.pre), e.start], [e.end, Math.min(dur, e.end + S.post)]]);
}

function playOriginal(e) {
  if (!S.ws || !e) return;
  if (e.part !== S.part) return;
  const dur = S.ws.getDuration();
  S.player.play([[Math.max(0, e.start - S.pre), Math.min(dur, e.end + S.post)]]);
}

// ---------- selection + decisions ----------

async function select(e, { play = S.autoplay, zoom = true } = {}) {
  S.sel = e;
  if (e.part !== S.part) {
    await showPart(e.part);
  } else if (zoom) {
    zoomTo(e);
  }
  renderList();
  renderDetail();
  renderTicks();
  document.querySelector(`.cand[data-id="${CSS.escape(e.id)}"]`)?.scrollIntoView({ block: "nearest" });
  if (play) setTimeout(() => playCut(e), e.part !== S.part ? 400 : 0);
}

async function decide(e, decision) {
  if (!e) return;
  const value = e.decision === decision ? null : decision;
  await guard(async () => {
    const resp = await api(`/api/books/${S.bookId}/chapters/${S.chapterId}/decisions`, { method: "PUT", json: { [e.id]: value } });
    e.decision = value;
    afterChange(resp);
    updateRegion(e);
    renderTranscript();
    document.getElementById("renderpanel")?.replaceWith(renderPanel());
    if (value && S.autoplay) {
      const next = nextCand(e, true);
      if (next) return select(next, { play: true });
    }
    renderList();
    renderDetail();
  });
}

// Flub or aside: what this extra speech is. Back to the AI's call if it
// already says so.
async function toggleKind(e) {
  if (!e || !judged(e)) return;
  const want = isAside(e) ? "flub" : "aside";
  const ai = e.ai_kind ?? null;
  await guard(async () => {
    const resp = await api(`/api/books/${S.bookId}/chapters/${S.chapterId}/kinds`, { method: "PUT", json: { [e.id]: want === ai ? null : want } });
    const fresh = await api(`/api/books/${S.bookId}/chapters/${S.chapterId}`);
    const f = fresh.events.find((x) => x.id === e.id);
    if (f) { e.kind = f.kind; e.ai_kind = f.ai_kind; e.suggestion = f.suggestion; e.needs_you = f.needs_you; }
    afterChange(resp);
    updateRegion(e);
    renderList(); renderDetail(); renderTranscript();
    document.getElementById("renderpanel")?.replaceWith(renderPanel());
    toast(isAside(e) ? "Marked as an aside: kept in the Extended edition, left out of Plain." : "Marked as a flub.");
  });
}

const waiting = (e) => e.needs_you && !e.decision;

function nextCand(from, undecidedOnly = false, dir = 1) {
  const list = S.cands;
  let i = list.indexOf(from);
  for (let n = 0; n < list.length; n++) {
    i = (i + dir + list.length) % list.length;
    if (!undecidedOnly || waiting(list[i])) return list[i] === from ? null : list[i];
  }
  return null;
}

async function acceptAll() {
  const todo = S.cands.filter((e) => !e.decision);
  if (!todo.length) return toast("Nothing left to decide");
  if (!confirm(`Accept the suggested decision for ${todo.length} undecided flub(s)?`)) return;
  await guard(async () => {
    const changes = Object.fromEntries(todo.map((e) => [e.id, e.suggestion.decision]));
    const resp = await api(`/api/books/${S.bookId}/chapters/${S.chapterId}/decisions`, { method: "PUT", json: changes });
    todo.forEach((e) => { e.decision = e.suggestion.decision; updateRegion(e); });
    afterChange(resp);
    renderList(); renderDetail(); renderTranscript();
    document.getElementById("renderpanel")?.replaceWith(renderPanel());
    toast(`Accepted ${todo.length} suggestion(s)`);
  });
}

function onKey(ev) {
  if (!S || ev.target.closest("input, textarea, select, dialog")) return;
  if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
  const e = S.sel;
  const keys = {
    " ": () => (S.pass ? (S.pass.audio.paused ? S.pass.audio.play() : S.pass.audio.pause())
      : S.active === "out" ? toggleOut() : togglePlay()),
    Escape: () => S.pass && stopPass(),
    g: () => (S.pass || S.active === "out") && rateCut("good"),
    b: () => (S.pass || S.active === "out") && rateCut("bad"),
    n: () => nextJoin(),
    Enter: () => (ev.shiftKey ? playOriginal(e) : playCut(e)),
    p: () => playCut(e),
    o: () => playOriginal(e),
    x: () => decide(e, "cut"),
    c: () => decide(e, "cut"),
    k: () => decide(e, "keep"),
    a: () => toggleKind(e),
    ArrowRight: () => { const n = nextCand(e, S.filter === "todo"); if (n) select(n); },
    ArrowLeft: () => { const n = nextCand(e, S.filter === "todo", -1); if (n) select(n); },
    j: () => { const n = nextCand(e, S.filter === "todo"); if (n) select(n); },
    ArrowDown: () => { const n = nextCand(e, false); if (n) select(n); },
    ArrowUp: () => { const n = nextCand(e, false, -1); if (n) select(n); },
    Backspace: () => e?.decision && decide(e, e.decision),
  };
  if (S.pass) Object.assign(keys, { ArrowRight: () => passStep(1), ArrowLeft: () => passStep(-1) });
  const fn = keys[ev.key];
  if (fn) { ev.preventDefault(); fn(); }
}

// ---------- side list + detail ----------

function decisionTag(e) {
  const word = (d) => (isAside(e) ? (d === "cut" ? "removed" : "extended") : d);
  if (e.decision) return h("span", { class: `tag ${isAside(e) && e.decision === "keep" ? "aside" : e.decision}` }, word(e.decision));
  if (!e.needs_you) return h("span", { class: `tag ${isAside(e) && e.suggestion.decision === "keep" ? "aside" : e.suggestion.decision}`, title: "Accepted as suggested; change it if you disagree" }, `auto ${word(e.suggestion.decision)}`);
  return h("span", { class: "tag pending", title: "Not sure: needs you" }, `${word(e.suggestion.decision)}?`);
}

function renderList() {
  const el = document.getElementById("candlist");
  if (!el) return;
  document.querySelectorAll("[data-filter]").forEach((b) => (b.className = "small " + (b.dataset.filter === S.filter ? "primary" : "ghost")));
  const todoBtn = document.querySelector('[data-filter="todo"]');
  if (todoBtn) todoBtn.textContent = `Needs you (${S.cands.filter(waiting).length})`;
  if (S.filter === "notes") {
    el.replaceChildren(...(S.notes.length ? S.notes.map((n) => h("div", { class: "cand", onclick: () => n.start !== undefined && seekTo(n.part, n.start) },
      h("div", { class: "meta" }, h("span", { class: "tag type" }, n.type === "sub" ? "different word" : "skipped"),
        n.start !== undefined ? h("span", { class: "mono" }, `${S.data.parts.length > 1 ? `P${n.part + 1} ` : ""}${fmtTime(n.start)}`) : null),
      h("div", { class: "said" }, n.type === "sub" ? `said “${n.said}”, book has “${n.book}”` : `“${n.book}” not read`)))
      : [h("div", { class: "empty small" }, "The recording matches the book word for word.")]));
    return;
  }
  const list = S.filter === "todo" ? S.cands.filter((e) => waiting(e) || e === S.sel) : S.cands;
  if (!list.length) {
    el.replaceChildren(h("div", { class: "empty" }, S.cands.length ? "Nothing needs you here. 🎉" : "No flubs found in this chapter.",
      S.cands.length ? h("div", { class: "small", style: "margin-top:8px" },
        `The other ${S.cands.length} edits are accepted as suggested (see All). Render the chapter below to hear the result.`) : null));
    return;
  }
  el.replaceChildren(...list.map((e) => h("div", {
    class: "cand" + (e === S.sel ? " sel" : ""), "data-id": e.id, onclick: () => select(e) },
    h("div", { class: "meta" },
      decisionTag(e),
      h("span", { class: "tag " + (isAside(e) ? "aside" : "type") }, typeLabel(e)),
      e.flag ? h("span", { class: "tag cut", title: e.flag }, "glitch") : null,
      h("span", { class: "spacer" }),
      h("span", { class: "mono" }, `${S.data.parts.length > 1 ? `P${e.part + 1} ` : ""}${fmtTime(e.start)}`)),
    h("div", { class: "said" }, `“${e.text}”`),
    h("div", { class: "why" }, `${e.suggestion.by === "ai" ? "AI" : e.suggestion.by === "rule" ? "Rule" : "Default"}: ${e.suggestion.reason}`))));
}

function renderDetail() {
  const el = document.getElementById("detail");
  if (!el) return;
  const e = S.sel;
  if (!e) {
    el.replaceChildren(h("div", { class: "empty" }, S.cands.length ? "Select a flub to review it." : "Nothing flagged in this chapter."));
    return;
  }
  const eff = effective(e);
  const words = S.data.parts[e.part].words;
  // A noise has no words: its context is either side of the pause.
  const [i0, i1] = e.words.length ? [e.words[0], e.words[e.words.length - 1]] : [e.between[1], e.between[0]];
  const first = words.findIndex((w) => w.i === i0);
  const last = words.findIndex((w) => w.i === i1);
  const before = words.slice(Math.max(0, first - 18), first).map((w) => w.word).join(" ");
  const after = words.slice(last + 1, last + 19).map((w) => w.word).join(" ");
  const sug = e.suggestion;
  el.replaceChildren(
    h("div", { class: "panel-head" },
      h("span", { class: "tag " + (isAside(e) ? "aside" : "type") }, typeLabel(e)),
      h("h2", {}, `${fmtTime(e.start)} – ${fmtTime(e.end)}`),
      h("span", { class: "muted small" }, `${(e.end - e.start).toFixed(1)}s`),
      h("span", { class: "spacer" }),
      h("button", { onclick: () => playCut(e), title: "Pre-roll, skip the flub, post-roll (P or Enter)" }, "▶ Play with cut"),
      h("button", { onclick: () => playOriginal(e), title: "Pre-roll, the flub, post-roll (O or Shift+Enter)" }, "▶ Play original"),
      judged(e) ? h("button", { class: "ghost", onclick: () => toggleKind(e),
        title: "Is this a mistake, or talk that was meant (a chat, a joke)? Asides go in the Extended edition only. (A)" },
        isAside(e) ? "Mark as flub" : "Mark as aside") : null,
      h("button", { class: "cut" + (e.decision === "cut" ? " on" : ""), onclick: () => decide(e, "cut") },
        isAside(e) ? "✂ Remove" : "✂ Cut it"),
      h("button", { class: "keep" + (e.decision === "keep" ? " on" : ""), onclick: () => decide(e, "keep") },
        isAside(e) ? "✓ Keep in Extended" : "✓ Keep it")),
    h("div", { class: "panel-body stack detail" },
      h("div", { class: "small" },
        h("b", {}, sug.by === "ai" ? "AI suggests " : sug.by === "rule" ? "Rule says " : "Default: "),
        h("span", { class: `tag ${isAside(e) && sug.decision === "keep" ? "aside" : sug.decision}` },
          isAside(e) ? (sug.decision === "keep" ? "keep in Extended" : "remove") : sug.decision),
        sug.confidence ? h("span", { class: "muted" }, ` ${Math.round(sug.confidence * 100)}% sure`) : null,
        " — ", sug.reason,
        e.decision ? h("span", { class: "muted" }, ` · You chose to ${e.decision} it.`) : null),
      h("div", {},
        h("div", { class: "small muted" }, "What you said"),
        h("div", { class: "ctx" }, before ? `…${before} ` : "", h("span", { class: `said ${e.decision ? eff : ""}` }, e.text), after ? ` ${after}…` : "")),
      h("div", {},
        h("div", { class: "small muted" }, "The book says"),
        h("div", { class: "ctx muted" }, e.context.book_before ? `…${e.context.book_before} ` : "", h("span", { class: "caret" }, "▲"), e.context.book_after ? ` ${e.context.book_after}…` : "")),
      h("div", { class: "keys" },
        h("span", {}, h("kbd", {}, "P"), " play with cut"), h("span", {}, h("kbd", {}, "O"), " original"),
        h("span", {}, h("kbd", {}, "X"), " cut"), h("span", {}, h("kbd", {}, "K"), " keep"),
        judged(e) ? h("span", {}, h("kbd", {}, "A"), " aside / flub") : null,
        h("span", {}, h("kbd", {}, "←"), h("kbd", {}, "→"), " prev/next"), h("span", {}, h("kbd", {}, "Space"), " play/pause"),
        h("span", {}, h("kbd", {}, "⌫"), " undo decision"),
        h("span", {}, h("kbd", {}, "G"), " glitch in cleaned audio"), h("span", {}, h("kbd", {}, "N"), " next cut"))));
}

// ---------- transcript ----------

function renderTranscript() {
  const el = document.getElementById("transcript");
  if (!el) return;
  const words = S.data.parts[S.part]?.words || [];
  if (!words.length) { el.replaceChildren(h("span", { class: "muted" }, "No transcript for this part yet.")); return; }
  S.wordEls = [];
  const frag = document.createDocumentFragment();
  for (const w of words) {
    const e = S.byWord.get(w.i);
    let cls = "";
    if (e) cls = (effective(e) === "cut" ? "x" : "k") + (e.decision ? "" : " p");
    const span = h("span", { class: cls, title: fmtTime(w.start), onclick: () => {
      if (e) select(e, { play: false });
      seekTo(S.part, w.start);
    } }, w.word);
    S.wordEls.push([w.start, span]);
    frag.append(span, " ");
  }
  el.replaceChildren(frag);
}

let lastWord = null;
function highlightWord(t) {
  if (!S?.wordEls?.length) return;
  let lo = 0, hi = S.wordEls.length - 1;
  while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (S.wordEls[mid][0] <= t) lo = mid; else hi = mid - 1; }
  const el = S.wordEls[lo][1];
  if (el === lastWord) return;
  lastWord?.classList.remove("now");
  el.classList.add("now");
  lastWord = el;
}

async function seekTo(part, t) {
  if (part !== S.part) await showPart(part);
  S.player?.stop();
  S.ws?.setTime(t);
}
