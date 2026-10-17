import WaveSurfer from "/static/vendor/wavesurfer/wavesurfer.esm.js";
import RegionsPlugin from "/static/vendor/wavesurfer/plugins/regions.esm.js";
import TimelinePlugin from "/static/vendor/wavesurfer/plugins/timeline.esm.js";
import MinimapPlugin from "/static/vendor/wavesurfer/plugins/minimap.esm.js";
import { api, fmtTime, guard, h, setCrumbs, stagePill, submitJob, toast } from "./app.js";

const TYPE_LABEL = { restart: "Restart", filler: "Filler", extra: "Extra speech" };
const MACHINE = ["to transcribe", "to align", "to judge"];

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
  document.removeEventListener("keydown", S.onKey);
  S.player?.stop();
  S.ws?.destroy();
  S.audio?.pause();
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
    pre: pref("pre", 3), post: pref("post", 3),
    zoom: pref("zoom", 40), autoplay: pref("autoplay", true),
    onKey: null,
  };
  S.cands = data.events.filter((e) => e.candidate);
  S.notes = data.events.filter((e) => e.type === "sub" || e.type === "missing");
  S.byWord = new Map();
  for (const e of S.cands) for (const w of e.words) S.byWord.set(w, e);

  build();

  const first = S.cands.find((e) => !e.decision) || S.cands[0];
  if (first) select(first, { play: false });

  // Called when background jobs change: rebuild only if this chapter changed.
  return async () => {
    if (!S || S.chapterId !== chapterId) return;
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
        stagePill(st),
        data.coverage !== null ? h("span", {}, `${Math.round(data.coverage * 100)}% of book text matched`) : null,
        S.notes.length ? h("span", {}, `${S.notes.length} word-level differences`) : null)),
    h("span", { class: "spacer" }),
    h("div", { id: "progress", style: "min-width:220px" }),
    MACHINE.includes(st.stage) && !st.busy
      ? h("button", { class: "primary", onclick: () => guard(() => submitJob("process", S.bookId, S.chapterId)) }, "Process chapter")
      : null);

  if (!st.aligned) {
    S.view.replaceChildren(h("div", { class: "stack" }, header,
      h("div", { class: "panel empty" },
        st.busy ? "Working on this chapter… this page updates when it's ready."
          : !data.parts.length ? h("span", {}, "This chapter has no recordings yet. ", h("a", { href: `#/book/${S.bookId}` }, "Add one on the book page"), ".")
            : "Not processed yet. Press “Process chapter” to transcribe and align it.")));
    return;
  }

  // Waveform panel
  const tabs = data.parts.length > 1 ? h("div", { class: "tabs" },
    data.parts.map((p, k) => h("button", { class: k === S.part ? "on" : "", onclick: () => showPart(k) },
      `Part ${k + 1}`, !p.transcribed ? " (not transcribed)" : ""))) : null;

  const zoom = h("input", { type: "range", min: 5, max: 400, value: S.zoom, style: "width:140px",
    oninput: () => { S.zoom = +zoom.value; setPref("zoom", S.zoom); S.ws?.zoom(S.zoom); } });
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
    renderPanel(),
  );

  right.append(
    h("div", { class: "panel-head" },
      h("h2", {}, "Flubs"),
      h("span", { class: "spacer" }),
      h("label", { class: "inline", title: "After a decision, jump to the next flub and play it" }, auto, "auto-advance")),
    h("div", { class: "row", style: "padding:8px 14px;border-bottom:1px solid var(--line)" },
      ["todo", "all", "notes"].map((f) => h("button", { class: "small " + (S.filter === f ? "primary" : "ghost"), "data-filter": f,
        onclick: () => { S.filter = f; setPref("filter", f); renderList(); } },
        f === "todo" ? "To review" : f === "all" ? "All" : "Word diffs"))),
    h("div", { id: "candlist" }),
    h("div", { style: "padding:12px 14px" },
      h("button", { onclick: acceptAll, title: "Take the suggested decision for every flub you haven't decided" }, "Accept all suggestions")));

  S.view.replaceChildren(h("div", { class: "stack" }, header, h("div", { class: "review" }, left, right)));

  S.onKey = onKey;
  document.addEventListener("keydown", S.onKey);
  renderProgress();
  renderList();
  renderDetail();
  showPart(S.part);
}

function renderPanel() {
  const { data } = S;
  const r = data.render;
  const out = `/api/books/${S.bookId}/chapters/${S.chapterId}/output`;
  return h("section", { class: "panel", id: "renderpanel" },
    h("div", { class: "panel-head" },
      h("h2", {}, "Cleaned chapter"),
      r ? h("span", { class: "small muted" }, `${fmtTime(r.input_seconds)} → ${fmtTime(r.output_seconds)} (${fmtTime(r.input_seconds - r.output_seconds)} removed)`) : null,
      h("span", { class: "spacer" }),
      data.status.rendered ? h("a", { class: "btn", href: out, download: true }, "Download WAV") : null,
      h("button", { class: data.status.stage === "to render" ? "primary" : "", disabled: data.status.busy,
        onclick: () => guard(() => submitJob("render", S.bookId, S.chapterId)) },
        data.status.rendered ? "Re-render" : "Render")),
    h("div", { class: "panel-body" },
      data.status.rendered
        ? h("audio", { controls: true, preload: "none", src: `${out}?t=${r?.rendered || 0}`, style: "width:100%" })
        : h("div", { class: "small muted" },
            r ? "Decisions changed since the last render. Render again to hear them."
              : "Render to produce the cleaned audio. Undecided flubs use the suggested decision.")));
}

function renderProgress() {
  const el = document.getElementById("progress");
  if (!el) return;
  const n = S.cands.length, done = S.cands.filter((e) => e.decision).length;
  el.replaceChildren(
    h("div", { class: "small muted", style: "text-align:right" }, `Reviewed ${done} of ${n}`),
    h("div", { class: "bar amber", style: "margin-top:4px" }, h("span", { style: `width:${n ? (done / n) * 100 : 100}%` })));
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
  S.ws.on("play", () => setPlayBtn(true));
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
  if (!e.decision) return { color: css("--pending-bg"), label: (eff === "cut" ? "✂ " : "") + (TYPE_LABEL[e.type] || e.type) };
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
  const minZoom = 80;
  if (S.zoom < minZoom) {
    S.zoom = minZoom;
    const z = document.querySelector('input[type=range]');
    if (z) z.value = minZoom;
    S.ws.zoom(minZoom);
  }
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
    await api(`/api/books/${S.bookId}/chapters/${S.chapterId}/decisions`, { method: "PUT", json: { [e.id]: value } });
    e.decision = value;
    S.data.status.rendered = false;
    updateRegion(e);
    renderProgress();
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

function nextCand(from, undecidedOnly = false, dir = 1) {
  const list = S.cands;
  let i = list.indexOf(from);
  for (let n = 0; n < list.length; n++) {
    i = (i + dir + list.length) % list.length;
    if (!undecidedOnly || !list[i].decision) return list[i] === from ? null : list[i];
  }
  return null;
}

async function acceptAll() {
  const todo = S.cands.filter((e) => !e.decision);
  if (!todo.length) return toast("Nothing left to decide");
  if (!confirm(`Accept the suggested decision for ${todo.length} undecided flub(s)?`)) return;
  await guard(async () => {
    const changes = Object.fromEntries(todo.map((e) => [e.id, e.suggestion.decision]));
    await api(`/api/books/${S.bookId}/chapters/${S.chapterId}/decisions`, { method: "PUT", json: changes });
    todo.forEach((e) => { e.decision = e.suggestion.decision; updateRegion(e); });
    S.data.status.rendered = false;
    renderProgress(); renderList(); renderDetail(); renderTranscript();
    document.getElementById("renderpanel")?.replaceWith(renderPanel());
    toast(`Accepted ${todo.length} suggestion(s)`);
  });
}

function onKey(ev) {
  if (!S || ev.target.closest("input, textarea, select, dialog")) return;
  if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
  const e = S.sel;
  const keys = {
    " ": () => togglePlay(),
    Enter: () => (ev.shiftKey ? playOriginal(e) : playCut(e)),
    p: () => playCut(e),
    o: () => playOriginal(e),
    x: () => decide(e, "cut"),
    c: () => decide(e, "cut"),
    k: () => decide(e, "keep"),
    ArrowRight: () => { const n = nextCand(e, S.filter === "todo"); if (n) select(n); },
    ArrowLeft: () => { const n = nextCand(e, S.filter === "todo", -1); if (n) select(n); },
    j: () => { const n = nextCand(e, S.filter === "todo"); if (n) select(n); },
    ArrowDown: () => { const n = nextCand(e, false); if (n) select(n); },
    ArrowUp: () => { const n = nextCand(e, false, -1); if (n) select(n); },
    Backspace: () => e?.decision && decide(e, e.decision),
  };
  const fn = keys[ev.key];
  if (fn) { ev.preventDefault(); fn(); }
}

// ---------- side list + detail ----------

function decisionTag(e) {
  if (e.decision) return h("span", { class: `tag ${e.decision}` }, e.decision === "cut" ? "cut" : "keep");
  return h("span", { class: "tag pending", title: "Suggested; not reviewed yet" }, `${e.suggestion.decision}?`);
}

function renderList() {
  const el = document.getElementById("candlist");
  if (!el) return;
  document.querySelectorAll("[data-filter]").forEach((b) => (b.className = "small " + (b.dataset.filter === S.filter ? "primary" : "ghost")));
  if (S.filter === "notes") {
    el.replaceChildren(...(S.notes.length ? S.notes.map((n) => h("div", { class: "cand", onclick: () => n.start !== undefined && seekTo(n.part, n.start) },
      h("div", { class: "meta" }, h("span", { class: "tag type" }, n.type === "sub" ? "different word" : "skipped"),
        n.start !== undefined ? h("span", { class: "mono" }, `${S.data.parts.length > 1 ? `P${n.part + 1} ` : ""}${fmtTime(n.start)}`) : null),
      h("div", { class: "said" }, n.type === "sub" ? `said “${n.said}”, book has “${n.book}”` : `“${n.book}” not read`)))
      : [h("div", { class: "empty small" }, "The recording matches the book word for word.")]));
    return;
  }
  const list = S.filter === "todo" ? S.cands.filter((e) => !e.decision || e === S.sel) : S.cands;
  if (!list.length) {
    el.replaceChildren(h("div", { class: "empty" }, S.cands.length ? "All flubs reviewed. 🎉" : "No flubs found in this chapter.",
      S.cands.length ? h("div", { class: "small", style: "margin-top:8px" }, "Render the chapter below to hear the result.") : null));
    return;
  }
  el.replaceChildren(...list.map((e) => h("div", {
    class: "cand" + (e === S.sel ? " sel" : ""), "data-id": e.id, onclick: () => select(e) },
    h("div", { class: "meta" },
      decisionTag(e),
      h("span", { class: "tag type" }, TYPE_LABEL[e.type] || e.type),
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
  const first = words.findIndex((w) => w.i === e.words[0]);
  const last = words.findIndex((w) => w.i === e.words[e.words.length - 1]);
  const before = words.slice(Math.max(0, first - 18), first).map((w) => w.word).join(" ");
  const after = words.slice(last + 1, last + 19).map((w) => w.word).join(" ");
  const sug = e.suggestion;
  el.replaceChildren(
    h("div", { class: "panel-head" },
      h("span", { class: "tag type" }, TYPE_LABEL[e.type] || e.type),
      h("h2", {}, `${fmtTime(e.start)} – ${fmtTime(e.end)}`),
      h("span", { class: "muted small" }, `${(e.end - e.start).toFixed(1)}s`),
      h("span", { class: "spacer" }),
      h("button", { onclick: () => playCut(e), title: "Pre-roll, skip the flub, post-roll (P or Enter)" }, "▶ Play with cut"),
      h("button", { onclick: () => playOriginal(e), title: "Pre-roll, the flub, post-roll (O or Shift+Enter)" }, "▶ Play original"),
      h("button", { class: "cut" + (e.decision === "cut" ? " on" : ""), onclick: () => decide(e, "cut") }, "✂ Cut it"),
      h("button", { class: "keep" + (e.decision === "keep" ? " on" : ""), onclick: () => decide(e, "keep") }, "✓ Keep it")),
    h("div", { class: "panel-body stack detail" },
      h("div", { class: "small" },
        h("b", {}, sug.by === "ai" ? "AI suggests " : sug.by === "rule" ? "Rule says " : "Default: "),
        h("span", { class: `tag ${sug.decision}` }, sug.decision),
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
        h("span", {}, h("kbd", {}, "←"), h("kbd", {}, "→"), " prev/next"), h("span", {}, h("kbd", {}, "Space"), " play/pause"),
        h("span", {}, h("kbd", {}, "⌫"), " undo decision"))));
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
