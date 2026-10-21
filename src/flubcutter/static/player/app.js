// Story Shelf: the household player for finished Flubcutter chapters.
const $ = (id) => document.getElementById(id);
const audio = $("audio");

const JEWELS = ["var(--sapphire)", "var(--emerald)", "var(--ruby)", "var(--amber)", "var(--amethyst)"];
const EDITIONS = [
  { key: "extended", name: "With our chat", hint: "The story plus everything we said along the way" },
  { key: "plain", name: "Story only", hint: "Just the book" },
];
const SPEEDS = [1, 1.25, 1.5, 0.75];
const SLEEPS = [0, 15, 30, 45, 60, "end"];
const PLAYED_TAIL = 30; // within 30 seconds of the end counts as played

let books = [];
let state = { chapters: {}, books: {}, last: null, speed: 1 };
let view = null;      // "continue" or a book id
let current = null;   // { book, ch, ed }
let dirty = { chapters: {}, books: {} };
let edition = "extended";
let sleep = { i: 0, at: null };

// ---------- formatting ----------
const fmt = (s) => {
  s = Math.max(0, Math.floor(s));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  return h ? `${h}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}` : `${m}:${String(sec).padStart(2, "0")}`;
};
const human = (s) => {
  const h = Math.floor(s / 3600), m = Math.max(1, Math.round((s % 3600) / 60));
  return h ? `${h} h ${m} min` : `${m} min`;
};
const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const bookOf = (id) => books.find((b) => b.id === id);
const jewel = (b) => JEWELS[books.indexOf(b) % JEWELS.length];
const key = (b, ch) => `${b.id}/${ch.id}`;
const numLabel = (ch) => ch.nums.length > 1 ? `${ch.nums[0]}–${ch.nums.at(-1)}` : ch.nums.length ? String(ch.nums[0]) : String(ch.index);
const ready = (ch) => Object.keys(ch.editions).length > 0;
const edFor = (ch) => ch.editions[edition] ? edition : Object.keys(ch.editions)[0];
const initials = (t) => t.replace(/^(the|a|an) /i, "").split(/[\s:]+/).filter((w) => /^[A-Za-z]/.test(w) && !/^(and|of|the|in|a)$/i.test(w))
  .slice(0, 2).map((w) => w[0].toUpperCase()).join("");
const cover = (b, cls = "") => `<div class="cover ${cls}" style="--c:${jewel(b)}"><span>${esc(initials(b.title))}</span></div>`;

// ---------- state ----------
const chState = (b, ch) => state.chapters[key(b, ch)] || {};
function setCh(b, ch, patch) {
  const k = key(b, ch);
  state.chapters[k] = { ...(state.chapters[k] || {}), ...patch };
  dirty.chapters[k] = state.chapters[k];
}
function setBook(b, patch) {
  state.books[b.id] = { ...(state.books[b.id] || {}), ...patch };
  dirty.books[b.id] = state.books[b.id];
}
function save(extra = {}) {
  const body = { ...dirty, ...extra };
  dirty = { chapters: {}, books: {} };
  return fetch("/api/state", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })
    .catch(() => {});
}
const posOf = (b, ch, ed) => (chState(b, ch).pos || {})[ed] || 0;
const progress = (b, ch) => {
  const s = chState(b, ch), ed = edFor(ch);
  if (s.played) return 1;
  return ed ? Math.min(1, posOf(b, ch, ed) / ch.editions[ed].duration) : 0;
};
const started = (b, ch) => !chState(b, ch).played && ready(ch) && posOf(b, ch, edFor(ch)) > 20;
const readyChapters = (b) => b.chapters.filter(ready);

// Where to pick a book up: the chapter last listened to, else the first unplayed ready one.
function resumeOf(b) {
  const last = b.chapters.find((c) => c.id === state.books[b.id]?.last);
  if (last && ready(last) && !chState(b, last).played) return last;
  const i = last ? b.chapters.indexOf(last) : -1;
  return b.chapters.slice(i + 1).find((c) => ready(c) && !chState(b, c).played)
    || b.chapters.find((c) => ready(c) && !chState(b, c).played);
}
const inProgress = () => books.filter((b) => state.books[b.id]?.at && resumeOf(b))
  .sort((x, y) => state.books[y.id].at - state.books[x.id].at);

// ---------- sidebar ----------
function renderSidebar() {
  const html = [];
  const cont = inProgress();
  if (cont.length) {
    html.push(`<button class="group" data-key="continue" style="--c:var(--amber)" ${view === "continue" ? 'aria-current="true"' : ""}>
      <span class="pane"></span><span><span class="group-name">Continue listening</span>
      <span class="group-sub">Pick up where you left off</span></span><span class="group-count">${cont.length}</span></button>`);
  }
  html.push(`<div class="side-label">Books</div>`);
  for (const b of books) {
    const rc = readyChapters(b), played = rc.filter((c) => chState(b, c).played).length;
    html.push(`<button class="group" data-key="${b.id}" style="--c:${jewel(b)}" ${view === b.id ? 'aria-current="true"' : ""}>
      <span class="pane"></span>
      <span><span class="group-name">${esc(b.title)}</span><span class="group-sub">${rc.length} of ${b.chapters.length} chapters ready</span>
      <div class="meter"><i style="width:${rc.length ? (played / rc.length) * 100 : 0}%"></i></div></span>
      <span class="group-count">${played}/${rc.length}</span></button>`);
  }
  $("sidebar").innerHTML = html.join("");
}
$("sidebar").addEventListener("click", (e) => {
  const g = e.target.closest(".group");
  if (g) select(g.dataset.key);
});

function select(k) {
  view = k;
  try { localStorage.setItem("view", k); } catch {}
  renderSidebar();
  renderMain();
  $("main").scrollTop = 0;
}

// ---------- main ----------
function editionSwitch() {
  return `<div class="seg" role="radiogroup" aria-label="Edition">${EDITIONS.map((e) =>
    `<button role="radio" aria-checked="${e.key === edition}" data-ed="${e.key}" title="${esc(e.hint)}">${esc(e.name)}</button>`).join("")}</div>`;
}

function rowHtml(b, ch) {
  const s = chState(b, ch), ed = edFor(ch), p = progress(b, ch);
  const isCur = current && current.book === b && current.ch === ch;
  if (!ready(ch)) {
    return `<div class="row soon" style="--c:${jewel(b)}"><div class="num">${numLabel(ch)}</div>
      <div><div class="ch-title">${esc(ch.title)}</div><div class="ch-meta">Still being edited</div></div><div class="ch-time"></div><span></span></div>`;
  }
  const dur = ch.editions[ed].duration, pos = posOf(b, ch, ed);
  const left = s.played ? "Played" : pos > 20 ? `${human(dur - pos)} left` : human(dur);
  const other = EDITIONS.find((e) => e.key !== ed);
  const meta = !ch.editions[edition] ? `Only “${EDITIONS.find((e) => e.key === ed).name}” so far`
    : ch.editions[other.key] && edition === "extended" ? `${human(Math.max(0, dur - ch.editions[other.key].duration))} of chat` : "";
  return `<div class="row ${s.played ? "played" : ""} ${isCur ? "current" : ""}" data-book="${b.id}" data-ch="${ch.id}" style="--c:${jewel(b)}">
    <div class="num">${isCur && !audio.paused ? `<span class="eq"><i></i><i></i><i></i></span>` : numLabel(ch)}</div>
    <div><div class="ch-title">${esc(ch.title)}</div><div class="ch-meta">${meta}</div>
      ${p > 0 && p < 1 ? `<div class="bar"><i style="width:${p * 100}%"></i></div>` : ""}</div>
    <div class="ch-time">${left}</div>
    <button class="mark" title="${s.played ? "Mark not played" : "Mark played"}" aria-label="${s.played ? "Mark not played" : "Mark played"}">
      <svg viewBox="0 0 24 24"><path d="M9 16.2 4.8 12l-1.4 1.4L9 19 21 7l-1.4-1.4z"/></svg></button>
  </div>`;
}

function bookHead(b) {
  const rc = readyChapters(b), r = resumeOf(b);
  const total = rc.reduce((t, c) => t + c.editions[edFor(c)].duration, 0);
  const resuming = r && posOf(b, r, edFor(r)) > 20;
  const action = r ? `<button class="big" data-play="${b.id}/${r.id}">
      <svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>${resuming ? "Resume" : "Play"} chapter ${numLabel(r)}</button>`
    : rc.length ? `<span class="done">All ready chapters played</span>` : `<span class="done">No chapters ready yet</span>`;
  return `<section class="book-head" style="--c:${jewel(b)}">${cover(b, "large")}
    <div class="book-info"><h1>${esc(b.title)}</h1>
      <div class="book-meta">${esc(b.author)}${b.author ? " · " : ""}${rc.length} of ${b.chapters.length} chapters ready${total ? ` · ${human(total)}` : ""}</div>
      <div class="book-actions">${action}${editionSwitch()}</div></div></section>`;
}

function renderMain() {
  if (!books.length) {
    $("main").innerHTML = `<div class="empty">No books yet. Finished chapters from Flubcutter show up here.</div>`;
    return;
  }
  if (view === "continue") {
    $("main").innerHTML = `<div class="toolbar"><h1>Continue listening</h1>${editionSwitch()}</div>` +
      inProgress().map((b) => { const r = resumeOf(b); return `<section class="cont" style="--c:${jewel(b)}">
        <div class="cont-head">${cover(b, "small")}<div><h2>${esc(b.title)}</h2>
        <div class="book-meta">Up next: chapter ${numLabel(r)}</div></div></div>${rowHtml(b, r)}</section>`; }).join("");
    return;
  }
  const b = bookOf(view);
  $("main").innerHTML = bookHead(b) + `<div class="chapters">${b.chapters.map((c) => rowHtml(b, c)).join("")}</div>`;
}

$("main").addEventListener("click", (e) => {
  const seg = e.target.closest(".seg button");
  if (seg) return setEdition(seg.dataset.ed);
  const big = e.target.closest("[data-play]");
  if (big) {
    const [bid, cid] = big.dataset.play.split("/");
    const b = bookOf(bid);
    return play(b, b.chapters.find((c) => c.id === cid));
  }
  const row = e.target.closest(".row[data-ch]");
  if (!row) return;
  const b = bookOf(row.dataset.book), ch = b.chapters.find((c) => c.id === row.dataset.ch);
  if (e.target.closest(".mark")) {
    const played = !chState(b, ch).played;
    setCh(b, ch, { played, pos: played ? {} : chState(b, ch).pos });
    save();
    return refresh();
  }
  if (current && current.book === b && current.ch === ch) togglePlay();
  else play(b, ch);
});

function refresh() { renderSidebar(); renderMain(); }

// ---------- edition ----------
function setEdition(ed) {
  if (ed === edition) return;
  edition = ed;
  try { localStorage.setItem("edition", ed); } catch {}
  if (current && current.ch.editions[ed] && current.ed !== ed) {
    // Same moment in the other edition, roughly: scale by length.
    const from = current.ch.editions[current.ed].duration, to = current.ch.editions[ed].duration;
    const t = audio.currentTime * (to / from), wasPlaying = !audio.paused;
    persist();
    load(current.book, current.ch, ed, t, wasPlaying);
  }
  refresh();
}

// ---------- playback ----------
function play(b, ch) {
  const ed = edFor(ch);
  const s = chState(b, ch);
  if (s.played) setCh(b, ch, { played: false, pos: {} });
  load(b, ch, ed, s.played ? 0 : posOf(b, ch, ed), true);
}

function load(b, ch, ed, start, autoplay) {
  if (current) persist();
  current = { book: b, ch, ed };
  audio.src = `/audio/${b.id}/${ch.id}/${ed}`;
  audio.playbackRate = state.speed || 1;
  showTime(start);
  audio.addEventListener("loadedmetadata", () => {
    audio.currentTime = Math.max(0, Math.min(start, audio.duration - 3));
    audio.playbackRate = state.speed || 1;
    showTime();
    if (autoplay) audio.play().catch(() => {});
  }, { once: true });
  $("player").hidden = false;
  $("npCover").outerHTML = cover(b, "small").replace('class="cover small"', 'id="npCover" class="cover small"');
  $("npTitle").textContent = `${numLabel(ch)}. ${ch.title}`;
  $("npBook").textContent = `${b.title} · ${EDITIONS.find((e) => e.key === ed).name}`;
  document.title = `${ch.title} · Story Shelf`;
  setBook(b, { last: ch.id, at: Date.now() });
  save({ last: `${b.id}/${ch.id}` });
  if ("mediaSession" in navigator) {
    navigator.mediaSession.metadata = new MediaMetadata({
      title: `${numLabel(ch)}. ${ch.title}`, artist: b.author || "Read aloud at home", album: b.title,
      artwork: [{ src: "/static/icon.svg", sizes: "any", type: "image/svg+xml" }],
    });
  }
  refresh();
}

function togglePlay() {
  if (!current) return;
  audio.paused ? audio.play().catch(() => {}) : audio.pause();
}
const skip = (d) => { if (current) audio.currentTime = Math.max(0, Math.min(audio.duration || 0, audio.currentTime + d)); };

function persist() {
  if (!current || !audio.duration) return;
  const { book: b, ch, ed } = current, t = audio.currentTime;
  const played = audio.duration - t < PLAYED_TAIL;
  setCh(b, ch, { pos: played ? {} : { ...(chState(b, ch).pos || {}), [ed]: Math.round(t) }, played, at: Date.now() });
  setBook(b, { last: ch.id, at: Date.now() });
  save();
}

const nextOf = ({ book: b, ch }) => b.chapters.slice(b.chapters.indexOf(ch) + 1).find(ready);

audio.addEventListener("play", () => { $("player").classList.add("playing"); renderMain(); });
audio.addEventListener("pause", () => { $("player").classList.remove("playing"); persist(); refresh(); });
audio.addEventListener("ended", () => {
  persist();
  if (sleep.at === "end") { setSleep(0); return refresh(); }
  const n = nextOf(current);
  if (n) play(current.book, n); else refresh();
});

let lastSave = 0, seeking = false;
function showTime(t = audio.currentTime) {
  const d = audio.duration || (current && current.ch.editions[current.ed]?.duration) || 0;
  $("cur").textContent = fmt(t);
  $("left").textContent = "-" + fmt(d - t);
  if (!seeking) {
    $("seek").value = d ? (t / d) * 1000 : 0;
    $("seek").style.setProperty("--p", `${d ? (t / d) * 100 : 0}%`);
  }
}
audio.addEventListener("timeupdate", () => {
  showTime();
  if (Date.now() - lastSave > 5000 && !audio.paused) { lastSave = Date.now(); persist(); }
  tickSleep();
});
$("seek").addEventListener("input", () => {
  seeking = true;
  const d = audio.duration || 0, t = ($("seek").value / 1000) * d;
  $("seek").style.setProperty("--p", `${$("seek").value / 10}%`);
  $("cur").textContent = fmt(t);
  $("left").textContent = "-" + fmt(d - t);
});
$("seek").addEventListener("change", () => {
  audio.currentTime = ($("seek").value / 1000) * (audio.duration || 0);
  seeking = false;
});

$("play").addEventListener("click", togglePlay);
$("back").addEventListener("click", () => skip(-15));
$("fwd").addEventListener("click", () => skip(30));

// ---------- speed, sleep, volume ----------
function setSpeed(v) {
  state.speed = v;
  audio.playbackRate = v;
  $("speed").textContent = `${v}×`;
  save({ speed: v });
}
$("speed").addEventListener("click", () => setSpeed(SPEEDS[(SPEEDS.indexOf(state.speed) + 1) % SPEEDS.length]));

function setSleep(i) {
  sleep.i = i;
  const v = SLEEPS[i];
  sleep.at = v === "end" ? "end" : v ? Date.now() + v * 60000 : null;
  audio.volume = Number($("vol").value);
  showSleep();
}
function showSleep() {
  const b = $("sleep");
  b.classList.toggle("on", !!sleep.at);
  b.textContent = sleep.at === "end" ? "Sleep: end of chapter" : sleep.at ? `Sleep: ${Math.ceil((sleep.at - Date.now()) / 60000)} min` : "Sleep";
}
function tickSleep() {
  if (typeof sleep.at !== "number") return;
  const left = (sleep.at - Date.now()) / 1000, vol = Number($("vol").value);
  if (left <= 0) { audio.pause(); setSleep(0); return; }
  audio.volume = left < 20 ? vol * (left / 20) : vol; // fade over the last 20 seconds
  showSleep();
}
$("sleep").addEventListener("click", () => setSleep((sleep.i + 1) % SLEEPS.length));

$("vol").addEventListener("input", () => {
  audio.volume = $("vol").value;
  $("vol").style.setProperty("--p", `${$("vol").value * 100}%`);
  try { localStorage.setItem("vol", $("vol").value); } catch {}
});

document.addEventListener("keydown", (e) => {
  if (e.target.matches("input:not([type=range])")) return;
  if (e.code === "Space") { e.preventDefault(); togglePlay(); }
  else if (e.key === "ArrowLeft") { e.preventDefault(); skip(-15); }
  else if (e.key === "ArrowRight") { e.preventDefault(); skip(30); }
});

if ("mediaSession" in navigator) {
  navigator.mediaSession.setActionHandler("play", () => audio.play());
  navigator.mediaSession.setActionHandler("pause", () => audio.pause());
  navigator.mediaSession.setActionHandler("seekbackward", () => skip(-15));
  navigator.mediaSession.setActionHandler("seekforward", () => skip(30));
  navigator.mediaSession.setActionHandler("nexttrack", () => { const n = current && nextOf(current); if (n) play(current.book, n); });
  navigator.mediaSession.setActionHandler("previoustrack", () => { const p = current && prevOf(current); if (p) play(current.book, p); });
}
// The Bluetooth button pad on this machine (via padkeys) drives the window here.
const prevOf = ({ book: b, ch }) => b.chapters.slice(0, b.chapters.indexOf(ch)).reverse().find(ready);
function remoteCmd(cmd) {
  if (!current) {
    const b = bookOf(view) || inProgress()[0] || books[0], r = b && resumeOf(b);
    if (r) play(b, r);
    return;
  }
  if (cmd === "toggle") togglePlay();
  else if (cmd === "next") { const n = nextOf(current); if (n) play(current.book, n); }
  else if (cmd === "prev") { const p = prevOf(current); if (p) play(current.book, p); }
}
if (["127.0.0.1", "localhost", "[::1]"].includes(location.hostname)) {
  new EventSource("/api/remote/events").onmessage = (e) => remoteCmd(e.data);
}

document.addEventListener("visibilitychange", () => { if (document.hidden) persist(); });
window.addEventListener("pagehide", () => {
  if (!current || !audio.duration) return;
  persist();
});

// ---------- start ----------
(async function init() {
  const [lib, st] = await Promise.all([fetch("/api/library").then((r) => r.json()), fetch("/api/state").then((r) => r.json())]);
  books = lib.books;
  state = { chapters: {}, books: {}, speed: 1, ...st };
  $("speed").textContent = `${state.speed}×`;
  try {
    edition = localStorage.getItem("edition") || "extended";
    const v = localStorage.getItem("vol");
    if (v != null) { $("vol").value = v; audio.volume = v; $("vol").style.setProperty("--p", `${v * 100}%`); }
    view = localStorage.getItem("view");
  } catch {}
  if (view !== "continue" && !bookOf(view)) view = null;
  if (!view || (view === "continue" && !inProgress().length)) view = inProgress().length ? "continue" : books[0]?.id;
  select(view);
  if (state.last) {
    const [bid, cid] = state.last.split("/"), b = bookOf(bid), ch = b?.chapters.find((c) => c.id === cid);
    if (ch && ready(ch) && !chState(b, ch).played) load(b, ch, edFor(ch), posOf(b, ch, edFor(ch)), false);
  }
})();
