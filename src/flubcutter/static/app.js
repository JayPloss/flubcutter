import { renderReview, teardownReview } from "./review.js";

// ---------- helpers ----------

export async function api(path, opts = {}) {
  const init = { ...opts };
  if (opts.json !== undefined) {
    init.method = opts.method || "POST";
    init.headers = { "Content-Type": "application/json" };
    init.body = JSON.stringify(opts.json);
  }
  const res = await fetch(path, init);
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch {}
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return res.json();
}

export function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "html") el.innerHTML = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) {
    if (kid === null || kid === undefined || kid === false) continue;
    el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return el;
}

let toastTimer;
export function toast(msg, err = false) {
  const t = document.getElementById("toast");
  t.textContent = msg;
  t.className = "toast" + (err ? " err" : "");
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), err ? 6000 : 2500);
}

export async function guard(fn) {
  try { return await fn(); } catch (e) { toast(e.message, true); }
}

export function fmtTime(s) {
  if (s === null || s === undefined || isNaN(s)) return "–";
  const m = Math.floor(s / 60);
  return `${m}:${(s - m * 60).toFixed(1).padStart(4, "0")}`;
}

function fmtSize(b) {
  return b > 1e9 ? (b / 1e9).toFixed(1) + " GB" : b > 1e6 ? (b / 1e6).toFixed(1) + " MB" : Math.ceil(b / 1e3) + " KB";
}

const STAGES = [
  ["needs text", "blocked", "#9a9a92"],
  ["needs audio", "blocked", "#c4c4bc"],
  ["to transcribe", "machine", "#93c5fd"],
  ["to align", "machine", "#60a5fa"],
  ["to judge", "machine", "#3b82f6"],
  ["to review", "review", "#e8a317"],
  ["to render", "review", "#f2c14e"],
  ["done", "done", "#2f9e6b"],
];
const stageInfo = Object.fromEntries(STAGES.map(([s, cls, color]) => [s, { cls, color }]));

export function stagePill(st) {
  const info = stageInfo[st.stage] || { cls: "blocked" };
  return h("span", { class: `pill s-${info.cls}${st.busy ? " busy" : ""}` }, st.busy ? "working…" : st.stage);
}

function bar(frac, color = "") {
  return h("div", { class: `bar ${color}` }, h("span", { style: `width:${Math.round((frac || 0) * 100)}%` }));
}

export function setCrumbs(items) {
  const nav = document.getElementById("crumbs");
  nav.replaceChildren(...items.flatMap((it, i) => [
    i ? h("span", { class: "sep" }, "/") : null,
    it.href ? h("a", { href: it.href }, it.label) : h("span", {}, it.label),
  ].filter(Boolean)));
}

function modal(title, body, { ok = "Save", onOk } = {}) {
  const dlg = h("dialog", {},
    h("form", { method: "dialog" },
      h("h2", {}, title), body,
      h("div", { class: "actions" },
        h("button", { value: "cancel", formnovalidate: true }, "Cancel"),
        h("button", { class: "primary", value: "ok" }, ok))));
  document.body.append(dlg);
  dlg.addEventListener("close", async () => {
    if (dlg.returnValue === "ok" && onOk) await guard(onOk);
    dlg.remove();
  });
  dlg.showModal();
  return dlg;
}

// ---------- job polling ----------

let lastJobsKey = "";
let onJobsChanged = null;
const shownErrors = new Set();

async function pollJobs() {
  try {
    const jobs = await api("/api/jobs");
    const box = document.getElementById("jobs");
    const active = jobs.filter((j) => j.state === "queued" || j.state === "running");
    const running = jobs.find((j) => j.state === "running");
    if (active.length) {
      box.hidden = false;
      const label = running
        ? `${running.step || running.kind} · ${running.chapter}`
        : `${active.length} queued`;
      box.replaceChildren(h("span", {}, label + (active.length > 1 ? ` (+${active.length - 1})` : "")),
        bar(running ? running.progress : 0, "blue"));
    } else {
      box.hidden = true;
    }
    for (const j of jobs) {
      if (j.state === "error" && !shownErrors.has(j.id)) {
        shownErrors.add(j.id);
        toast(`${j.kind} failed for ${j.chapter}: ${j.error}`, true);
      }
    }
    const key = jobs.map((j) => `${j.id}:${j.state}`).join(",");
    if (key !== lastJobsKey) {
      const changed = lastJobsKey !== "";
      lastJobsKey = key;
      if (changed && onJobsChanged) onJobsChanged();
    }
  } catch {}
}
setInterval(pollJobs, 1500);
pollJobs();

export async function submitJob(kind, book, chapter, part) {
  await api("/api/jobs", { json: { kind, book, chapter, part } });
  toast(`Queued: ${kind}`);
  pollJobs();
}

// ---------- dashboard ----------

async function renderDashboard(view) {
  setCrumbs([{ label: "Library" }]);
  const lib = await api("/api/library");
  const all = lib.books.flatMap((b) => b.chapters.map((c) => c.status));
  const sum = (f) => all.reduce((a, s) => a + f(s), 0);
  const parts = sum((s) => s.parts), transcribed = sum((s) => s.transcribed);
  const cands = sum((s) => s.candidates), reviewed = sum((s) => s.reviewed);
  const withAudio = all.filter((s) => s.parts > 0).length;
  const aligned = all.filter((s) => s.aligned).length;
  const done = all.filter((s) => s.stage === "done").length;
  const rendered = all.filter((s) => s.input_seconds !== null);
  const inSec = rendered.reduce((a, s) => a + s.input_seconds, 0);
  const outSec = rendered.reduce((a, s) => a + s.output_seconds, 0);
  const fmtDur = (s) => (s >= 3600 ? `${Math.floor(s / 3600)}h ${Math.round((s % 3600) / 60)}m` : `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`);

  const stat = (label, value, total, color) =>
    h("div", { class: "panel stat" },
      h("div", { class: "label" }, label),
      h("div", { class: "value" }, value, h("small", {}, ` / ${total}`)),
      bar(total ? value / total : 0, color));

  const books = lib.books.map((b) => {
    const counts = {};
    b.chapters.forEach((c) => (counts[c.status.stage] = (counts[c.status.stage] || 0) + 1));
    const n = b.chapters.length || 1;
    const stagebar = h("div", { class: "stagebar", title: Object.entries(counts).map(([k, v]) => `${v} ${k}`).join(", ") },
      STAGES.filter(([s]) => counts[s]).map(([s, , color]) =>
        h("span", { style: `width:${(counts[s] / n) * 100}%;background:${color}` })));
    const rows = b.chapters.map((c, i) => {
      const s = c.status;
      return h("tr", { class: "clickable", onclick: () => (location.hash = s.aligned ? `#/book/${b.id}/ch/${c.id}` : `#/book/${b.id}`) },
        h("td", { class: "num" }, i + 1),
        h("td", {}, c.title),
        h("td", {}, stagePill(s)),
        h("td", { class: "prog small muted" }, `transcribed ${s.transcribed}/${s.parts}`, bar(s.parts ? s.transcribed / s.parts : 0, "blue")),
        h("td", { class: "prog small muted" }, `reviewed ${s.reviewed}/${s.candidates}`, bar(s.candidates ? s.reviewed / s.candidates : (s.aligned ? 1 : 0), "amber")));
    });
    return h("section", { class: "panel" },
      h("div", { class: "panel-head" },
        h("div", {}, h("h2", {}, h("a", { href: `#/book/${b.id}`, style: "text-decoration:none" }, b.title)),
          h("div", { class: "small muted" }, [b.author, `${b.chapters.length} chapters`].filter(Boolean).join(" · "))),
        stagebar,
        h("a", { class: "btn", href: `#/book/${b.id}` }, "Manage")),
      b.chapters.length
        ? h("table", { class: "chapters" },
            h("thead", {}, h("tr", {}, h("th", {}, "#"), h("th", {}, "Chapter"), h("th", {}, "Stage"),
              h("th", { class: "prog" }, "Transcription"), h("th", { class: "prog" }, "Review"))),
            h("tbody", {}, rows))
        : h("div", { class: "empty" }, "No chapters yet. ", h("a", { href: `#/book/${b.id}` }, "Import the book text"), "."));
  });

  view.replaceChildren(
    h("div", { class: "stack" },
      h("div", { class: "row" }, h("h1", {}, "Library"), h("span", { class: "spacer" }),
        h("span", { class: "small muted mono" }, lib.root),
        h("button", { class: "primary", onclick: newBook }, "New book")),
      !lib.judge ? h("div", { class: "panel panel-body small" }, "The AI judge needs the ", h("code", {}, "claude"), " CLI on your PATH. Without it, unexplained extra speech is kept until you review it.") : null,
      lib.books.length ? h("div", { class: "stats" },
        stat("Parts transcribed", transcribed, parts, "blue"),
        stat("Chapters aligned", aligned, withAudio, "blue"),
        stat("Flubs reviewed", reviewed, cands, "amber"),
        stat("Chapters finished", done, all.length, "green"),
        h("div", { class: "panel stat" },
          h("div", { class: "label" }, "Audio trimmed"),
          h("div", { class: "value" }, rendered.length ? fmtDur(inSec - outSec) : "–",
            rendered.length ? h("small", {}, ` of ${fmtDur(inSec)}`) : null),
          h("div", { class: "small muted" }, rendered.length
            ? `${Math.round((1 - outSec / inSec) * 100)}% removed across ${rendered.length} rendered chapter(s)`
            : "Shows up once chapters are rendered"))) : null,
      lib.books.length ? h("div", { class: "legend" }, STAGES.map(([s, , color]) => h("span", {}, h("i", { style: `background:${color}` }), s))) : null,
      ...books,
      lib.books.length ? null : h("div", { class: "panel empty" },
        h("p", {}, "Your library is empty."),
        h("button", { class: "primary", onclick: newBook }, "Add your first book"))));
}

function newBook() {
  const title = h("input", { type: "text", placeholder: "Harry Potter and the Philosopher's Stone", required: true });
  const author = h("input", { type: "text", placeholder: "Author (optional)" });
  modal("New book", h("div", { class: "stack" }, title, author), {
    ok: "Create",
    onOk: async () => {
      const b = await api("/api/books", { json: { title: title.value, author: author.value } });
      location.hash = `#/book/${b.id}`;
    },
  });
  setTimeout(() => title.focus(), 50);
}

// ---------- book page ----------

async function renderBook(view, bookId) {
  const book = await api(`/api/books/${bookId}`);
  setCrumbs([{ label: "Library", href: "#/" }, { label: book.title }]);
  const refresh = () => renderBook(view, bookId);
  const unassigned = book.audio.filter((a) => !a.chapter);

  const fileInput = h("input", { type: "file", accept: ".epub,.txt,text/plain,application/epub+zip", hidden: true,
    onchange: () => guard(async () => {
      const fd = new FormData();
      fd.append("file", fileInput.files[0]);
      const r = await api(`/api/books/${bookId}/import`, { method: "POST", body: fd });
      toast(`Added ${r.added} chapters`);
      refresh();
    }) });

  const audioInput = h("input", { type: "file", accept: "audio/*,.m4a,.caf", multiple: true, hidden: true,
    onchange: () => uploadAudio(audioInput.files) });

  async function uploadAudio(files) {
    if (!files.length) return;
    await guard(async () => {
      const fd = new FormData();
      for (const f of files) fd.append("files", f);
      toast(`Uploading ${files.length} file(s)…`);
      const r = await api(`/api/books/${bookId}/audio`, { method: "POST", body: fd });
      toast(`Added ${r.saved.length} recording(s)`);
      refresh();
    });
  }

  const drop = h("div", { class: "dropzone",
    ondragover: (e) => { e.preventDefault(); drop.classList.add("over"); },
    ondragleave: () => drop.classList.remove("over"),
    ondrop: (e) => { e.preventDefault(); drop.classList.remove("over"); uploadAudio(e.dataTransfer.files); } },
    "Drop recordings here, or ", h("a", { href: "#", onclick: (e) => { e.preventDefault(); audioInput.click(); } }, "choose files"),
    h("div", { class: "small", style: "margin-top:6px" }, "You can also copy files straight into ", h("span", { class: "mono" }, book.audio_dir)));

  const files = h("div", { class: "filelist" },
    book.audio.length ? book.audio.map((a) => h("div", {},
      h("span", { class: "mono", style: "flex:1;overflow:hidden;text-overflow:ellipsis" }, a.name),
      h("span", { class: "muted small" }, fmtSize(a.size)),
      h("span", { class: "small " + (a.chapter ? "" : "muted") }, a.chapter ? `→ ${a.chapter}` : "unassigned")))
      : h("div", { class: "muted small" }, "No recordings yet."));

  const rows = book.chapters.map((c, i) => {
    const s = c.status;
    const select = h("select", { onchange: () => guard(async () => {
      if (!select.value) return;
      await api(`/api/books/${bookId}/chapters/${c.id}/parts`, { json: { audio: select.value } });
      refresh();
    }) },
      h("option", { value: "" }, unassigned.length ? "+ add recording…" : "no unassigned recordings"),
      unassigned.map((a) => h("option", { value: a.path }, a.name)));
    const parts = c.parts.map((p, k) => h("span", { class: "chip", title: p.audio },
      `${k + 1}. ${p.audio.replace(/^audio\//, "")}`,
      k > 0 ? h("button", { title: "Move earlier", onclick: () => guard(async () => { await api(`/api/books/${bookId}/chapters/${c.id}/parts/${p.id}/move`, { json: { delta: -1 } }); refresh(); }) }, "↑") : null,
      h("button", { title: "Remove from chapter", onclick: () => guard(async () => { await api(`/api/books/${bookId}/chapters/${c.id}/parts/${p.id}`, { method: "DELETE" }); refresh(); }) }, "✕")));
    const canProcess = ["to transcribe", "to align", "to judge"].includes(s.stage);
    return h("tr", {},
      h("td", { class: "num" }, i + 1),
      h("td", {}, h("div", {}, c.title), h("div", { class: "small muted" }, `${s.words.toLocaleString()} words`)),
      h("td", {}, parts, h("div", {}, select)),
      h("td", {}, stagePill(s)),
      h("td", { style: "text-align:right;white-space:nowrap" },
        canProcess && !s.busy ? h("button", { onclick: () => guard(async () => { await submitJob("process", bookId, c.id); refresh(); }) }, "Process") : null,
        s.aligned ? h("a", { class: "btn primary", href: `#/book/${bookId}/ch/${c.id}` }, "Review") : null,
        " ",
        h("button", { class: "ghost icon", title: "Edit text", onclick: () => editChapter(bookId, c, refresh) }, "✎"),
        h("button", { class: "ghost icon", title: "Move up", disabled: i === 0, onclick: () => guard(async () => { await api(`/api/books/${bookId}/chapters/${c.id}/move`, { json: { delta: -1 } }); refresh(); }) }, "↑"),
        h("button", { class: "ghost icon", title: "Move down", disabled: i === book.chapters.length - 1, onclick: () => guard(async () => { await api(`/api/books/${bookId}/chapters/${c.id}/move`, { json: { delta: 1 } }); refresh(); }) }, "↓"),
        h("button", { class: "ghost icon danger", title: "Delete chapter", onclick: () => {
          if (confirm(`Delete "${c.title}"? Its review decisions are lost. Recordings stay in the audio folder.`))
            guard(async () => { await api(`/api/books/${bookId}/chapters/${c.id}`, { method: "DELETE" }); refresh(); });
        } }, "🗑")));
  });

  const processable = book.chapters.filter((c) => ["to transcribe", "to align", "to judge"].includes(c.status.stage) && !c.status.busy).length;

  view.replaceChildren(h("div", { class: "stack" },
    h("div", { class: "row" },
      h("div", {}, h("h1", {}, book.title), h("div", { class: "muted" }, book.author || "")),
      h("span", { class: "spacer" }),
      h("button", { onclick: () => {
        const t = h("input", { type: "text", value: book.title });
        const a = h("input", { type: "text", value: book.author || "", placeholder: "Author" });
        modal("Rename book", h("div", { class: "stack" }, t, a), { onOk: async () => { await api(`/api/books/${bookId}`, { method: "PATCH", json: { title: t.value, author: a.value } }); refresh(); } });
      } }, "Rename"),
      h("button", { class: "danger", onclick: () => {
        if (confirm(`Delete "${book.title}"? Recordings are moved to the library's _removed folder, not deleted.`))
          guard(async () => { await api(`/api/books/${bookId}`, { method: "DELETE" }); location.hash = "#/"; });
      } }, "Delete book")),
    h("section", { class: "panel" },
      h("div", { class: "panel-head" }, h("h2", {}, "Chapters"), h("span", { class: "spacer" }),
        fileInput, audioInput,
        h("button", { onclick: () => fileInput.click() }, "Import EPUB / text…"),
        h("button", { onclick: () => editChapter(bookId, null, refresh) }, "Add chapter"),
        h("button", { class: "primary", disabled: !processable, onclick: () => guard(async () => {
          const r = await api(`/api/books/${bookId}/process`, { json: {} });
          toast(`Queued ${r.queued} chapter(s)`);
          pollJobs(); refresh();
        }) }, `Process all${processable ? ` (${processable})` : ""}`)),
      book.chapters.length
        ? h("table", { class: "chapters" },
            h("thead", {}, h("tr", {}, h("th", {}, "#"), h("th", {}, "Chapter"), h("th", {}, "Recording parts, in order"), h("th", {}, "Stage"), h("th", {}))),
            h("tbody", {}, rows))
        : h("div", { class: "empty" }, "Import the book from an EPUB or text file, or add chapters one by one and paste their text.")),
    h("section", { class: "panel" },
      h("div", { class: "panel-head" }, h("h2", {}, "Recordings"), h("span", { class: "muted small" }, `${book.audio.length} files, ${unassigned.length} unassigned`),
        h("span", { class: "spacer" }), h("button", { onclick: refresh }, "Rescan folder")),
      h("div", { class: "panel-body" }, drop, files))));
}

async function editChapter(bookId, ch, refresh) {
  const title = h("input", { type: "text", value: ch ? ch.title : "", placeholder: "Chapter title", required: true });
  const text = h("textarea", { placeholder: "Paste the chapter's text here" });
  if (ch) {
    const detail = await api(`/api/books/${bookId}/chapters/${ch.id}`);
    text.value = detail.text;
  }
  modal(ch ? "Edit chapter" : "Add chapter", h("div", { class: "stack" }, title, text,
    ch ? h("div", { class: "small muted" }, "Changing the text re-runs the alignment for this chapter. Your decisions are kept where they still apply.") : null), {
    onOk: async () => {
      if (ch) await api(`/api/books/${bookId}/chapters/${ch.id}`, { method: "PATCH", json: { title: title.value, text: text.value } });
      else await api(`/api/books/${bookId}/chapters`, { json: { title: title.value, text: text.value } });
      refresh();
    },
  });
}

// ---------- router ----------

async function route() {
  const view = document.getElementById("view");
  teardownReview();
  onJobsChanged = null;
  const parts = location.hash.replace(/^#\/?/, "").split("/").filter(Boolean);
  try {
    if (parts[0] === "book" && parts[2] === "ch") {
      onJobsChanged = await renderReview(view, parts[1], parts[3]);
    } else if (parts[0] === "book") {
      await renderBook(view, parts[1]);
      onJobsChanged = () => renderBook(view, parts[1]);
    } else {
      await renderDashboard(view);
      onJobsChanged = () => renderDashboard(view);
    }
  } catch (e) {
    view.replaceChildren(h("div", { class: "panel empty" }, e.message, " · ", h("a", { href: "#/" }, "Back to library")));
  }
}

window.addEventListener("hashchange", route);
route();
