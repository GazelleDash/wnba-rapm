/* WNBA RAPM explorer — dense sortable table with percentile shading.
   No framework, no build step. Loads data/*.json and does all sorting,
   filtering, percentile ranking and heat shading in the browser.

   Percentiles are ALWAYS recomputed over the currently qualified pool, so
   changing season/window/team/min-poss re-ranks everything live. */

/* ── tiny helpers ────────────────────────────────────────────────────── */
const $ = id => document.getElementById(id);
const _esc = s => String(s ?? "").replace(/[&<>"]/g,
  c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

/* ── state ───────────────────────────────────────────────────────────── */
let DATA = { players: null, td: null, meta: null };
let view = "players";
let sortKey = null, sortDesc = true;
let filtMode = false;
let colsOn = new Set();
let statFilters = {};          // {colKey: {min, max}}
let pctlCache = {};            // {colKey: ascending values over current pool}
let showAll = false;
const ROW_CAP = 300;

/* ── column definitions ──────────────────────────────────────────────────
   k: key (also the JSON column name)   l: header label   g: group band
   d: decimals   pm: signed (+/-)   lo: lower-is-better   txt: text column
   noPct: no percentile subscript   cls: extra td class                  */
const DEFS = {
  players: [
    { k: "name",        l: "Player",  g: "Overview",   txt: 1, cls: "name", noPct: 1 },
    { k: "team",        l: "Team",    g: "Overview",   txt: 1, cls: "team", noPct: 1 },
    { k: "total_poss",  l: "Poss",    g: "Overview",   d: 0,   noPct: 1 },
    { k: "rapm",        l: "RAPM",    g: "RAPM",       d: 2, pm: 1 },
    { k: "orapm",       l: "ORAPM",   g: "RAPM",       d: 2, pm: 1 },
    { k: "drapm",       l: "DRAPM",   g: "RAPM",       d: 2, pm: 1 },
    { k: "off_ts_val",  l: "oTS",     g: "Offense",    d: 2, pm: 1 },
    { k: "off_tov_val", l: "oTOV",    g: "Offense",    d: 2, pm: 1 },
    { k: "off_reb_val", l: "oREB",    g: "Offense",    d: 2, pm: 1 },
    { k: "def_ts_val",  l: "dTS",     g: "Defense",    d: 2, pm: 1 },
    { k: "def_tov_val", l: "dTOV",    g: "Defense",    d: 2, pm: 1 },
    { k: "def_reb_val", l: "dREB",    g: "Defense",    d: 2, pm: 1 },
    { k: "o_poss_val",  l: "oPoss",   g: "Possession", d: 2, pm: 1 },
    { k: "d_poss_val",  l: "dPoss",   g: "Possession", d: 2, pm: 1 },
    { k: "poss_val",    l: "PossVal", g: "Possession", d: 2, pm: 1 },
  ],
  td: [
    { k: "name",         l: "Player",   g: "Overview", txt: 1, cls: "name", noPct: 1 },
    { k: "team",         l: "Team",     g: "Overview", txt: 1, cls: "team", noPct: 1 },
    { k: "total_poss",   l: "Eff Poss", g: "Overview", d: 0,   noPct: 1 },
    { k: "RAPM",         l: "RAPM",     g: "RAPM",     d: 2, pm: 1 },
    { k: "ORAPM",        l: "ORAPM",    g: "RAPM",     d: 2, pm: 1 },
    { k: "DRAPM",        l: "DRAPM",    g: "RAPM",     d: 2, pm: 1 },
    { k: "off_ts_pts",   l: "oTS",      g: "Offense",  d: 2, pm: 1 },
    { k: "off_tov_pts",  l: "oTOV",     g: "Offense",  d: 2, pm: 1 },
    { k: "off_reb_pts",  l: "oREB",     g: "Offense",  d: 2, pm: 1 },
    { k: "def_ts_pts",   l: "dTS",      g: "Defense",  d: 2, pm: 1 },
    { k: "def_tov_pts",  l: "dTOV",     g: "Defense",  d: 2, pm: 1 },
    { k: "def_reb_pts",  l: "dREB",     g: "Defense",  d: 2, pm: 1 },
  ],
};
const DEFAULT_SORT = { players: "rapm", td: "RAPM" };

function defs() { return DEFS[view]; }
function visibleDefs() { return defs().filter(d => colsOn.has(d.k)); }
function dataset() { return DATA[view]; }

/* Column key -> index in the current dataset's row arrays. */
function colIdx(k) {
  const ds = dataset();
  return ds ? ds.cols.indexOf(k) : -1;
}
function cellValue(r, d) {
  const i = colIdx(d.k);
  return i < 0 ? null : r[i];
}

/* ── percentiles (mirrors WBPM _rankPctl) ────────────────────────────── */
function buildPctlCache(pool) {
  pctlCache = {};
  for (const d of defs()) {
    if (d.txt || d.noPct) continue;
    const i = colIdx(d.k);
    if (i < 0) continue;
    const vals = [];
    for (const r of pool) {
      const v = r[i];
      if (v != null && !Number.isNaN(v)) vals.push(v);
    }
    vals.sort((a, b) => a - b);
    pctlCache[d.k] = vals;
  }
}
function pctlOf(d, v) {
  const vals = pctlCache[d.k];
  if (!vals || !vals.length || v == null) return null;
  let a = 0, b = vals.length;
  while (a < b) { const mid = (a + b) >> 1; if (vals[mid] < v) a = mid + 1; else b = mid; }
  let p = Math.round(100 * a / vals.length);
  if (d.lo) p = 100 - p;
  return Math.min(99, Math.max(1, p));
}

/* ── heat color (mirrors WBPM heatColor exactly) ─────────────────────── */
function heatColor(p) {
  if (p == null) return "";
  const dist = Math.abs(p - 50) / 50;
  return `hsl(${Math.round(p * 1.2)}, ${Math.round(45 + 30 * dist)}%, ${Math.round(93 - 6 * dist)}%)`;
}

/* ── filtering ───────────────────────────────────────────────────────── */
function possKey() { return "total_poss"; }

/* Rows matching only the *pool* filters (season/window/min-poss/team).
   Percentiles rank within THIS pool — search and the ≥/≤ strip narrow what
   is displayed without moving the goalposts on the ranking. */
function poolRows() {
  const ds = dataset();
  if (!ds || !ds.rows.length) return [];
  let rows = ds.rows;

  if (view === "players") {
    const yi = ds.cols.indexOf("end_year"), wi = ds.cols.indexOf("rapm_length");
    const y = +$("yearsel").value, w = +$("windowsel").value;
    if (yi >= 0 && wi >= 0 && y && w) rows = rows.filter(r => r[yi] === y && r[wi] === w);
  }
  const pi = ds.cols.indexOf(possKey());
  const minp = +$("minposs").value || 0;
  if (pi >= 0 && minp > 0) rows = rows.filter(r => (r[pi] ?? 0) >= minp);

  const ti = ds.cols.indexOf("team");
  const team = $("teamsel").value;
  if (ti >= 0 && team) rows = rows.filter(r => r[ti] === team);

  return rows;
}

function displayRows(pool) {
  const ds = dataset();
  let rows = pool;
  const q = $("search").value.toLowerCase().trim();
  if (q) {
    const ni = ds.cols.indexOf("name"), ti = ds.cols.indexOf("team");
    rows = rows.filter(r =>
      String(r[ni] ?? "").toLowerCase().includes(q) ||
      String(r[ti] ?? "").toLowerCase().includes(q));
  }
  for (const [k, f] of Object.entries(statFilters)) {
    const d = defs().find(x => x.k === k);
    if (!d) continue;
    const i = colIdx(k);
    if (i < 0) continue;
    if (f.min != null) rows = rows.filter(r => r[i] != null && r[i] >= f.min);
    if (f.max != null) rows = rows.filter(r => r[i] != null && r[i] <= f.max);
  }
  return rows;
}

function sortRows(rows) {
  const key = sortKey || DEFAULT_SORT[view];
  const d = defs().find(x => x.k === key);
  if (!d) return rows;
  const i = colIdx(key);
  if (i < 0) return rows;
  const out = rows.slice();
  out.sort((ra, rb) => {
    const a = ra[i], b = rb[i];
    if (a == null && b == null) return 0;
    if (a == null) return 1;              // nulls always last
    if (b == null) return -1;
    let c;
    if (d.txt) c = String(a).localeCompare(String(b));
    else c = a - b;
    return sortDesc ? -c : c;
  });
  return out;
}

/* ── rendering ───────────────────────────────────────────────────────── */
function cellHTML(d, r, sorted) {
  let klass = d.cls || "", heat = null, inner;
  if (sorted) klass += " sorted";
  const v = cellValue(r, d);

  if (d.txt) {
    inner = _esc(v ?? "");
  } else if (v == null) {
    inner = "";
  } else {
    const s = d.d === 0 ? Number(v).toLocaleString()
            : d.pm ? (v >= 0 ? "+" : "") + Number(v).toFixed(d.d ?? 1)
            : Number(v).toFixed(d.d ?? 1);
    if (d.pm) klass += v >= 0 ? " pos" : " neg";
    const p = d.noPct ? null : pctlOf(d, v);
    heat = p;
    inner = `<span class="v">${s}</span>` + (p != null ? `<span class="pct">${p}%</span>` : "");
    if (p != null) inner = `<span class="cellstack" title="${_esc(d.l)}: ${p}th percentile">${inner}</span>`;
  }
  const style = heat != null ? ` style="--heat:${heatColor(heat)}"` : "";
  return `<td class="${klass.trim()}"${style}>${inner}</td>`;
}

function buildHead() {
  const thead = document.querySelector("#stats thead");
  thead.innerHTML = "";
  const vis = visibleDefs();
  const effSort = sortKey || DEFAULT_SORT[view];

  // group band, run-length encoded over consecutive d.g values
  const gtr = document.createElement("tr");
  gtr.className = "grprow";
  const gSpacer = document.createElement("th");
  gSpacer.className = "rkcol";
  gtr.appendChild(gSpacer);
  let run = null;
  for (const d of vis) {
    if (run && run.g === d.g) { run.n++; continue; }
    if (run) gtr.appendChild(mkGroupTh(run));
    run = { g: d.g, n: 1 };
  }
  if (run) gtr.appendChild(mkGroupTh(run));
  thead.appendChild(gtr);

  const tr = document.createElement("tr");
  const rk = document.createElement("th");
  rk.className = "rkcol"; rk.textContent = "Rk";
  tr.appendChild(rk);
  for (const d of vis) {
    const th = document.createElement("th");
    th.textContent = d.l;
    th.tabIndex = 0;
    if (d.cls === "name") th.className = "namecol";
    if (d.k === effSort) {
      th.className = (th.className + " sorted").trim();
      const a = document.createElement("span");
      a.className = "arr"; a.textContent = sortDesc ? "▼" : "▲";
      th.appendChild(a);
    }
    const doSort = () => {
      if (sortKey === d.k) sortDesc = !sortDesc;
      else { sortKey = d.k; sortDesc = !d.txt; }
      writeHash(); render();
    };
    th.onclick = doSort;
    th.onkeydown = e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); doSort(); } };
    tr.appendChild(th);
  }
  thead.appendChild(tr);

  if (filtMode) thead.appendChild(buildFiltRow(vis));

  // offset the sticky header rows so they stack instead of overlapping
  requestAnimationFrame(() => {
    document.documentElement.style.setProperty("--grph", gtr.offsetHeight + "px");
    document.documentElement.style.setProperty(
      "--hdrh", (gtr.offsetHeight + tr.offsetHeight) + "px");
  });
}

function mkGroupTh(run) {
  const th = document.createElement("th");
  th.colSpan = run.n; th.textContent = run.g;
  return th;
}

function buildFiltRow(vis) {
  const tr = document.createElement("tr");
  tr.className = "filtrow";
  tr.appendChild(document.createElement("th"));
  for (const d of vis) {
    const th = document.createElement("th");
    if (!d.txt) {
      const f = statFilters[d.k] || {};
      const mk = (which, ph, val) => {
        const inp = document.createElement("input");
        inp.type = "number"; inp.placeholder = ph; inp.step = "any";
        if (val != null) inp.value = val;
        inp.oninput = () => {
          statFilters[d.k] = statFilters[d.k] || {};
          const n = inp.value === "" ? null : Number(inp.value);
          statFilters[d.k][which] = Number.isNaN(n) ? null : n;
          if (statFilters[d.k].min == null && statFilters[d.k].max == null)
            delete statFilters[d.k];
          renderBody();
        };
        return inp;
      };
      th.appendChild(mk("min", "≥", f.min));
      th.appendChild(mk("max", "≤", f.max));
    }
    tr.appendChild(th);
  }
  return tr;
}

function renderBody() {
  const tbody = document.querySelector("#stats tbody");
  const pool = poolRows();
  buildPctlCache(pool);
  const rows = sortRows(displayRows(pool));
  const vis = visibleDefs();
  const effSort = sortKey || DEFAULT_SORT[view];

  const capped = !showAll && rows.length > ROW_CAP;
  const shown = capped ? rows.slice(0, ROW_CAP) : rows;

  let html = "";
  shown.forEach((r, i) => {
    html += "<tr><td class='rk'>" + (i + 1) + "</td>";
    for (const d of vis) html += cellHTML(d, r, d.k === effSort);
    html += "</tr>";
  });
  if (capped) {
    html += `<tr class="capnote"><td colspan="${vis.length + 1}">`
         +  `Showing ${ROW_CAP.toLocaleString()} of ${rows.length.toLocaleString()} rows — `
         +  `<a href="#" id="showall">show all</a></td></tr>`;
  }
  tbody.innerHTML = html || `<tr class="capnote"><td colspan="${vis.length + 1}">No rows match these filters.</td></tr>`;
  const sa = $("showall");
  if (sa) sa.onclick = e => { e.preventDefault(); showAll = true; renderBody(); };

  updateQuery(rows.length, capped ? ROW_CAP : null);
}

function render() { buildHead(); renderBody(); }

/* ── query echo ──────────────────────────────────────────────────────── */
function updateQuery(n, shown) {
  const parts = [];
  if (view === "players") {
    parts.push(`season == ${$("yearsel").value}`, `window == ${$("windowsel").value}`);
  }
  const minp = +$("minposs").value || 0;
  if (minp) parts.push(`poss >= ${minp}`);
  if ($("teamsel").value) parts.push(`team == "${$("teamsel").value}"`);
  const q = $("search").value.trim();
  if (q) parts.push(`str_detect(player, "${q}")`);
  for (const [k, f] of Object.entries(statFilters)) {
    if (f.min != null) parts.push(`${k} >= ${f.min}`);
    if (f.max != null) parts.push(`${k} <= ${f.max}`);
  }
  const key = sortKey || DEFAULT_SORT[view];
  const label = { players: "rapm", td: "td_rapm" }[view];
  let line = `<span class="fn">${label}</span>`;
  if (parts.length) line += ` <span class="pipe">|&gt;</span> <span class="fn">filter</span>(${_esc(parts.join(", "))})`;
  line += ` <span class="pipe">|&gt;</span> <span class="fn">arrange</span>(`
       +  (sortDesc ? `<span class="fn">desc</span>(${_esc(key)})` : _esc(key)) + `)`;
  $("queryline").innerHTML = line;
  $("tibble").textContent = `## A tibble: ${n.toLocaleString()} × ${visibleDefs().length}`
    + (shown != null && n > shown ? ` — with ${(n - shown).toLocaleString()} more rows` : "");
}

/* ── panels ──────────────────────────────────────────────────────────── */
function buildColPanel() {
  const groups = [...new Set(defs().map(d => d.g))];
  $("grplinks").innerHTML =
    `<button type="button" class="linklike" data-g="__all">All</button>` +
    groups.map(g => `<button type="button" class="linklike" data-g="${_esc(g)}">${_esc(g)}</button>`).join("");
  $("grplinks").querySelectorAll("button").forEach(b => {
    b.onclick = () => {
      const g = b.dataset.g;
      if (g === "__all") defs().forEach(d => colsOn.add(d.k));
      else defs().filter(d => d.g === g).forEach(d => colsOn.add(d.k));
      buildColPanel(); writeHash(); render();
    };
  });
  $("colchecks").innerHTML = groups.map(g =>
    `<div class="colgroup"><span class="colgroup-head">${_esc(g)}</span><span class="colgroup-body">` +
    defs().filter(d => d.g === g).map(d =>
      `<label class="colchk"><input type="checkbox" data-k="${_esc(d.k)}"${colsOn.has(d.k) ? " checked" : ""}> ${_esc(d.l)}</label>`
    ).join("") + `</span></div>`).join("");
  $("colchecks").querySelectorAll("input").forEach(cb => {
    cb.onchange = () => {
      const k = cb.dataset.k;
      if (cb.checked) colsOn.add(k); else colsOn.delete(k);
      if (!colsOn.size) { colsOn.add(defs()[0].k); buildColPanel(); }
      writeHash(); render();
    };
  });
}

function resetCols() { colsOn = new Set(defs().map(d => d.k)); }

/* ── CSV export ──────────────────────────────────────────────────────── */
function toCSV() {
  const vis = visibleDefs();
  const pool = poolRows();
  buildPctlCache(pool);
  const rows = sortRows(displayRows(pool));
  const esc = s => {
    const t = String(s ?? "");
    return /[",\n]/.test(t) ? `"${t.replace(/"/g, '""')}"` : t;
  };
  const lines = [["Rk", ...vis.map(d => d.l)].map(esc).join(",")];
  rows.forEach((r, i) => {
    lines.push([i + 1, ...vis.map(d => {
      const v = cellValue(r, d);
      return v == null ? "" : v;
    })].map(esc).join(","));
  });
  const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `wnba_${view}${view === "players" ? `_${$("yearsel").value}_${$("windowsel").value}Y` : ""}.csv`;
  document.body.appendChild(a); a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 0);
}

/* ── hash state ──────────────────────────────────────────────────────── */
let applyingHash = false;
function writeHash() {
  if (applyingHash) return;
  const p = [`view=${view}`];
  if (view === "players") p.push(`y=${$("yearsel").value}`, `w=${$("windowsel").value}`);
  if (sortKey) p.push(`sort=${sortKey}`, `dir=${sortDesc ? "d" : "a"}`);
  const minp = $("minposs").value;
  if (minp) p.push(`mp=${minp}`);
  if ($("teamsel").value) p.push(`team=${encodeURIComponent($("teamsel").value)}`);
  history.replaceState(null, "", "#" + p.join("&"));
}
function readHash() {
  const h = location.hash.replace(/^#/, "");
  if (!h) return {};
  const o = {};
  for (const kv of h.split("&")) {
    const [k, v] = kv.split("=");
    if (k) o[k] = decodeURIComponent(v ?? "");
  }
  return o;
}

/* ── view switching ──────────────────────────────────────────────────── */
function setView(v, opts = {}) {
  view = DEFS[v] ? v : "players";
  sortKey = opts.sort || null;
  sortDesc = opts.dir ? opts.dir === "d" : true;
  statFilters = {}; showAll = false;
  resetCols();

  document.querySelectorAll("#viewnav a").forEach(a =>
    a.classList.toggle("active", a.dataset.view === view));
  const playersOnly = view === "players";
  $("yearwrap").style.display = playersOnly ? "" : "none";
  $("windowwrap").style.display = playersOnly ? "" : "none";

  // possession floors differ wildly between views
  if (!opts.mp) $("minposs").value = view === "players" ? 500 : view === "td" ? 300 : 200;
  else $("minposs").value = opts.mp;

  populateTeams();
  buildColPanel();
  render();
}

function populateTeams() {
  const ds = dataset();
  const sel = $("teamsel");
  const keep = sel.value;
  const ti = ds ? ds.cols.indexOf("team") : -1;
  let teams = [];
  if (ds && ti >= 0) {
    // only teams present in the current season/window slice
    const base = view === "players" ? poolRowsIgnoringTeam() : ds.rows;
    teams = [...new Set(base.map(r => r[ti]).filter(Boolean))].sort();
  }
  sel.innerHTML = `<option value="">All Teams</option>` +
    teams.map(t => `<option value="${_esc(t)}">${_esc(t)}</option>`).join("");
  if (teams.includes(keep)) sel.value = keep;
}
function poolRowsIgnoringTeam() {
  const ds = dataset();
  if (!ds) return [];
  let rows = ds.rows;
  const yi = ds.cols.indexOf("end_year"), wi = ds.cols.indexOf("rapm_length");
  const y = +$("yearsel").value, w = +$("windowsel").value;
  if (yi >= 0 && wi >= 0 && y && w) rows = rows.filter(r => r[yi] === y && r[wi] === w);
  return rows;
}

/* ── boot ────────────────────────────────────────────────────────────── */
function showError(msg) {
  document.querySelector("#stats tbody").innerHTML = "";
  const box = document.createElement("div");
  box.className = "errbox";
  box.textContent = msg;
  document.querySelector(".tablewrap").prepend(box);
}

async function boot() {
  const tbody = document.querySelector("#stats tbody");
  tbody.innerHTML = `<tr><td class="loading">Loading data…</td></tr>`;
  try {
    const [players, td, meta] = await Promise.all(
      ["players", "td", "meta"].map(n =>
        fetch(`data/${n}.json`).then(r => {
          if (!r.ok) throw new Error(`data/${n}.json → HTTP ${r.status}`);
          return r.json();
        })));
    DATA = { players, td, meta };
  } catch (e) {
    showError(`Could not load the data files (${e.message}). If you opened this page as a file:// URL, serve the folder over HTTP instead.`);
    return;
  }

  if (DATA.meta?.updated) $("updated").textContent = DATA.meta.updated;
  if (DATA.meta?.td_as_of) {
    // td_as_of is the latest game date actually present in the source data —
    // NOT when the pipeline last ran. The two can (and do) diverge: the site
    // updates daily even on days with no new games.
    const d = new Date(DATA.meta.td_as_of + "T00:00:00Z");
    const pretty = isNaN(d) ? DATA.meta.td_as_of
      : d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" });
    $("gamesThrough").textContent = pretty;
    $("asofBadge").innerHTML = `Games through <b>${_esc(pretty)}</b>`;
  }

  const seasons = (DATA.meta?.seasons || []).slice().sort((a, b) => b - a);
  $("yearsel").innerHTML = seasons.map(s => `<option value="${s}">${s}</option>`).join("");

  const h = readHash();
  const defaultYear = seasons.includes(2026) ? "2026" : String(seasons[0] ?? "");
  $("yearsel").value = (h.y && seasons.includes(+h.y)) ? h.y : defaultYear;
  $("windowsel").value = h.w || "1";

  // wire controls
  $("yearsel").onchange = () => { showAll = false; populateTeams(); writeHash(); render(); };
  $("windowsel").onchange = () => { showAll = false; populateTeams(); writeHash(); render(); };
  $("search").oninput = () => { showAll = false; renderBody(); };
  $("minposs").oninput = () => { showAll = false; writeHash(); renderBody(); };
  $("teamsel").onchange = () => { showAll = false; writeHash(); renderBody(); };
  $("colbtn").onclick = () => {
    const p = $("colpanel"), open = p.classList.toggle("hidden") === false;
    $("colbtn").classList.toggle("on", open);
    $("colbtn").setAttribute("aria-expanded", String(open));
  };
  $("filtbtn").onclick = () => {
    filtMode = !filtMode;
    if (!filtMode) statFilters = {};
    $("filtbtn").classList.toggle("on", filtMode);
    $("filtbtn").setAttribute("aria-pressed", String(filtMode));
    render();
  };
  $("csvbtn").onclick = toCSV;
  $("colreset").onclick = () => { resetCols(); buildColPanel(); render(); };
  $("resetbtn").onclick = () => {
    $("search").value = ""; $("teamsel").value = "";
    statFilters = {}; filtMode = false; showAll = false;
    $("filtbtn").classList.remove("on");
    setView(view);
    writeHash();
  };
  document.querySelectorAll("#viewnav a").forEach(a => {
    a.onclick = e => { e.preventDefault(); setView(a.dataset.view); writeHash(); };
  });
  // Editing the URL only fires hashchange — boot() does NOT re-run — so this
  // has to reapply every piece of hash state, not just the view.
  window.addEventListener("hashchange", () => {
    const hh = readHash();
    applyingHash = true;
    if (hh.y && $("yearsel").querySelector(`option[value="${hh.y}"]`)) $("yearsel").value = hh.y;
    if (hh.w) $("windowsel").value = hh.w;
    if (hh.view && hh.view !== view) {
      setView(hh.view, { sort: hh.sort, dir: hh.dir, mp: hh.mp });
    } else {
      if (hh.mp) $("minposs").value = hh.mp;
      if (hh.sort) { sortKey = hh.sort; sortDesc = hh.dir !== "a"; }
      populateTeams();
      render();
    }
    if (hh.team) $("teamsel").value = hh.team;
    renderBody();
    applyingHash = false;
  });

  setView(h.view || "players", { sort: h.sort, dir: h.dir, mp: h.mp });
  if (h.team) { $("teamsel").value = h.team; renderBody(); }
  writeHash();
}

boot();
