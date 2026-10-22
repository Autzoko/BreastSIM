// BreastSIM real-time preview client.
const $ = (id) => document.getElementById(id);
const S = {
  phantom: null, info: null, probes: {}, overview: null,
  pose: { u: 0, v: 0, rotation: 0, tilt: 0, rock: 0, standoff: 0 },
  params: {}, dirty: false, inFlight: 0, pendingMeta: null, frames: [], sent: [],
};

const POSE_SLIDERS = [
  ["rotation", "rotation °", -180, 180, 1],
  ["tilt", "tilt °", -30, 30, 1],
  ["rock", "rock °", -30, 30, 1],
  ["standoff", "gel mm", 0, 10, 0.5],
];
const IMG_SLIDERS = [
  ["depth_mm", "depth mm", 20, 60, 1],
  ["gain_db", "gain dB", -40, 20, 1],
  ["dynamic_range_db", "DR dB", 30, 80, 1],
  ["tgc_db_cm_mhz", "TGC", 0, 1.5, 0.05],
  ["specular", "specular", 0, 40, 1],
  ["lateral_f_number", "F#", 1, 4, 0.1],
];

function slider(parent, obj, [key, name, min, max, step]) {
  const row = document.createElement("label");
  row.className = "slider";
  row.innerHTML = `<span class="name">${name}</span><input type="range" min="${min}" max="${max}" step="${step}"><span class="val"></span>`;
  const input = row.querySelector("input"), val = row.querySelector(".val");
  const sync = () => { input.value = obj[key]; val.textContent = (+obj[key]).toFixed(step < 1 ? 2 : 0); };
  input.addEventListener("input", () => { obj[key] = +input.value; sync(); request(); });
  parent.appendChild(row);
  return sync;
}

let syncers = [];
function buildControls() {
  const pose = $("pose"), img = $("imaging");
  pose.innerHTML = "<div class='muted small'>Probe pose</div>";
  img.innerHTML = "<div class='muted small'>Imaging</div>";
  const probeRow = document.createElement("label");
  probeRow.innerHTML = `Probe <select id="probe">${Object.keys(S.probes).map((p) => `<option>${p}</option>`).join("")}</select>`;
  img.appendChild(probeRow);
  $("probe").value = S.params.probe;
  $("probe").onchange = (e) => { S.params.probe = e.target.value; request(); };
  syncers = POSE_SLIDERS.map((d) => slider(pose, S.pose, d)).concat(IMG_SLIDERS.map((d) => slider(img, S.params, d)));
  syncAll();
}
const syncAll = () => syncers.forEach((f) => f());

// ------------------------------------------------------------- websocket
let ws;
function connect() {
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.binaryType = "blob";
  ws.onopen = () => { status("connected"); if (S.phantom) load(S.phantom); };
  ws.onclose = () => { status("disconnected — retrying"); S.inFlight = 0; setTimeout(connect, 1500); };
  ws.onmessage = async (ev) => {
    if (typeof ev.data !== "string") return drawFrame(await createImageBitmap(ev.data));
    const msg = JSON.parse(ev.data);
    if (msg.type === "status") status(msg.msg);
    else if (msg.type === "loaded") onLoaded(msg.phantom);
    else if (msg.type === "frame") S.pendingMeta = msg.meta;
  };
}
const status = (t) => ($("status").textContent = t);

function load(pid) {
  S.phantom = pid;
  S.info = null;
  ws.send(JSON.stringify({ type: "load", phantom: pid }));
}

async function onLoaded(info) {
  S.info = info;
  Object.assign(S.pose, { u: info.start.u, v: info.start.v, rotation: 0, tilt: 0, rock: 0 });
  $("pdesc").textContent = `${info.description} · ${info.extent_mm.map((x) => x.toFixed(0)).join("×")} mm @ ${info.voxel_mm} mm`;
  const blob = await (await fetch("/api/overview")).blob();
  S.overview = await createImageBitmap(blob);
  syncAll();
  status(`${info.id} ready`);
  request();
}

function request() {
  S.dirty = true;
  pump();
  drawOverview();
}
function pump() {
  // two requests in flight hide the network round trip behind rendering
  if (!S.dirty || S.inFlight >= 2 || !S.info || ws.readyState !== 1) return;
  S.dirty = false;
  S.inFlight += 1;
  S.sent.push(performance.now());
  ws.send(JSON.stringify({ type: "render", pose: S.pose, params: S.params }));
}

function drawFrame(bmp) {
  const c = $("bmode");
  if (c.width !== bmp.width || c.height !== bmp.height) { c.width = bmp.width; c.height = bmp.height; }
  const g = c.getContext("2d");
  g.drawImage(bmp, 0, 0);
  const m = S.pendingMeta || {};
  // depth ruler every 10 mm
  g.fillStyle = "rgba(255,255,255,0.7)"; g.font = "11px ui-monospace, monospace";
  for (let d = 10; d < m.depth_mm; d += 10) {
    const y = d / m.pixel_mm;
    g.fillRect(c.width - 8, y, 8, 1);
    g.fillText(`${d}`, c.width - 26, y + 4);
  }
  const now = performance.now();
  S.frames = S.frames.filter((t) => now - t < 1000).concat(now);
  $("readout").textContent =
    `${S.frames.length} fps · render ${m.render_ms} ms · rtt ${(now - S.sent.shift()).toFixed(0)} ms · ` +
    `u ${S.pose.u.toFixed(1)} v ${S.pose.v.toFixed(1)} mm · face a ${(+m.face_a_mm).toFixed(1)} mm · lesion px ${m.lesion_pixels}`;
  S.inFlight = Math.max(0, S.inFlight - 1);
  pump();
}

// -------------------------------------------------------------- overview
function ovMap() {
  const c = $("overview"), [, U, V] = S.info.shape, vox = S.info.voxel_mm;
  const sc = Math.min(c.width / V, c.height / U), ox = (c.width - V * sc) / 2, oy = (c.height - U * sc) / 2;
  return {
    toPx: (u, v) => [ox + (v / vox) * sc, oy + (u / vox) * sc],
    toMm: (x, y) => [((y - oy) / sc) * vox, ((x - ox) / sc) * vox],
    sc, ox, oy, U, V,
  };
}

function drawOverview() {
  const c = $("overview"), g = c.getContext("2d");
  g.fillStyle = "#000"; g.fillRect(0, 0, c.width, c.height);
  if (!S.info || !S.overview) return;
  const m = ovMap();
  g.drawImage(S.overview, m.ox, m.oy, m.V * m.sc, m.U * m.sc);
  const spec = S.probes[S.params.probe], w = spec.n_elements * spec.pitch * 1e3, h = spec.height * 1e3;
  const r = (S.pose.rotation * Math.PI) / 180;
  const lat = [Math.cos(r), Math.sin(r)], el = [-Math.sin(r), Math.cos(r)];
  const corner = (a, b) => m.toPx(S.pose.u + a * lat[0] * w / 2 + b * el[0] * h / 2, S.pose.v + a * lat[1] * w / 2 + b * el[1] * h / 2);
  g.strokeStyle = "#4cc38a"; g.lineWidth = 2; g.beginPath();
  [[-1, -1], [1, -1], [1, 1], [-1, 1]].forEach(([a, b], i) => { const [x, y] = corner(a, b); i ? g.lineTo(x, y) : g.moveTo(x, y); });
  g.closePath(); g.stroke();
  const [x0, y0] = corner(-1, 0); g.fillStyle = "#4cc38a"; g.beginPath(); g.arc(x0, y0, 3, 0, 7); g.fill();  // left edge marker
}

$("overview").addEventListener("click", (e) => {
  if (!S.info) return;
  const c = e.target, rect = c.getBoundingClientRect();
  const [u, v] = ovMap().toMm((e.clientX - rect.left) * (c.width / rect.width), (e.clientY - rect.top) * (c.height / rect.height));
  Object.assign(S.pose, { u, v });
  request();
});
$("overview").addEventListener("wheel", (e) => {
  e.preventDefault();
  S.pose.rotation = ((S.pose.rotation + Math.sign(e.deltaY) * 5 + 540) % 360) - 180;
  syncAll(); request();
}, { passive: false });

// -------------------------------------------------------------- keyboard
window.addEventListener("keydown", (e) => {
  if (!S.info || ["INPUT", "SELECT"].includes(document.activeElement.tagName) && e.key.length > 1) return;
  const k = e.key.toLowerCase(), step = e.shiftKey ? 2.5 : 0.5, ang = e.shiftKey ? 5 : 1;
  const r = (S.pose.rotation * Math.PI) / 180, lat = [Math.cos(r), Math.sin(r)], el = [-Math.sin(r), Math.cos(r)];
  const move = (d, s) => { S.pose.u += d[0] * s; S.pose.v += d[1] * s; };
  const acts = {
    w: () => move(el, step), s: () => move(el, -step), d: () => move(lat, step), a: () => move(lat, -step),
    q: () => (S.pose.rotation -= ang), e: () => (S.pose.rotation += ang),
    r: () => (S.pose.tilt = Math.min(30, S.pose.tilt + ang)), f: () => (S.pose.tilt = Math.max(-30, S.pose.tilt - ang)),
    z: () => (S.pose.rock = Math.max(-30, S.pose.rock - ang)), x: () => (S.pose.rock = Math.min(30, S.pose.rock + ang)),
  };
  if (!acts[k]) return;
  e.preventDefault();
  acts[k]();
  syncAll(); request();
});

$("overlay").onchange = (e) => { S.params.overlay = e.target.value; request(); };
$("phantom").onchange = (e) => load(e.target.value);
$("save").onclick = async () => {
  const res = await fetch("/api/scene", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ phantom: S.phantom, pose: S.pose, params: S.params }),
  });
  status(`saved ${(await res.json()).path}`);
};

// ------------------------------------------------------------------ init
(async () => {
  S.probes = await (await fetch("/api/probes")).json();
  S.params = await (await fetch("/api/params")).json();
  const list = await (await fetch("/api/phantoms")).json();
  $("phantom").innerHTML = list.map((p) => `<option value="${p.id}">${p.id}${p.cached ? "" : " (build)"}</option>`).join("");
  buildControls();
  S.phantom = list.find((p) => p.cached)?.id || list[0]?.id;
  $("phantom").value = S.phantom;
  connect();
})();
