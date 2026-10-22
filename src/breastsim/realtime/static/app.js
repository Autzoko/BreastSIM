// BreastSIM Probe Lab client: 3D anatomy + probe, live B-mode over WebSocket.
// Phantom frame (a, u, v) mm: a = chest wall -> nipple. three.js frame: x = v, y = a, z = u.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const $ = (id) => document.getElementById(id);
const S = {
  config: null, phantom: null, info: null,
  pose: { u: 0, v: 0, rotation: 0, tilt: 0, rock: 0, standoff: 0 },
  params: {}, dirty: false, inFlight: 0, seq: 0, sent: {}, frames: [], lastMeta: null, pendingMeta: null,
};
window.__breastsim = S;  // for debugging / automated checks

// ------------------------------------------------------------------ controls
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
  ["speckle_reduction", "smooth", 0, 3, 0.1],
  ["lateral_f_number", "F#", 1, 4, 0.1],
];
const PRESET_KEYS = ["probe", "depth_mm", "px_x_mm", "px_z_mm", "gain_db", "dynamic_range_db",
  "speckle_reduction", "lateral_f_number", "conform"];

let syncers = [];
function slider(parent, obj, [key, name, min, max, step]) {
  const row = document.createElement("label");
  row.className = "slider";
  row.innerHTML = `<span class="name">${name}</span><input type="range" min="${min}" max="${max}" step="${step}"><span class="val"></span>`;
  const input = row.querySelector("input"), val = row.querySelector(".val");
  const sync = () => { input.value = obj[key]; val.textContent = (+obj[key]).toFixed(step < 1 ? (step < 0.1 ? 2 : 1) : 0); };
  input.addEventListener("input", () => { obj[key] = +input.value; sync(); request(); });
  parent.appendChild(row);
  syncers.push(sync);
}
function buildControls() {
  POSE_SLIDERS.forEach((d) => slider($("pose"), S.pose, d));
  const probeRow = document.createElement("label");
  probeRow.innerHTML = `<span class="muted" style="width:70px">probe</span><select id="probe">${
    Object.keys(S.config.probes).map((p) => `<option>${p}</option>`).join("")}</select>
    <label><input type="checkbox" id="conform"> press skin</label>`;
  $("imaging").appendChild(probeRow);
  $("probe").onchange = (e) => { S.params.probe = e.target.value; request(); };
  $("conform").onchange = (e) => { S.params.conform = e.target.checked; request(); };
  syncers.push(() => { $("probe").value = S.params.probe; $("conform").checked = !!S.params.conform; });
  IMG_SLIDERS.forEach((d) => slider($("imaging"), S.params, d));
  syncAll();
}
const syncAll = () => syncers.forEach((f) => f());

function applyPreset(name) {
  const p = S.config.presets[name];
  PRESET_KEYS.forEach((k) => (S.params[k] = p[k]));
  S.params.preset = name;
  $("preset").value = name;
  syncAll();
  request();
}

// ------------------------------------------------------------------ 3D scene
const three = {};
function initScene() {
  const host = $("scene3d");
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setClearColor(0x0b0e12);
  host.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(40, 1, 1, 5000);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  scene.add(new THREE.HemisphereLight(0xffffff, 0x334455, 1.1));
  const sun = new THREE.DirectionalLight(0xffffff, 1.2);
  sun.position.set(200, 400, 300);
  scene.add(sun);
  const anatomy = new THREE.Group();
  scene.add(anatomy);

  // probe: body + face strip, local x = lateral, y = up (against the beam), z = elevation
  const probe = new THREE.Group();
  const body = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshStandardMaterial({ color: 0xd8dde4, metalness: 0.1, roughness: 0.5 }));
  const face = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshStandardMaterial({ color: 0x4cc38a, emissive: 0x1b5c3d }));
  const marker = new THREE.Mesh(new THREE.SphereGeometry(1.6, 16, 12), new THREE.MeshStandardMaterial({ color: 0x4cc38a, emissive: 0x2a7a52 }));
  probe.add(body, face, marker);
  scene.add(probe);

  // scan plane: 33 columns x 2 rows, textured with the live frame
  const texCanvas = document.createElement("canvas");
  const texture = new THREE.CanvasTexture(texCanvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const planeGeo = new THREE.BufferGeometry();
  const cols = 33, pos = new Float32Array(cols * 2 * 3), uv = new Float32Array(cols * 2 * 2), idx = [];
  for (let i = 0; i < cols; i++) {
    uv.set([i / (cols - 1), 1, i / (cols - 1), 0], i * 4);
    if (i < cols - 1) { const a = 2 * i; idx.push(a, a + 1, a + 2, a + 2, a + 1, a + 3); }
  }
  planeGeo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  planeGeo.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
  planeGeo.setIndex(idx);
  const plane = new THREE.Mesh(planeGeo, new THREE.MeshBasicMaterial({ map: texture, side: THREE.DoubleSide, transparent: true, opacity: 0.92 }));
  scene.add(plane);

  Object.assign(three, { renderer, scene, camera, controls, anatomy, probe, body, face, marker, plane, planeGeo, texCanvas, texture, meshes: {} });
  new ResizeObserver(resize).observe(host);
  resize();
  bindPicking();
  (function loop() {
    controls.update();
    renderer.render(scene, camera);
    requestAnimationFrame(loop);
  })();
}

function resize() {
  const host = $("scene3d"), w = host.clientWidth || 1, h = host.clientHeight || 1;
  three.renderer.setSize(w, h, false);
  three.renderer.domElement.style.width = "100%";
  three.renderer.domElement.style.height = "100%";
  three.camera.aspect = w / h;
  three.camera.updateProjectionMatrix();
}

const T = ([a, u, v]) => new THREE.Vector3(v, a, u);  // phantom (a,u,v) -> three

function decode(b64, Type) {
  const bin = atob(b64), buf = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
  return new Type(buf.buffer);
}

async function loadMeshes() {
  const data = await (await fetch("/api/meshes")).json();
  three.anatomy.clear();
  three.meshes = {};
  for (const [name, m] of Object.entries(data)) {
    if (!m.n_faces) continue;
    const src = decode(m.verts, Float32Array), verts = new Float32Array(src.length);
    for (let i = 0; i < src.length; i += 3) { verts[i] = src[i + 2]; verts[i + 1] = src[i]; verts[i + 2] = src[i + 1]; }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(verts, 3));
    geo.setIndex(new THREE.BufferAttribute(decode(m.faces, Uint32Array), 1));
    geo.computeVertexNormals();
    let mat;
    if (name === "skin") mat = new THREE.MeshStandardMaterial({ color: 0xe9c4a6, transparent: true, opacity: +$("skinOpacity").value, depthWrite: false, side: THREE.DoubleSide, roughness: 0.8 });
    else if (name === "gland") mat = new THREE.MeshStandardMaterial({ color: 0x6fa8ff, transparent: true, opacity: 0.35, depthWrite: false, side: THREE.DoubleSide });
    else mat = new THREE.MeshStandardMaterial({ color: 0xff4d4d, emissive: 0x551010, side: THREE.DoubleSide });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.name = name;
    mesh.renderOrder = name === "skin" ? 2 : name === "gland" ? 1 : 0;
    three.meshes[name] = mesh;
    three.anatomy.add(mesh);
  }
  applyVisibility();
  recenter(true);
}

function applyVisibility() {
  for (const [name, m] of Object.entries(three.meshes)) {
    m.visible = name === "skin" ? $("showSkin").checked : name === "gland" ? $("showGland").checked : $("showLesions").checked;
  }
  three.plane.visible = $("showPlane").checked;
  if (three.meshes.skin) three.meshes.skin.material.opacity = +$("skinOpacity").value;
}

function recenter(full) {
  if (!S.info) return;
  const [A, U, V] = S.info.extent_mm;
  const target = full ? T([A * 0.5, U / 2, V / 2]) : T([S.lastMeta ? S.lastMeta.face_a_mm - 15 : A / 2, S.pose.u, S.pose.v]);
  const r = Math.max(A, U, V) * (full ? 1.6 : 0.9);
  three.controls.target.copy(target);
  three.camera.position.copy(target).add(new THREE.Vector3(r * 0.55, r * 0.75, r * 0.65));
  three.camera.near = r / 100; three.camera.far = r * 20; three.camera.updateProjectionMatrix();
}

function updateProbe3D(meta, bitmap) {
  const [lat, ax, el] = meta.axes, o = meta.origin_mm, spec = S.config.probes[S.params.probe];
  const W = spec.aperture_mm, H = spec.height * 1e3;
  const X = T(lat), Y = T(ax).negate(), Z = T(el).negate();  // right-handed basis
  const m = new THREE.Matrix4().makeBasis(X, Y, Z).setPosition(T(o));
  three.probe.matrixAutoUpdate = false;
  three.probe.matrix.copy(m);
  const handle = Math.max(18, W * 0.15);
  three.body.scale.set(W + 4, handle, H + 6); three.body.position.set(0, handle / 2 + 1.5, 0);
  three.face.scale.set(W, 1.5, H); three.face.position.set(0, 0.75, 0);
  three.marker.position.set(-W / 2 - 3, handle * 0.7, 0);  // orientation marker on the left end

  // scan plane follows the (possibly skin-conforming) line starts
  const offs = meta.plane_offsets_mm, n = offs.length, pos = three.planeGeo.attributes.position.array;
  const vlat = new THREE.Vector3(...lat), vax = new THREE.Vector3(...ax), vo = new THREE.Vector3(...o);
  for (let i = 0; i < n; i++) {
    const x = -W / 2 + (W * i) / (n - 1);
    const top = vo.clone().addScaledVector(vlat, x).addScaledVector(vax, Math.max(0, offs[i]));
    const bot = top.clone().addScaledVector(vax, meta.depth_mm);
    pos.set([top.z, top.x, top.y, bot.z, bot.x, bot.y], i * 6);  // (a,u,v) -> (v,a,u)
  }
  three.planeGeo.attributes.position.needsUpdate = true;
  three.planeGeo.computeBoundingSphere();
  if (bitmap) {
    three.texCanvas.width = bitmap.width; three.texCanvas.height = bitmap.height;
    three.texCanvas.getContext("2d").drawImage(bitmap, 0, 0);
    three.texture.needsUpdate = true;
  }
}

function bindPicking() {
  const el = three.renderer.domElement, ray = new THREE.Raycaster();
  let down = null;
  el.addEventListener("pointerdown", (e) => (down = [e.clientX, e.clientY, e.button]));
  el.addEventListener("pointerup", (e) => {
    if (!down || down[2] !== 0 || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) return;
    const r = el.getBoundingClientRect();
    ray.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), three.camera);
    const targets = [three.meshes.skin, ...Object.entries(three.meshes).filter(([k]) => k.startsWith("lesion")).map(([, m]) => m)].filter(Boolean);
    const hit = ray.intersectObjects(targets, false)[0];
    if (!hit) return;
    S.pose.u = hit.point.z; S.pose.v = hit.point.x;  // three (x=v, y=a, z=u)
    request();
  });
}

// ------------------------------------------------------------------ websocket
let ws;
function connect() {
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.binaryType = "blob";
  ws.onopen = () => { status("connected"); if (S.phantom) load(S.phantom); };
  ws.onclose = () => { status("disconnected — retrying"); S.inFlight = 0; setTimeout(connect, 1500); };
  ws.onmessage = async (ev) => {
    if (typeof ev.data !== "string") {
      const meta = S.pendingMeta;  // bind before awaiting: the next frame's meta may arrive meanwhile
      return onFrame(await createImageBitmap(ev.data), meta);
    }
    const msg = JSON.parse(ev.data);
    if (msg.type === "status") status(msg.msg);
    else if (msg.type === "error") { status(msg.msg); S.inFlight = Math.max(0, S.inFlight - 1); }
    else if (msg.type === "loaded") onLoaded(msg.phantom);
    else if (msg.type === "frame") S.pendingMeta = msg.meta;
  };
}
const status = (t) => ($("status").textContent = t);

function load(pid) {
  S.phantom = pid;
  S.info = null;
  S.inFlight = 0;
  $("phantom").value = pid;
  ws.send(JSON.stringify({ type: "load", phantom: pid }));
}

async function onLoaded(info) {
  S.info = info;
  Object.assign(S.pose, { u: info.start.u, v: info.start.v, rotation: 0, tilt: 0, rock: 0 });
  renderLesions();
  syncAll();
  status(`${info.id} · ${info.description} · ${info.extent_mm.map((x) => x.toFixed(0)).join("×")} mm`);
  await loadMeshes();
  request();
}

function renderLesions() {
  const L = S.info.lesions;
  $("lesions").innerHTML = L.length ? L.map((l) => `
    <div class="lesion"><span><span class="t">#${l.id} ${l.type}</span> · Ø ${l.diameter_mm} mm</span>
      <button data-id="${l.id}">Go</button>
      <span class="muted">a ${l.center_mm[0].toFixed(0)} · u ${l.center_mm[1].toFixed(0)} · v ${l.center_mm[2].toFixed(0)} mm` +
      `${l.depth_below_skin_mm != null ? ` · ${l.depth_below_skin_mm} mm deep` : ""}</span></div>`).join("")
    : `<span class="muted">no lesions in this phantom</span>`;
  $("lesions").querySelectorAll("button").forEach((b) => (b.onclick = () => goLesion(L[+b.dataset.id])));
}

function goLesion(l) {
  S.pose.u = l.center_mm[1]; S.pose.v = l.center_mm[2];
  if (l.depth_below_skin_mm != null && l.depth_below_skin_mm + 10 > S.params.depth_mm) S.params.depth_mm = Math.min(60, Math.ceil(l.depth_below_skin_mm + 12));
  syncAll(); request();
}

function request() {
  S.dirty = true;
  pump();
}
function pump() {
  // two requests in flight hide the network round trip behind rendering
  if (!S.dirty || S.inFlight >= 2 || !S.info || !ws || ws.readyState !== 1) return;
  S.dirty = false;
  S.inFlight += 1;
  const seq = ++S.seq;
  S.sent[seq] = performance.now();
  ws.send(JSON.stringify({ type: "render", seq, pose: S.pose, params: S.params }));
}

function onFrame(bmp, m) {
  S.lastMeta = m;
  const c = $("bmode");
  if (c.width !== bmp.width || c.height !== bmp.height) { c.width = bmp.width; c.height = bmp.height; }
  c.style.aspectRatio = `${bmp.width * m.px_x_mm} / ${bmp.height * m.px_z_mm}`;
  const g = c.getContext("2d");
  g.drawImage(bmp, 0, 0);
  g.fillStyle = "rgba(255,255,255,0.75)"; g.font = `${Math.max(10, bmp.height / 45)}px ui-monospace, monospace`;
  for (let d = 10; d < m.depth_mm; d += 10) {           // depth ruler, 10 mm ticks
    const y = d / m.px_z_mm;
    g.fillRect(c.width - 10, y, 10, Math.max(1, bmp.height / 400));
    g.fillText(`${d}`, c.width - 34, y + 4);
  }
  g.fillStyle = "#4cc38a"; g.beginPath(); g.arc(8, 8, 4, 0, 7); g.fill();  // orientation marker (left end)
  updateProbe3D(m, bmp);
  const now = performance.now(), rtt = S.sent[m.seq] ? now - S.sent[m.seq] : NaN;
  delete S.sent[m.seq];
  S.frames = S.frames.filter((t) => now - t < 1000).concat(now);
  const hit = m.lesion_area_mm2 > 0;
  $("readouts").innerHTML = [
    ["DEPTH", `${(+m.depth_mm).toFixed(0)} mm`], ["ROTATION", `${S.pose.rotation.toFixed(0)}°`],
    ["TILT", `${S.pose.tilt.toFixed(0)}°`], ["ROCK", `${S.pose.rock.toFixed(0)}°`],
    ["PROBE u, v", `${S.pose.u.toFixed(1)}, ${S.pose.v.toFixed(1)}`],
    ["LESION IN PLANE", hit ? `${m.lesion_area_mm2} mm²` : "—"],
  ].map(([k, v], i) => `<div class="${i === 5 && hit ? "hit" : ""}"><b>${k}</b><span>${v}</span></div>`).join("");
  $("perf").textContent = `${S.frames.length} fps · render ${m.render_ms} ms · rtt ${rtt.toFixed(0)} ms · ` +
    `${bmp.width}×${bmp.height} px @ ${(+m.px_x_mm).toFixed(3)}×${(+m.px_z_mm).toFixed(3)} mm · ${S.params.probe}` +
    (hit ? ` · ${m.lesion_types.join(", ")}` : "");
  S.inFlight = Math.max(0, S.inFlight - 1);
  pump();
}

// ------------------------------------------------------------------ keyboard
window.addEventListener("keydown", (e) => {
  if (!S.info || document.activeElement.tagName === "SELECT") return;
  const k = e.key.toLowerCase(), step = e.shiftKey ? 2.5 : 0.5, ang = e.shiftKey ? 5 : 1;
  const r = (S.pose.rotation * Math.PI) / 180, lat = [Math.cos(r), Math.sin(r)], el = [-Math.sin(r), Math.cos(r)];
  const move = (d, s) => { S.pose.u += d[0] * s; S.pose.v += d[1] * s; };
  const clamp = (x) => Math.max(-30, Math.min(30, x));
  const acts = {
    w: () => move(el, step), s: () => move(el, -step), d: () => move(lat, step), a: () => move(lat, -step),
    q: () => (S.pose.rotation = ((S.pose.rotation - ang + 540) % 360) - 180),
    e: () => (S.pose.rotation = ((S.pose.rotation + ang + 540) % 360) - 180),
    r: () => (S.pose.tilt = clamp(S.pose.tilt + ang)), f: () => (S.pose.tilt = clamp(S.pose.tilt - ang)),
    z: () => (S.pose.rock = clamp(S.pose.rock - ang)), x: () => (S.pose.rock = clamp(S.pose.rock + ang)),
  };
  if (!acts[k]) return;
  e.preventDefault();
  acts[k]();
  syncAll(); request();
});

// ------------------------------------------------------------------ init
["showSkin", "showGland", "showLesions", "showPlane", "skinOpacity"].forEach((id) => $(id).addEventListener("input", applyVisibility));
$("recenter").onclick = () => recenter(false);
$("overlay").onchange = (e) => { S.params.overlay = e.target.value; request(); };
$("phantom").onchange = (e) => load(e.target.value);
$("preset").onchange = (e) => applyPreset(e.target.value);
$("save").onclick = async () => {
  const res = await fetch("/api/scene", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ phantom: S.phantom, pose: S.pose, params: S.params }),
  });
  status(`saved ${(await res.json()).path}`);
};

(async () => {
  S.config = await (await fetch("/api/config")).json();
  S.params = { ...S.config.params };
  $("preset").innerHTML = Object.entries(S.config.presets).map(([k, p]) => `<option value="${k}" title="${p.notes}">${p.label}</option>`).join("");
  $("preset").value = S.params.preset;
  const list = await (await fetch("/api/phantoms")).json();
  $("phantom").innerHTML = list.map((p) => `<option value="${p.id}">${p.id}${p.cached ? "" : " (build)"} — ${p.description}</option>`).join("");
  buildControls();
  initScene();
  S.phantom = list.find((p) => p.cached)?.id || list[0]?.id;
  connect();
})();
