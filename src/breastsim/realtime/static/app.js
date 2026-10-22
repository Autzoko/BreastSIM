// BreastSIM Probe Lab client: 3D anatomy + probe, live B-mode and ground truth over WebSocket.
// Phantom frame (a, u, v) mm: a = chest wall -> nipple, u assumed superior, v toward the patient's left.
// three.js frame: x = v, y = a, z = u.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const $ = (id) => document.getElementById(id);
const S = {
  config: null, phantom: null, info: null, side: "RT",
  pose: { u: 0, v: 0, rotation: 0, tilt: 0, rock: 0, standoff: 0 },
  params: {}, dirty: false, inFlight: 0, seq: 0, sent: {}, frames: [], lastMeta: null,
  frame: null, caliper: null, meshCount: 0,
};
window.__breastsim = S;  // debugging / automated checks

// ================================================================== controls
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
  ["specular", "specular", 0, 100, 1],
  ["speckle_reduction", "smooth", 0, 3, 0.1],
  ["lateral_f_number", "F#", 1, 4, 0.1],
  ["shadow_blur_mm", "shadow fill", 0, 4, 0.1],
];
const PRESET_KEYS = ["probe", "depth_mm", "px_x_mm", "px_z_mm", "gain_db", "dynamic_range_db",
  "speckle_reduction", "lateral_f_number", "conform"];

const syncers = [];
function slider(parent, obj, [key, name, min, max, step]) {
  const row = document.createElement("label");
  row.className = "slider";
  row.innerHTML = `<span class="name">${name}</span><input type="range" min="${min}" max="${max}" step="${step}"><span class="val"></span>`;
  const input = row.querySelector("input"), val = row.querySelector(".val");
  const digits = step >= 1 ? 0 : step >= 0.1 ? 1 : 2;
  const sync = () => { input.value = obj[key]; val.textContent = (+obj[key]).toFixed(digits); };
  input.addEventListener("input", () => { obj[key] = +input.value; sync(); request(); });
  parent.appendChild(row);
  syncers.push(sync);
}
function buildControls() {
  POSE_SLIDERS.forEach((d) => slider($("pose"), S.pose, d));
  const probeRow = document.createElement("label");
  probeRow.innerHTML = `<span class="muted" style="width:74px">probe</span><select id="probe">${
    Object.keys(S.config.probes).map((p) => `<option>${p}</option>`).join("")}</select>
    <label title="probe presses the skin flat"><input type="checkbox" id="conform"> press skin</label>`;
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

// ================================================================ clinical orientation
// Clock face as seen facing the patient: 12 = superior (+u), 3 = patient's left (+v).
function clockInfo() {
  if (!S.info) return null;
  const [, nu, nv] = S.info.nipple_mm, du = S.pose.u - nu, dv = S.pose.v - nv;
  const dist = Math.hypot(du, dv);
  const ang = (Math.atan2(dv, du) * 180) / Math.PI;               // 0 at 12 o'clock, clockwise
  const halfHours = Math.round((((ang % 360) + 360) % 360) / 15) % 24;
  const h = Math.floor(halfHours / 2) || 12, clock = `${h}:${halfHours % 2 ? "30" : "00"}`;
  // probe orientation relative to the nipple direction
  const arr = ((S.pose.rotation % 180) + 180) % 180, rad = ((ang % 180) + 180) % 180;
  const d = Math.min(Math.abs(arr - rad), 180 - Math.abs(arr - rad));
  const orient = dist < 5 ? (Math.abs(arr - 90) < 20 ? "transverse" : arr < 20 || arr > 160 ? "sagittal" : "oblique")
    : d < 20 ? "radial" : d > 70 ? "antiradial" : "oblique";
  return { clock: dist < 5 ? "SA" : clock, cm: dist / 10, orient, ang };
}

// ================================================================ 3D scene
const three = {};
const T = ([a, u, v]) => new THREE.Vector3(v, a, u);  // phantom (a,u,v) -> three

function textSprite(text, color = "#e6e9ee", size = 6) {
  const c = document.createElement("canvas"), g = c.getContext("2d");
  g.font = "bold 48px system-ui, sans-serif";
  c.width = Math.ceil(g.measureText(text).width) + 16; c.height = 64;
  g.font = "bold 48px system-ui, sans-serif"; g.fillStyle = color; g.textBaseline = "middle";
  g.fillText(text, 8, 34);
  const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace;
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true }));
  sp.scale.set((size * c.width) / c.height, size, 1);
  sp.renderOrder = 10;
  return sp;
}

function initScene() {
  const host = $("scene3d");
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setClearColor(0x080a0d);
  host.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, 1, 1, 5000);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  scene.add(new THREE.HemisphereLight(0xffffff, 0x334455, 1.1));
  const sun = new THREE.DirectionalLight(0xffffff, 1.3);
  sun.position.set(200, 400, 300);
  scene.add(sun);
  const anatomy = new THREE.Group(), guides = new THREE.Group();
  scene.add(anatomy, guides);

  // probe: local x = array (marker at -x), y = handle (against the beam), z = elevation
  const probe = new THREE.Group();
  const body = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshStandardMaterial({ color: 0xd8dde4, roughness: 0.5, transparent: true, opacity: 0.5, depthWrite: false }));
  const face = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshStandardMaterial({ color: 0x4cc38a, emissive: 0x1b5c3d }));
  const marker = new THREE.Mesh(new THREE.SphereGeometry(1.6, 16, 12), new THREE.MeshStandardMaterial({ color: 0x4cc38a, emissive: 0x2a7a52 }));
  probe.add(body, face, marker);
  probe.matrixAutoUpdate = false;
  scene.add(probe);

  // scan plane: 33 columns x 2 rows, textured with the live frame
  const texCanvas = document.createElement("canvas");
  const texture = new THREE.CanvasTexture(texCanvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const planeGeo = new THREE.BufferGeometry();
  const cols = 33, idx = [], uv = new Float32Array(cols * 4);
  for (let i = 0; i < cols; i++) {
    uv.set([i / (cols - 1), 1, i / (cols - 1), 0], i * 4);
    if (i < cols - 1) { const a = 2 * i; idx.push(a, a + 1, a + 2, a + 2, a + 1, a + 3); }
  }
  planeGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(cols * 6), 3));
  planeGeo.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
  planeGeo.setIndex(idx);
  const plane = new THREE.Mesh(planeGeo, new THREE.MeshBasicMaterial({ map: texture, side: THREE.DoubleSide, transparent: true, opacity: 0.95 }));
  plane.renderOrder = 3;
  scene.add(plane);

  Object.assign(three, { renderer, scene, camera, controls, anatomy, guides, probe, body, face, marker, plane, planeGeo, texCanvas, texture, meshes: {} });
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
  Object.assign(three.renderer.domElement.style, { width: "100%", height: "100%" });
  three.camera.aspect = w / h;
  three.camera.updateProjectionMatrix();
}

function decode(b64, Type) {
  const bin = atob(b64), buf = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
  return new Type(buf.buffer);
}

function disposeGroup(g) {
  g.traverse((o) => { o.geometry?.dispose(); o.material?.map?.dispose(); o.material?.dispose(); });
  g.clear();
}

async function loadMeshes() {
  const data = await (await fetch("/api/meshes")).json();
  disposeGroup(three.anatomy);
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
    if (name === "skin") mat = new THREE.MeshStandardMaterial({ color: 0xe9c4a6, transparent: true, opacity: +$("skinOpacity").value, depthWrite: false, side: THREE.DoubleSide, roughness: 0.85 });
    else if (name === "gland") mat = new THREE.MeshStandardMaterial({ color: 0x6fa8ff, transparent: true, opacity: 0.3, depthWrite: false, side: THREE.DoubleSide });
    else mat = new THREE.MeshStandardMaterial({ color: 0xff4d4d, emissive: 0x551010, side: THREE.DoubleSide });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.name = name;
    mesh.renderOrder = name === "skin" ? 2 : name === "gland" ? 1 : 0;
    three.meshes[name] = mesh;
    three.anatomy.add(mesh);
  }
  S.meshCount = Object.keys(three.meshes).length;
  buildGuides();
  applyVisibility();
  recenter(true);
}

function buildGuides() {
  disposeGroup(three.guides);
  const [A, U, V] = S.info.extent_mm, g = three.guides, n = S.info.nipple_mm;
  // chest wall reference plane (a = 0)
  const grid = new THREE.GridHelper(Math.max(U, V), Math.round(Math.max(U, V) / 10), 0x3a4452, 0x222a33);
  grid.position.copy(T([0, U / 2, V / 2]));
  g.add(grid);
  const chest = textSprite("chest wall", "#8b95a3", 4); chest.position.copy(T([0, U + 4, V / 2])); g.add(chest);
  // nipple marker
  const nip = new THREE.Mesh(new THREE.SphereGeometry(1.8, 16, 12), new THREE.MeshStandardMaterial({ color: 0xff9ec7, emissive: 0x552233 }));
  nip.position.copy(T(n)); g.add(nip);
  const nl = textSprite("nipple", "#ff9ec7", 4); nl.position.copy(T([n[0] + 6, n[1], n[2]])); g.add(nl);
  // anatomical direction labels around the breast (assumed phantom frame, see help)
  const mid = [A * 0.35, U / 2, V / 2], r = Math.max(U, V) / 2 + 10;
  const plusVLateral = S.side === "LT";  // +v is toward the patient's left: lateral for a left breast
  [["S", [0, r, 0]], ["I", [0, -r, 0]], [plusVLateral ? "LAT" : "MED", [0, 0, r]], [plusVLateral ? "MED" : "LAT", [0, 0, -r]], ["A", [A * 0.75 + 12, 0, 0]]]
    .forEach(([t, d]) => { const sp = textSprite(t, "#f5d76e", 6); sp.position.copy(T([mid[0] + d[0], mid[1] + d[1], mid[2] + d[2]])); g.add(sp); });
  // lesion tags
  S.info.lesions.forEach((l) => {
    const sp = textSprite(`#${l.id}`, "#ff8080", 4.5);
    sp.position.copy(T([l.center_mm[0] + l.diameter_mm / 2 + 4, l.center_mm[1], l.center_mm[2]]));
    sp.userData.lesion = true;
    g.add(sp);
  });
}

function applyVisibility() {
  for (const [name, m] of Object.entries(three.meshes)) {
    m.visible = name === "skin" ? $("showSkin").checked : name === "gland" ? $("showGland").checked : $("showLesions").checked;
  }
  three.plane.visible = $("showPlane").checked;
  three.guides.visible = $("showGuides").checked;
  three.guides.children.forEach((c) => { if (c.userData.lesion) c.visible = $("showLesions").checked; });
  if (three.meshes.skin) three.meshes.skin.material.opacity = +$("skinOpacity").value;
}

function recenter(full) {
  if (!S.info) return;
  const [A, U, V] = S.info.extent_mm;
  const target = full ? T([A * 0.45, U / 2, V / 2]) : T([S.lastMeta ? S.lastMeta.face_a_mm - 15 : A / 2, S.pose.u, S.pose.v]);
  const r = Math.max(A, U, V) * (full ? 1.7 : 0.9);
  three.controls.target.copy(target);
  three.camera.position.copy(target).add(new THREE.Vector3(r * 0.5, r * 0.8, r * 0.6));
  three.camera.near = r / 100; three.camera.far = r * 20; three.camera.updateProjectionMatrix();
}

function updateProbe3D(meta, bitmap) {
  const [lat, ax, el] = meta.axes, o = meta.origin_mm, spec = S.config.probes[S.params.probe];
  const W = spec.aperture_mm, H = spec.height * 1e3;
  const X = T(lat), Y = T(ax).negate(), Z = T(el).negate();  // right-handed basis
  three.probe.matrix.copy(new THREE.Matrix4().makeBasis(X, Y, Z).setPosition(T(o)));
  const handle = 12;
  three.body.scale.set(W + 2, handle, H + 3); three.body.position.set(0, handle / 2 + 1.5, 0);
  three.face.scale.set(W, 1.5, H); three.face.position.set(0, 0.75, 0);
  three.marker.position.set(-W / 2 - 3, handle * 0.7, 0);  // orientation marker = image left
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
  const c = three.texCanvas;
  if (c.width !== bitmap.width || c.height !== bitmap.height) {
    c.width = bitmap.width; c.height = bitmap.height;
    three.texture.dispose();  // GPU storage is immutable: reallocate at the new size
  }
  c.getContext("2d").drawImage(bitmap, 0, 0);
  three.texture.needsUpdate = true;
}

function bindPicking() {
  const el = three.renderer.domElement, ray = new THREE.Raycaster();
  let down = null;
  el.addEventListener("pointerdown", (e) => (down = [e.clientX, e.clientY, e.button]));
  el.addEventListener("pointerup", (e) => {
    if (!down || down[2] !== 0 || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) return;
    const r = el.getBoundingClientRect();
    ray.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), three.camera);
    const hit = three.meshes.skin && ray.intersectObject(three.meshes.skin, false)[0];
    if (!hit) return;
    S.pose.u = hit.point.z; S.pose.v = hit.point.x;  // three (x=v, y=a, z=u)
    request();
  });
}

// ================================================================ websocket
let ws;
function connect() {
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.binaryType = "blob";
  ws.onopen = () => { status("connected"); if (S.phantom) load(S.phantom); };
  ws.onclose = () => { status("disconnected — retrying…"); S.inFlight = 0; setTimeout(connect, 1500); };
  ws.onmessage = (ev) => {
    if (typeof ev.data !== "string") {
      const f = S.frame;                  // binaries follow their frame message in order
      if (!f) return;
      f.blobs.push(ev.data);
      if (f.blobs.length === f.names.length) { S.frame = null; onFrame(f); }
      return;
    }
    const msg = JSON.parse(ev.data);
    if (msg.type === "frame") S.frame = { meta: msg.meta, names: msg.binaries || ["bmode"], blobs: [] };
    else if (msg.type === "loading") showLoading(msg);
    else if (msg.type === "loaded") onLoaded(msg.phantom);
    else if (msg.type === "error") { hideLoading(); status(msg.msg); S.inFlight = Math.max(0, S.inFlight - 1); }
    else if (msg.type === "status") status(msg.msg);
  };
}
const status = (t) => ($("status").textContent = t);

let loadTimer = null;
function showLoading(msg) {
  $("loading").hidden = false;
  $("loadTitle").textContent = `Loading ${msg.phantom || S.phantom}`;
  const note = msg.cached === false ? " — first open: building the preview cache (a few minutes)" : "";
  const t0 = performance.now() - (msg.elapsed || 0) * 1000;
  clearInterval(loadTimer);
  const tick = () => ($("loadMsg").textContent = `${msg.msg}${note} · ${Math.round((performance.now() - t0) / 1000)} s`);
  tick();
  loadTimer = setInterval(tick, 1000);
}
function hideLoading() { clearInterval(loadTimer); $("loading").hidden = true; }

function load(pid) {
  S.phantom = pid;
  S.info = null;
  S.inFlight = 0;
  S.caliper = null;
  $("phantom").value = pid;
  showLoading({ phantom: pid, msg: "requesting", elapsed: 0 });
  ws.send(JSON.stringify({ type: "load", phantom: pid }));
}

async function refreshPhantomList() {
  const list = await (await fetch("/api/phantoms")).json();
  $("phantom").innerHTML = list.map((p) => `<option value="${p.id}">${p.id} — ${p.description}${p.cached ? "" : " (not cached)"}</option>`).join("");
  if (S.phantom) $("phantom").value = S.phantom;
  return list;
}

async function onLoaded(info) {
  if (info.id !== S.phantom) return;       // a newer selection superseded this one
  S.info = info;
  Object.assign(S.pose, { u: info.start.u, v: info.start.v, rotation: 0, tilt: 0, rock: 0 });
  renderLesions();
  syncAll();
  status(`${info.id} · ${info.description} · ${info.extent_mm.map((x) => x.toFixed(0)).join("×")} mm`);
  await loadMeshes();
  hideLoading();
  refreshPhantomList();
  request();
}

function renderLesions() {
  const L = S.info.lesions;
  $("lesions").innerHTML = L.length ? L.map((l) => `
    <div class="lesion"><span><span class="t">#${l.id} ${l.type}</span> · Ø ${l.diameter_mm} mm</span>
      <button data-id="${l.id}">Go</button>
      <span class="muted">${l.parts.length > 1 ? l.parts.join(" + ") + " · " : ""}a ${l.center_mm[0].toFixed(0)} · u ${l.center_mm[1].toFixed(0)} · v ${l.center_mm[2].toFixed(0)} mm` +
      `${l.depth_below_skin_mm != null ? ` · ${l.depth_below_skin_mm} mm deep` : ""}</span></div>`).join("")
    : `<span class="muted">no lesions in this phantom</span>`;
  $("lesions").querySelectorAll("button").forEach((b) => (b.onclick = () => goLesion(L[+b.dataset.id])));
}

function goLesion(l) {
  S.pose.u = l.center_mm[1]; S.pose.v = l.center_mm[2];
  if (l.depth_below_skin_mm != null && l.depth_below_skin_mm + 10 > S.params.depth_mm) S.params.depth_mm = Math.min(60, Math.ceil(l.depth_below_skin_mm + 12));
  syncAll(); request();
}

function request() { S.dirty = true; pump(); }
function pump() {
  // two requests in flight hide the network round trip behind rendering
  if (!S.dirty || S.inFlight >= 2 || !S.info || !ws || ws.readyState !== 1) return;
  S.dirty = false;
  S.inFlight += 1;
  const seq = ++S.seq;
  S.sent[seq] = performance.now();
  ws.send(JSON.stringify({ type: "render", seq, pose: S.pose, params: S.params }));
}

// ================================================================ frames
async function onFrame(f) {
  S.inFlight = Math.max(0, S.inFlight - 1);
  const m = f.meta;
  if (m.phantom !== S.phantom) { pump(); return; }  // stale frame from the previous phantom
  const bmp = await createImageBitmap(f.blobs[0]);
  const lab = f.blobs[1] ? await createImageBitmap(f.blobs[1]) : null;
  S.lastMeta = m;
  const c = $("bmode");
  if (c.width !== bmp.width || c.height !== bmp.height) { c.width = bmp.width; c.height = bmp.height; }
  c.getContext("2d").drawImage(bmp, 0, 0);
  const aspect = `${bmp.width * m.px_x_mm} / ${bmp.height * m.px_z_mm}`;
  c.style.aspectRatio = aspect;
  if (lab) drawGT(lab, aspect);
  drawAnnotations();
  updateProbe3D(m, bmp);
  const now = performance.now(), rtt = S.sent[m.seq] ? now - S.sent[m.seq] : NaN;
  delete S.sent[m.seq];
  S.frames = S.frames.filter((t) => now - t < 1000).concat(now);
  const ck = clockInfo(), hit = m.lesion_area_mm2 > 0;
  $("readouts").innerHTML = [
    ["POSITION", ck ? `${S.side} ${ck.clock} ${ck.clock === "SA" ? "" : ck.cm.toFixed(1) + " cm FN"}` : "—"],
    ["ORIENT.", ck ? ck.orient : "—"],
    ["DEPTH", `${(+m.depth_mm).toFixed(0)} mm`], ["ROT / TILT / ROCK", `${S.pose.rotation.toFixed(0)}° / ${S.pose.tilt.toFixed(0)}° / ${S.pose.rock.toFixed(0)}°`],
    ["PROBE u, v", `${S.pose.u.toFixed(1)}, ${S.pose.v.toFixed(1)}`],
    ["SKIN → FACE", `${m.skin_offset_mm[0].toFixed(1)}–${m.skin_offset_mm[1].toFixed(1)} mm`],
    ["LESION IN PLANE", hit ? `${m.lesion_area_mm2} mm²` : "—"],
  ].map(([k, v], i) => `<div class="${i === 6 && hit ? "hit" : ""}"><b>${k}</b><span>${v}</span></div>`).join("");
  $("perf").textContent = `${S.frames.length} fps · render ${m.render_ms} ms · rtt ${rtt.toFixed(0)} ms · ` +
    `${bmp.width}×${bmp.height} px @ ${(+m.px_x_mm).toFixed(3)}×${(+m.px_z_mm).toFixed(3)} mm · ${S.params.probe}` +
    (hit ? ` · ${m.lesion_types.join(", ")}` : "");
  pump();
}

function drawGT(lab, aspect) {
  const c = $("gt");
  if (c.width !== lab.width || c.height !== lab.height) { c.width = lab.width; c.height = lab.height; }
  c.style.aspectRatio = aspect;
  const g = c.getContext("2d", { willReadFrequently: true });
  g.drawImage(lab, 0, 0);
  const im = g.getImageData(0, 0, c.width, c.height), d = im.data, colors = S.config.tissue_colors, present = new Set();
  for (let i = 0; i < d.length; i += 4) {
    const code = d[i], col = colors[code] || [255, 0, 255];
    present.add(code);
    d[i] = col[0]; d[i + 1] = col[1]; d[i + 2] = col[2]; d[i + 3] = 255;
  }
  g.putImageData(im, 0, 0);
  $("legend").innerHTML = [...present].sort((a, b) => a - b).map((k) =>
    `<span style="--c: rgb(${(colors[k] || [255, 0, 255]).join(",")})">${S.config.tissue_names[k] || k}</span>`).join("");
}

// scanner-style annotations, calipers and body marker on a separate canvas
function drawAnnotations() {
  const m = S.lastMeta, img = $("bmode"), c = $("annot");
  if (!m) return;
  const r = img.getBoundingClientRect(), dpr = devicePixelRatio || 1;
  c.width = Math.round(r.width * dpr); c.height = Math.round(r.height * dpr);
  const g = c.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, r.width, r.height);
  const sx = r.width / m.width_mm, sz = r.height / m.depth_mm;   // px per mm
  if ($("showAnnot").checked) {
    g.fillStyle = g.strokeStyle = "rgba(245,215,110,0.9)";
    g.font = "11px ui-monospace, monospace"; g.lineWidth = 1;
    for (let d = 5; d < m.depth_mm; d += 5) {                    // depth scale
      const y = d * sz, major = d % 10 === 0;
      g.fillRect(r.width - (major ? 10 : 5), y, major ? 10 : 5, 1);
      if (major) g.fillText(`${d}`, r.width - 30, y + 4);
    }
    for (let x = 10; x < m.width_mm; x += 10) g.fillRect(x * sx, 0, 1, 6);  // lateral scale
    const spec = S.config.probes[S.params.probe], fy = spec.elev_focus * 1e3 * sz;
    g.beginPath(); g.moveTo(r.width - 14, fy); g.lineTo(r.width - 20, fy - 4); g.lineTo(r.width - 20, fy + 4); g.fill();  // elevation focus
    g.fillStyle = "#4cc38a"; g.beginPath(); g.arc(9, 9, 4, 0, 7); g.fill();   // probe marker side
    g.fillStyle = "rgba(230,233,238,0.85)";
    const lines = [S.config.presets[S.params.preset]?.label || "custom",
      `${S.params.probe}  ${(spec.fc / 1e6).toFixed(1)} MHz`,
      `D ${m.depth_mm.toFixed(0)}mm  G ${S.params.gain_db}dB  DR ${S.params.dynamic_range_db}`];
    lines.forEach((t, i) => g.fillText(t, 18, 14 + i * 13));
    drawBodyMarker(g, 10, r.height - 78, 64);
  }
  if (S.caliper) {                                               // caliper in image mm
    const [a, b] = S.caliper, px = (p) => [p[0] * sx, p[1] * sz];
    const [x0, y0] = px(a), [x1, y1] = px(b);
    g.strokeStyle = g.fillStyle = "#4cc3ff"; g.lineWidth = 1.5;
    g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y1); g.stroke();
    [[x0, y0], [x1, y1]].forEach(([x, y]) => { g.beginPath(); g.moveTo(x - 5, y); g.lineTo(x + 5, y); g.moveTo(x, y - 5); g.lineTo(x, y + 5); g.stroke(); });
    g.font = "12px ui-monospace, monospace";
    g.fillText(`${Math.hypot(b[0] - a[0], b[1] - a[1]).toFixed(1)} mm`, (x0 + x1) / 2 + 6, (y0 + y1) / 2 - 6);
  }
}

function drawBodyMarker(g, x, y, s) {
  const ck = clockInfo();
  const cx = x + s / 2, cy = y + s / 2, R = s / 2 - 4;
  g.save();
  g.strokeStyle = "rgba(230,233,238,0.85)"; g.fillStyle = "rgba(0,0,0,0.5)"; g.lineWidth = 1.2;
  g.beginPath(); g.arc(cx, cy, R, 0, 7); g.fill(); g.stroke();
  g.fillStyle = "#ff9ec7"; g.beginPath(); g.arc(cx, cy, 2.5, 0, 7); g.fill();
  g.fillStyle = "rgba(230,233,238,0.85)"; g.font = "10px system-ui"; g.fillText(S.side, x, y - 2);
  if (ck) {
    // probe: drawn at its clock position, distance scaled to the breast radius (~6 cm)
    const rr = Math.min(R - 4, (ck.cm / 6) * R), th = (ck.ang * Math.PI) / 180;
    const px = cx + rr * Math.sin(th), py = cy - rr * Math.cos(th);
    const rot = (S.pose.rotation * Math.PI) / 180, L = 9;
    const dx = Math.sin(rot) * L, dy = -Math.cos(rot) * L;           // array direction on the face (u up, v right)
    g.strokeStyle = "#4cc38a"; g.lineWidth = 3;
    g.beginPath(); g.moveTo(px - dx, py - dy); g.lineTo(px + dx, py + dy); g.stroke();
    g.fillStyle = "#4cc38a"; g.beginPath(); g.arc(px - dx, py - dy, 2.2, 0, 7); g.fill();  // marker end
  }
  g.restore();
}

function bindCaliper() {
  const c = $("annot");
  let start = null;
  const toMm = (e) => {
    const r = c.getBoundingClientRect(), m = S.lastMeta;
    return [((e.clientX - r.left) / r.width) * m.width_mm, ((e.clientY - r.top) / r.height) * m.depth_mm];
  };
  c.addEventListener("pointerdown", (e) => { if (!S.lastMeta) return; start = toMm(e); S.caliper = [start, start]; c.setPointerCapture(e.pointerId); });
  c.addEventListener("pointermove", (e) => { if (start) { S.caliper = [start, toMm(e)]; drawAnnotations(); } });
  c.addEventListener("pointerup", () => {
    if (S.caliper && start && Math.hypot(S.caliper[1][0] - start[0], S.caliper[1][1] - start[1]) < 0.3) S.caliper = null;
    start = null; drawAnnotations();
  });
}

// ================================================================ keyboard
window.addEventListener("keydown", (e) => {
  if (e.key === "Escape") { S.caliper = null; $("help").hidden = true; drawAnnotations(); return; }
  const el = document.activeElement;
  if (!S.info || el.tagName === "SELECT" || (el.tagName === "INPUT" && el.type !== "range" && el.type !== "checkbox")) return;
  const k = e.key.toLowerCase(), step = e.shiftKey ? 2.5 : 0.5, ang = e.shiftKey ? 5 : 1;
  const r = (S.pose.rotation * Math.PI) / 180, lat = [Math.cos(r), Math.sin(r)], elv = [-Math.sin(r), Math.cos(r)];
  const move = (d, s) => { S.pose.u += d[0] * s; S.pose.v += d[1] * s; };
  const clamp = (x) => Math.max(-30, Math.min(30, x));
  const acts = {
    w: () => move(elv, step), s: () => move(elv, -step), d: () => move(lat, step), a: () => move(lat, -step),
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

// ================================================================ init
["showSkin", "showGland", "showLesions", "showPlane", "showGuides", "skinOpacity"].forEach((id) => $(id).addEventListener("input", applyVisibility));
$("showGT").addEventListener("input", () => {
  $("gtView").hidden = !$("showGT").checked;
  document.querySelector(".views").classList.toggle("single", !$("showGT").checked);
  setTimeout(drawAnnotations, 0);
});
$("showAnnot").addEventListener("input", drawAnnotations);
$("recenter").onclick = () => recenter(false);
$("overlay").onchange = (e) => { S.params.overlay = e.target.value; request(); };
$("phantom").onchange = (e) => load(e.target.value);
$("side").onchange = (e) => { S.side = e.target.value; if (S.info) buildGuides(); applyVisibility(); drawAnnotations(); request(); };
$("preset").onchange = (e) => applyPreset(e.target.value);
$("helpBtn").onclick = () => ($("help").hidden = false);
$("helpClose").onclick = () => ($("help").hidden = true);
$("save").onclick = async () => {
  const res = await fetch("/api/scene", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ phantom: S.phantom, side: S.side, pose: S.pose, params: S.params, clock: clockInfo() }),
  });
  status(`saved ${(await res.json()).path}`);
};
new ResizeObserver(() => drawAnnotations()).observe($("bmode"));

(async () => {
  S.config = await (await fetch("/api/config")).json();
  S.params = { ...S.config.params };
  $("preset").innerHTML = Object.entries(S.config.presets).map(([k, p]) => `<option value="${k}" title="${p.notes}">${p.label}</option>`).join("");
  $("preset").value = S.params.preset;
  const list = await refreshPhantomList();
  buildControls();
  initScene();
  bindCaliper();
  S.phantom = list.find((p) => p.cached)?.id || list[0]?.id;
  $("phantom").value = S.phantom;
  connect();
})();
