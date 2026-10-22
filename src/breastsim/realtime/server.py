"""Real-time preview server (FastAPI + WebSocket).

Binds to localhost only; reach it from another machine through an SSH
tunnel (see src/scripts/realtime_remote.sh).

    python -m breastsim.realtime.server --port 8765
"""
import argparse
import asyncio
import base64
import dataclasses
import io
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from .. import paths, phantom, tissue
from ..presets import DEFAULT_PRESET, PRESETS
from ..probe import PROBES, ProbePose
from .renderer import ImagingParams, PreviewRenderer

STATIC = Path(__file__).parent / "static"
SCENES = paths.OUTPUTS / "scenes"
PARAM_FIELDS = {f.name for f in dataclasses.fields(ImagingParams)}
POSE_FIELDS = {f.name for f in dataclasses.fields(ProbePose)}

app = FastAPI(title="BreastSIM real-time preview")
app.add_middleware(GZipMiddleware, minimum_size=2048)
app.mount("/static", StaticFiles(directory=STATIC), name="static")

renderer = None
lock = threading.Lock()   # one GPU renderer shared by all clients


def _encode(arr, fmt="JPEG"):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt, **({"quality": 88} if fmt == "JPEG" else {}))
    return buf.getvalue()


def _overview(vol):
    """Coronal top view: skin height shading, lesions in red."""
    s = vol.surface
    ok = np.isfinite(s)
    g = np.zeros(s.shape, np.float32)
    if ok.any():
        lo, hi = np.nanpercentile(s, [2, 98])
        g[ok] = 0.25 + 0.75 * np.clip((s[ok] - lo) / max(hi - lo, 1e-3), 0, 1)
    rgb = np.repeat((g * 200)[..., None], 3, -1)
    rgb[vol.lesion_proj] = [235, 50, 50]
    return rgb.astype(np.uint8)


def _b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def _nipple(vol):
    """Nipple position (a, u, v) mm: the nipple label if present, else the skin apex."""
    if tissue.NIPPLE in vol.meta.get("labels", {}) or str(tissue.NIPPLE) in vol.meta.get("labels", {}):
        idx = np.argwhere(vol.label == tissue.NIPPLE)
        if len(idx):
            return [float(c) for c in idx.mean(0) * vol.voxel_mm]
    iu, iv = np.unravel_index(np.nanargmax(vol.surface), vol.surface.shape)
    return [float(vol.surface[iu, iv]), float(iu * vol.voxel_mm), float(iv * vol.voxel_mm)]


def _phantom_info(vol):
    lesions = [dict(l) for l in vol.meta.get("lesions", [])]
    for les in lesions:  # depth of the lesion center below the skin straight above it
        a, u, v = les["center_mm"]
        iu, iv = (int(np.clip(round(c / vol.voxel_mm), 0, n - 1)) for c, n in zip((u, v), vol.surface.shape))
        s = vol.surface[iu, iv]
        les["depth_below_skin_mm"] = round(float(s - a), 1) if np.isfinite(s) else None
    start = lesions[0]["center_mm"][1:] if lesions else [
        float(i.mean()) * vol.voxel_mm for i in np.nonzero(np.isfinite(vol.surface))]
    return {
        "id": vol.id, "description": vol.meta.get("description", ""), "voxel_mm": vol.voxel_mm,
        "shape": list(vol.label.shape), "extent_mm": list(vol.extent_mm),
        "lesions": lesions, "start": {"u": start[0], "v": start[1]},
        "surface_range_mm": [float(np.nanmin(vol.surface)), float(np.nanmax(vol.surface))],
        "nipple_mm": _nipple(vol),
        "a_range_mm": [vol.a0_mm, vol.a0_mm + vol.label.shape[0] * vol.voxel_mm],
        "chest_wall": vol.meta.get("chest_wall"),
    }


class BuildJob:
    """One cache build per phantom, run in a subprocess (h5py holds the GIL for
    long reads, which would stall the event loop and drop WebSockets). Every
    client waiting for the same phantom receives its progress lines."""

    jobs = {}

    def __init__(self, pid):
        self.pid, self.listeners, self.started = pid, set(), time.time()
        self.task = asyncio.create_task(self._run())

    async def _run(self):
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-u", "-m", "breastsim.phantom", self.pid,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, env=os.environ.copy())
        tail = []
        async for raw in proc.stdout:
            line = raw.decode(errors="replace").rstrip()
            tail = (tail + [line])[-20:]
            for cb in list(self.listeners):
                await cb(line, time.time() - self.started)
        if await proc.wait() != 0 or not phantom.is_cached(self.pid):
            raise RuntimeError("cache build failed: " + " | ".join(tail[-5:]))

    @classmethod
    async def ensure(cls, pid, listener):
        if phantom.is_cached(pid):
            return
        job = cls.jobs.get(pid)
        if job is None or job.task.done() and job.task.exception():
            job = cls.jobs[pid] = cls(pid)
        job.listeners.add(listener)
        try:
            await asyncio.shield(job.task)
        finally:
            job.listeners.discard(listener)


_recent = {}          # pid -> Volume, most recently used last (host RAM is plentiful)
RECENT_VOLUMES = 3


async def _load(pid, listener):
    if renderer.vol is not None and renderer.vol.id == pid:
        return _phantom_info(renderer.vol)
    vol = _recent.pop(pid, None)
    if vol is None:
        await BuildJob.ensure(pid, listener)
        await listener("reading preview volume", 0)
        vol = await asyncio.to_thread(phantom.load, pid)
    _recent[pid] = vol
    while len(_recent) > RECENT_VOLUMES:
        _recent.pop(next(iter(_recent)))

    def swap():
        with lock:
            renderer.set_volume(vol)
    await asyncio.to_thread(swap)
    return _phantom_info(vol)


# ------------------------------------------------------------------- routes
@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/phantoms")
def phantoms():
    return [{"id": p.id, "description": p.description, "cached": phantom.is_cached(p.id)}
            for p in phantom.list_phantoms()]


@app.get("/api/config")
def config():
    return {
        "probes": {k: v.to_dict() for k, v in PROBES.items()},
        "presets": {k: v.to_dict() for k, v in PRESETS.items()},
        "default_preset": DEFAULT_PRESET,
        "params": dataclasses.asdict(ImagingParams.from_preset(DEFAULT_PRESET)),
        "tissue_names": {int(k): v for k, v in tissue.NAMES.items()},
        "tissue_colors": {int(k): list(v) for k, v in tissue.COLORS.items()},
        "lesion_labels": list(tissue.LESIONS),
    }


@app.get("/api/overview")
def overview():
    if renderer.vol is None:
        return JSONResponse({"error": "no phantom loaded"}, 404)
    return Response(_encode(_overview(renderer.vol), "PNG"), media_type="image/png")


@app.get("/api/meshes")
async def meshes():
    """Surfaces of the loaded phantom for the 3D view, in (a, u, v) mm (built once, then cached)."""
    vol = renderer.vol
    if vol is None:
        return JSONResponse({"error": "no phantom loaded"}, 404)
    out = {}
    for name, (verts, faces) in (await asyncio.to_thread(phantom.display_meshes, vol)).items():
        out[name] = {"verts": _b64(verts.astype(np.float32)), "faces": _b64(faces.astype(np.uint32)),
                     "n_verts": int(len(verts)), "n_faces": int(len(faces))}
    return out


@app.post("/api/scene")
async def save_scene(scene: dict):
    """Persist the current view for the data-generation side."""
    SCENES.mkdir(parents=True, exist_ok=True)
    params = scene.get("params", {})
    scene = {**scene, "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
             "probe_spec": PROBES[params.get("probe", "L12-hf")].to_dict(),
             "preset_spec": PRESETS[params["preset"]].to_dict() if params.get("preset") in PRESETS else None}
    path = SCENES / f"{time.strftime('%Y%m%d_%H%M%S')}_{scene.get('phantom', 'x').replace('/', '_')}.json"
    path.write_text(json.dumps(scene, indent=1))
    return {"path": str(path.relative_to(paths.ROOT))}


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()

    async def send(obj):  # the client may have gone away while we were busy
        try:
            await sock.send_text(json.dumps(obj))
        except Exception:
            pass

    async def progress(line, elapsed):
        await send({"type": "loading", "msg": line.split(": ", 1)[-1], "elapsed": round(elapsed)})

    async def do_load(pid):
        await send({"type": "loading", "phantom": pid, "msg": "opening", "elapsed": 0,
                    "cached": phantom.is_cached(pid)})
        try:
            info = await _load(pid, progress)
            await send({"type": "loaded", "phantom": info})
        except Exception as e:  # report instead of dropping the connection
            await send({"type": "error", "msg": f"load failed: {e}"})

    load_task = None
    try:
        while True:
            msg = json.loads(await sock.receive_text())
            if msg["type"] == "load":
                if load_task and not load_task.done():
                    load_task.cancel()          # newer choice wins; the build itself keeps running
                load_task = asyncio.create_task(do_load(msg["phantom"]))
            elif msg["type"] == "render":
                if renderer.vol is None:
                    await sock.send_text(json.dumps({"type": "error", "msg": "no phantom loaded"}))
                    continue
                pose = ProbePose(**{k: float(v) for k, v in msg["pose"].items() if k in POSE_FIELDS})
                params = ImagingParams(**{k: v for k, v in msg.get("params", {}).items() if k in PARAM_FIELDS})

                def run():
                    with lock:
                        return renderer.render(pose, params)
                try:
                    img, meta, labels = await asyncio.to_thread(run)
                except Exception as e:
                    await sock.send_text(json.dumps({"type": "error", "msg": f"render failed: {e!r}"}))
                    continue
                t0 = time.perf_counter()
                data = _encode(img)
                lab_png = _encode(labels, "PNG")    # ground-truth tissue codes on the image grid
                meta.update(encode_ms=round((time.perf_counter() - t0) * 1e3, 1), phantom=renderer.vol.id,
                            seq=msg.get("seq"))
                await sock.send_text(json.dumps({"type": "frame", "meta": meta, "binaries": ["bmode", "labels"]}))
                await sock.send_bytes(data)
                await sock.send_bytes(lab_png)
    except (WebSocketDisconnect, RuntimeError):  # RuntimeError: send after the client closed
        pass


def main():
    global renderer
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    renderer = PreviewRenderer(args.device)
    print(f"[realtime] renderer on {renderer.device}; serving http://{args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
