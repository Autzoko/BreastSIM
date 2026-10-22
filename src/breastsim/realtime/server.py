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
    }


def _load(pid, send_status):
    vol = phantom.load(pid, log=send_status)
    with lock:
        renderer.set_volume(vol)
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
    }


@app.get("/api/overview")
def overview():
    if renderer.vol is None:
        return JSONResponse({"error": "no phantom loaded"}, 404)
    return Response(_encode(_overview(renderer.vol), "PNG"), media_type="image/png")


@app.get("/api/meshes")
def meshes():
    """Surfaces of the loaded phantom for the 3D view, in (a, u, v) mm."""
    if renderer.vol is None:
        return JSONResponse({"error": "no phantom loaded"}, 404)
    out = {}
    for name, (verts, faces) in phantom.load_meshes(renderer.vol.id).items():
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
    loop = asyncio.get_running_loop()

    def status(msg):
        asyncio.run_coroutine_threadsafe(sock.send_text(json.dumps({"type": "status", "msg": str(msg)})), loop)

    try:
        while True:
            msg = json.loads(await sock.receive_text())
            if msg["type"] == "load":
                await sock.send_text(json.dumps({"type": "status", "msg": f"loading {msg['phantom']} ..."}))
                try:
                    info = await asyncio.to_thread(_load, msg["phantom"], status)
                    await sock.send_text(json.dumps({"type": "loaded", "phantom": info}))
                except Exception as e:  # report instead of dropping the connection
                    await sock.send_text(json.dumps({"type": "error", "msg": f"load failed: {e!r}"}))
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
                    img, meta, _ = await asyncio.to_thread(run)
                except Exception as e:
                    await sock.send_text(json.dumps({"type": "error", "msg": f"render failed: {e!r}"}))
                    continue
                t0 = time.perf_counter()
                data = _encode(img)
                meta.update(encode_ms=round((time.perf_counter() - t0) * 1e3, 1), phantom=renderer.vol.id,
                            seq=msg.get("seq"))
                await sock.send_text(json.dumps({"type": "frame", "meta": meta}))
                await sock.send_bytes(data)
    except WebSocketDisconnect:
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
