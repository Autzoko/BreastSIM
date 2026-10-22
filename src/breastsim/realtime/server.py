"""Real-time preview server (FastAPI + WebSocket).

Binds to localhost only; reach it from another machine through an SSH
tunnel (see src/scripts/realtime_remote.sh).

    python -m breastsim.realtime.server --port 8765
"""
import argparse
import asyncio
import dataclasses
import io
import json
import threading
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from .. import paths, phantom
from ..probe import PROBES, ProbePose
from .renderer import ImagingParams, PreviewRenderer

STATIC = Path(__file__).parent / "static"
SCENES = paths.OUTPUTS / "scenes"

app = FastAPI(title="BreastSIM real-time preview")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

renderer = None
lock = threading.Lock()   # one GPU renderer shared by all clients


def _png(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _jpeg(arr, quality=88):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
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


def _load(pid, send_status):
    vol = phantom.load(pid, log=send_status)
    with lock:
        renderer.set_volume(vol)
    iu, iv = np.nonzero(vol.lesion_proj)
    if len(iu) == 0:
        iu, iv = np.nonzero(np.isfinite(vol.surface))
    return {
        "id": vol.id, "description": vol.meta.get("description", ""), "voxel_mm": vol.voxel_mm,
        "shape": list(vol.label.shape), "extent_mm": list(vol.extent_mm),
        "has_lesion": bool(vol.lesion_proj.any()),
        "start": {"u": float(iu.mean()) * vol.voxel_mm, "v": float(iv.mean()) * vol.voxel_mm},
    }


# ------------------------------------------------------------------- routes
@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/phantoms")
def phantoms():
    return [{"id": p.id, "description": p.description,
             "cached": (phantom.cache_dir(p.id) / "meta.json").exists()} for p in phantom.list_phantoms()]


@app.get("/api/probes")
def probes():
    return {k: v.to_dict() for k, v in PROBES.items()}


@app.get("/api/params")
def default_params():
    return dataclasses.asdict(ImagingParams())


@app.get("/api/overview")
def overview():
    if renderer.vol is None:
        return JSONResponse({"error": "no phantom loaded"}, 404)
    return Response(_png(_overview(renderer.vol)), media_type="image/png")


@app.post("/api/scene")
async def save_scene(scene: dict):
    """Persist the current view for the data-generation side."""
    SCENES.mkdir(parents=True, exist_ok=True)
    scene = {**scene, "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
             "probe_spec": PROBES[scene.get("params", {}).get("probe", "L11-5v")].to_dict()}
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
                info = await asyncio.to_thread(_load, msg["phantom"], status)
                await sock.send_text(json.dumps({"type": "loaded", "phantom": info}))
            elif msg["type"] == "render":
                if renderer.vol is None:
                    continue
                pose = ProbePose(**msg["pose"])
                fields = {f.name for f in dataclasses.fields(ImagingParams)}
                params = ImagingParams(**{k: v for k, v in msg.get("params", {}).items() if k in fields})

                def run():
                    with lock:
                        return renderer.render(pose, params)
                img, meta = await asyncio.to_thread(run)
                t0 = time.perf_counter()
                data = _jpeg(img)
                meta["encode_ms"] = round((time.perf_counter() - t0) * 1e3, 1)
                meta["phantom"] = renderer.vol.id
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
