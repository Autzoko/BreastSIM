import base64
import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from breastsim import phantom
from breastsim.realtime import server
from breastsim.realtime.renderer import PreviewRenderer


@pytest.fixture()
def client(synthetic_volume, monkeypatch, tmp_path):
    meshes = phantom.build_meshes(synthetic_volume.label, synthetic_volume.voxel_mm, synthetic_volume.meta["lesions"])
    monkeypatch.setattr(server.phantom, "load", lambda pid, log=print: synthetic_volume)
    monkeypatch.setattr(server.phantom, "load_meshes", lambda pid: meshes)
    monkeypatch.setattr(server, "SCENES", tmp_path / "scenes")
    monkeypatch.setattr(server.paths, "ROOT", tmp_path)
    server.renderer = PreviewRenderer("cpu")
    return TestClient(server.app)


def test_config_and_index(client):
    cfg = client.get("/api/config").json()
    assert cfg["default_preset"] in cfg["presets"]
    assert cfg["params"]["probe"] in cfg["probes"]
    assert all(p["probe"] in cfg["probes"] for p in cfg["presets"].values())
    assert client.get("/").status_code == 200
    for f in ("app.js", "style.css"):
        assert client.get(f"/static/{f}").status_code == 200


def test_ws_load_render_and_meshes(client):
    cfg = client.get("/api/config").json()
    params = {**cfg["params"], "elevation_samples": 4}
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "load", "phantom": "synthetic/dome"}))
        while True:
            msg = json.loads(ws.receive_text())
            if msg["type"] == "loaded":
                info = msg["phantom"]
                break
            assert msg["type"] == "status"
        assert len(info["lesions"]) == 1
        assert info["lesions"][0]["depth_below_skin_mm"] == pytest.approx(15, abs=0.6)
        pose = {"u": info["start"]["u"], "v": info["start"]["v"], "rotation": 0, "tilt": 0, "rock": 0, "standoff": 0}
        ws.send_text(json.dumps({"type": "render", "seq": 1, "pose": pose, "params": params}))
        meta = json.loads(ws.receive_text())
        assert meta["type"] == "frame" and meta["meta"]["seq"] == 1 and meta["meta"]["lesion_area_mm2"] > 0
        img = Image.open(io.BytesIO(ws.receive_bytes()))
        assert img.format == "JPEG" and img.size[0] > 100
        # unknown fields are ignored, bad phantom ids are reported, not fatal
        ws.send_text(json.dumps({"type": "render", "seq": 2, "pose": {**pose, "junk": 1}, "params": {**params, "junk": 1}}))
        assert json.loads(ws.receive_text())["meta"]["seq"] == 2
        ws.receive_bytes()
    m = client.get("/api/meshes").json()
    v = np.frombuffer(base64.b64decode(m["lesion0"]["verts"]), np.float32)
    assert len(v) == 3 * m["lesion0"]["n_verts"]
    assert client.get("/api/overview").headers["content-type"] == "image/png"


def test_save_scene(client):
    r = client.post("/api/scene", json={"phantom": "synthetic/dome", "pose": {"u": 1, "v": 2},
                                        "params": {"preset": "busi", "probe": "ML6-15"}}).json()
    assert r["path"].startswith("scenes/")
