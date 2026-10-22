#!/usr/bin/env python
"""k-Wave pilot: N full-wave B-mode images from IDB-8164905 phantoms, each next to the
real-time preview of the same pose and the ground-truth labels.

    python src/scripts/kwave_pilot.py --n 10 --out outputs/kwave_pilot
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from breastsim import phantom, tissue
from breastsim.datagen.kwave_sim import render_pose
from breastsim.probe import PROBES, ProbePose, _lookup
from breastsim.realtime.renderer import ImagingParams, PreviewRenderer


def flattest_rotation(vol, u, v, spec):
    """Rotation whose footprint has the smallest skin-height spread (least gel under the probe)."""
    best = None
    for rot in (0, 30, 60, 90, 120, 150):
        pose = ProbePose(u=u, v=v, rotation=rot)
        lat = pose.axes()[0]
        s = np.linspace(-1, 1, 41) * spec.aperture * 1e3 / 2
        h = _lookup(vol.surface, u + s * lat[1], v + s * lat[2], vol.voxel_mm)
        if not np.isfinite(h).all():
            continue
        spread = float(h.max() - h.min())
        if best is None or spread < best[0]:
            best = (spread, rot)
    return best


def choose_poses(n, rng):
    ids = [p.id for p in phantom.list_phantoms() if p.id.startswith("oat64/") and p.id.endswith("l")
           and phantom.is_cached(p.id)]
    poses = []
    spec = PROBES["L12-hf"]
    for pid in ids:
        vol = phantom.load(pid, chest_wall=False)
        for les in vol.meta.get("lesions", []):
            a, u, v = les["center_mm"]
            fr = flattest_rotation(vol, u, v, spec)
            if fr is None:
                continue
            depth = float(np.nanmax(vol.surface) - a)
            poses.append(dict(pid=pid, u=u, v=v, rot=fr[1], gap=fr[0], kind=f"lesion #{les['id']} {les['type']}",
                              depth=float(np.clip(depth + 10, 30, 45))))
    poses = sorted(poses, key=lambda p: p["gap"])[: max(1, n - n // 3)]
    # normal-tissue views near the dome of a few phantoms
    for pid in ids[: n - len(poses)]:
        vol = phantom.load(pid, chest_wall=False)
        iu, iv = np.unravel_index(np.nanargmax(vol.surface), vol.surface.shape)
        u, v = iu * vol.voxel_mm + rng.uniform(-8, 8), iv * vol.voxel_mm + rng.uniform(-8, 8)
        fr = flattest_rotation(vol, u, v, spec) or (0, 0)
        poses.append(dict(pid=pid, u=u, v=v, rot=fr[1], gap=fr[0], kind="normal tissue", depth=35.0))
    return poses[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", default="outputs/kwave_pilot")
    ap.add_argument("--strength", type=float, default=0.02, help="scatterer perturbation for echogenicity 1")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    poses = choose_poses(args.n, np.random.default_rng(0))
    print(json.dumps(poses, indent=1), flush=True)
    r = PreviewRenderer()
    rows, report = [], []
    for k, ps in enumerate(poses):
        t0 = time.time()
        vol = phantom.load(ps["pid"], chest_wall=False)
        for les in vol.meta.get("lesions", []):               # phantom's own lesion attenuation
            les.pop("attenuation_db_cm_mhz", None)
        pose = ProbePose(u=ps["u"], v=ps["v"], rotation=ps["rot"])
        print(f"[{k + 1}/{len(poses)}] {ps['pid']} {ps['kind']} depth {ps['depth']:.0f} mm", flush=True)
        kw, lab, info = render_pose(ps["pid"], vol, pose, depth_mm=ps["depth"], scatter_strength=args.strength)
        r.set_volume(vol)
        prev, _, _ = r.render(pose, ImagingParams(probe="L12-hf", depth_mm=ps["depth"], px_x_mm=0.1, px_z_mm=0.1,
                                                  conform=False, overlay="none", speckle_reduction=0.0))
        prev = np.array(Image.fromarray(prev).resize((kw.shape[1], kw.shape[0])))
        gt = tissue.color_lut()[lab]
        sep = np.full((kw.shape[0], 6, 3), 255, np.uint8)
        panel = np.concatenate([np.repeat(kw[..., None], 3, -1), sep, prev, sep, gt], 1)
        im = Image.fromarray(panel)
        ImageDraw.Draw(im).text((6, 6), f"{k + 1}. {ps['pid']}  {ps['kind']}  k-Wave | preview | labels", fill=(255, 255, 0))
        im.save(out / f"{k + 1:02d}.png")
        Image.fromarray(kw).save(out / f"{k + 1:02d}_kwave.png")
        info.update(ps, total_s=round(time.time() - t0, 1))
        report.append(info)
        print(f"   done in {info['total_s']} s", flush=True)
        (out / "report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
