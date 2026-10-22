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
from breastsim.probe import PROBES, ProbePose
from breastsim.realtime.renderer import ImagingParams, PreviewRenderer


def skin_normal_pose(vol, target, spec, rots=12):
    """Probe on the skin point nearest to `target` (a, u, v) mm, beam pointing at the target,
    rotated about the beam so the skin under the array is as flat as possible."""
    skin = np.argwhere(np.isin(vol.label, (tissue.SKIN, tissue.EPIDERMIS)))
    pts = skin * vol.voxel_mm
    s = pts[np.argmin(((pts - target) ** 2).sum(1))]
    d = (target - s) / np.linalg.norm(target - s)
    ref = np.array([0.0, 1.0, 0.0]) if abs(d[1]) < 0.9 else np.array([0.0, 0.0, 1.0])
    e1 = np.cross(d, ref); e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    origin = s - 0.5 * d                                      # face just outside the skin
    best = None
    half = spec.aperture * 1e3 / 2
    for th in np.linspace(0, np.pi, rots, endpoint=False):
        lat = np.cos(th) * e1 + np.sin(th) * e2
        el = np.cross(lat, d)
        # distance from the face plane to the skin under the array (gel gap)
        xs = np.linspace(-half, half, 25)
        gaps = []
        for x in xs:
            p = origin + x * lat
            dist = np.linalg.norm(pts - p, axis=1)
            near = pts[dist < 3]
            gaps.append(0.0 if len(near) == 0 else float(((near - p) @ d).min()))
        g = max(gaps)
        if best is None or g < best[0]:
            best = (g, lat, el)
    g, lat, el = best
    pose = ProbePose(u=float(s[1]), v=float(s[2]), origin_mm=tuple(origin), axes_rows=(tuple(lat), tuple(d), tuple(el)))
    return pose, g, float(np.linalg.norm(target - s))


def choose_poses(n, rng):
    ids = [p.id for p in phantom.list_phantoms() if p.id.startswith("oat64/") and p.id.endswith("l")
           and phantom.is_cached(p.id)]
    spec = PROBES["L12-hf"]
    poses = []
    n_les = n - n // 3
    for pid in ids:
        if len(poses) >= n_les:
            break
        vol = phantom.load(pid, chest_wall=False)
        for les in vol.meta.get("lesions", [])[:2]:
            pose, gap, depth = skin_normal_pose(vol, np.array(les["center_mm"]), spec)
            if depth > 40:
                continue
            poses.append(dict(pid=pid, pose=pose, gap=gap, kind=f"lesion #{les['id']} {les['type']}",
                              depth=float(np.clip(depth + 12, 25, 45)), focus=float(depth)))
    for pid in ids:
        if len(poses) >= n:
            break
        vol = phantom.load(pid, chest_wall=False)
        tissue_vox = np.argwhere(np.isin(vol.label, (tissue.GLAND,)))
        target = tissue_vox[rng.integers(len(tissue_vox))] * vol.voxel_mm
        pose, gap, depth = skin_normal_pose(vol, target, spec)
        d = float(np.clip(depth + 12, 25, 40))
        poses.append(dict(pid=pid, pose=pose, gap=gap, kind="glandular / normal tissue", depth=d, focus=d / 2))
    return poses[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", default="outputs/kwave_pilot")
    ap.add_argument("--strength", type=float, default=0.02, help="scatterer perturbation for echogenicity 1")
    ap.add_argument("--roughness", type=float, default=0.5, help="sub-voxel boundary jitter, source voxels")
    ap.add_argument("--only", default="", help="comma-separated image numbers to render (1-based)")
    ap.add_argument("--tx", choices=("focused", "plane"), default="focused")
    ap.add_argument("--angles", default="-8,-4,0,4,8", help="plane-wave angles (deg) for --tx plane")
    ap.add_argument("--no-conform", action="store_true", help="flat probe face over the curved skin (water wedges)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    poses = choose_poses(args.n, np.random.default_rng(0))
    print(json.dumps([{k: v for k, v in p.items() if k != "pose"} for p in poses], indent=1), flush=True)
    r = PreviewRenderer()
    rows, report = [], []
    only = {int(i) for i in args.only.split(",") if i}
    for k, ps in enumerate(poses):
        if only and k + 1 not in only:
            continue
        t0 = time.time()
        vol = phantom.load(ps["pid"], chest_wall=False)
        for les in vol.meta.get("lesions", []):               # phantom's own lesion attenuation
            les.pop("attenuation_db_cm_mhz", None)
        pose = ps["pose"]
        print(f"[{k + 1}/{len(poses)}] {ps['pid']} {ps['kind']} depth {ps['depth']:.0f} mm", flush=True)
        kw, lab, info = render_pose(ps["pid"], vol, pose, depth_mm=ps["depth"], scatter_strength=args.strength,
                                   roughness=args.roughness, tx=args.tx, focus_mm=ps["focus"],
                                   angles_deg=[float(a) for a in args.angles.split(",")],
                                   conform=not args.no_conform)
        r.set_volume(vol)
        prev, _, _ = r.render(pose, ImagingParams(probe="L12-hf", depth_mm=ps["depth"], px_x_mm=0.1, px_z_mm=0.1,
                                                  conform=not args.no_conform, overlay="none", speckle_reduction=0.0))
        prev = np.array(Image.fromarray(prev).resize((kw.shape[1], kw.shape[0])))
        gt = tissue.color_lut()[lab]
        sep = np.full((kw.shape[0], 6, 3), 255, np.uint8)
        panel = np.concatenate([np.repeat(kw[..., None], 3, -1), sep, prev, sep, gt], 1)
        im = Image.fromarray(panel)
        ImageDraw.Draw(im).text((6, 6), f"{k + 1}. {ps['pid']}  {ps['kind']}  k-Wave | preview | labels", fill=(255, 255, 0))
        im.save(out / f"{k + 1:02d}.png")
        Image.fromarray(kw).save(out / f"{k + 1:02d}_kwave.png")
        info.update({k: v for k, v in ps.items() if k != "pose"}, total_s=round(time.time() - t0, 1),
                    origin_mm=list(pose.origin_mm), axes=[list(r) for r in pose.axes_rows])
        report.append(info)
        print(f"   done in {info['total_s']} s", flush=True)
        (out / "report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
