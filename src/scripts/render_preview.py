#!/usr/bin/env python
"""Render preview frames to PNG (no server), e.g. for checks and benchmarks.

    python src/scripts/render_preview.py usct/A --out outputs/preview_test.png
    python src/scripts/render_preview.py usct/A --sweep 5 --step 0.5   # elevation sweep
"""
import argparse
import time

import numpy as np
from PIL import Image

from breastsim import phantom
from breastsim.probe import ProbePose
from breastsim.realtime.renderer import ImagingParams, PreviewRenderer


def lesion_center(vol):
    iu, iv = np.nonzero(vol.lesion_proj)
    if len(iu) == 0:
        iu, iv = np.nonzero(np.isfinite(vol.surface))
    return float(iu.mean()) * vol.voxel_mm, float(iv.mean()) * vol.voxel_mm


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phantom")
    ap.add_argument("--out", default="outputs/preview_test.png")
    ap.add_argument("--rotation", type=float, default=0.0)
    ap.add_argument("--sweep", type=int, default=1, help="frames along elevation")
    ap.add_argument("--step", type=float, default=0.5, help="sweep step, mm")
    ap.add_argument("--overlay", default="contour")
    args = ap.parse_args()

    vol = phantom.load(args.phantom)
    r = PreviewRenderer()
    r.set_volume(vol)
    u, v = lesion_center(vol)
    params = ImagingParams(overlay=args.overlay)
    frames, times = [], []
    for k in range(args.sweep):
        pose = ProbePose(u=u, v=v + (k - args.sweep // 2) * args.step, rotation=args.rotation)
        img, meta = r.render(pose, params)
        frames.append(img)
        times.append(meta["render_ms"])
    for _ in range(5):  # warm timing
        t0 = time.perf_counter()
        r.render(pose, params)
        times.append((time.perf_counter() - t0) * 1e3)
    Image.fromarray(np.concatenate(frames, axis=1)).save(args.out)
    print(f"{args.phantom} pose u={u:.1f} v={v:.1f} mm | {meta} | warm render {np.median(times[-5:]):.1f} ms"
          f" on {r.device} -> {args.out}")


if __name__ == "__main__":
    main()
