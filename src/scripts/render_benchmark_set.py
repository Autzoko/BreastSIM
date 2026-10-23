#!/usr/bin/env python
"""Render the standard simulated set for the realism benchmark (breastsim.realism).

    python src/scripts/render_benchmark_set.py --out outputs/realism/sim/v5 --preset breast
    python src/scripts/render_benchmark_set.py --out outputs/realism/sim/x --set '{"compound_deg": []}'

One frame through every lesion of the lesion phantoms (oat64 *l, USCT A-D) at a random probe
rotation and lateral offset, plus normal-tissue frames at random positions of the same phantoms.
Writes img/<n>.png (gray B-mode), lesion/<n>.png (mask), label/<n>.png (tissue labels),
raw/<n>.npy (linear echo intensity before TGC, float32, relative to the unit-speckle reference;
display variants can be evaluated without re-rendering) and meta.jsonl (phantom, pose, px_mm,
view). Deterministic for a given --seed.
"""
import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
from PIL import Image

from breastsim import phantom, tissue
from breastsim.probe import PROBES, ProbePose
from breastsim.realtime.renderer import ImagingParams, PreviewRenderer


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--preset", default="breast")
    ap.add_argument("--set", default="{}", help="ImagingParams overrides (JSON)")
    ap.add_argument("--normal-per-phantom", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--every", type=int, default=1, help="use every k-th phantom (quick iterations)")
    ap.add_argument("--probe-set", default="{}", help="probe spec overrides for experiments (JSON), e.g. bandwidth")
    args = ap.parse_args()
    out = Path(args.out)
    for sub in ("img", "lesion", "label", "raw"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    p = ImagingParams.from_preset(args.preset, overlay="none", **json.loads(args.set))
    if json.loads(args.probe_set):
        PROBES[p.probe] = dataclasses.replace(PROBES[p.probe], **json.loads(args.probe_set))
    r = PreviewRenderer()
    pids = [i.id for i in phantom.list_phantoms() if (i.source == "oat64" and i.id.endswith("l")) or i.source == "usct"]
    pids = pids[::args.every]
    n = 0
    with open(out / "meta.jsonl", "w") as meta_f:
        for pid in pids:
            vol = phantom.load(pid, log=lambda *a: None)
            r.set_volume(vol)
            iu, iv = np.nonzero(np.isfinite(vol.surface))
            views = [("lesion", np.array(l["center_mm"][1:]) + rng.normal(0, 3, 2)) for l in vol.meta.get("lesions", [])]
            for _ in range(args.normal_per_phantom):
                j = rng.integers(len(iu))
                views.append(("normal", np.array([iu[j], iv[j]]) * vol.voxel_mm))
            for view, (u, v) in views:
                pose = ProbePose(u=float(u), v=float(v), rotation=float(rng.uniform(-180, 180)))
                img, meta, lab = r.render(pose, p)
                les = np.isin(lab, tissue.LESIONS)
                if view == "lesion" and not les.any():
                    continue
                name = f"{n:05d}.png"
                Image.fromarray(img[..., 0]).save(out / "img" / name)
                Image.fromarray((les * 255).astype(np.uint8)).save(out / "lesion" / name)
                Image.fromarray(lab).save(out / "label" / name)
                np.save(out / "raw" / name.replace(".png", ".npy"), (r.last_raw / r.last_ref).cpu().numpy().astype(np.float32))
                meta_f.write(json.dumps(dict(file=name, phantom=pid, view=view, u=float(u), v=float(v),
                                             rotation=pose.rotation, px_mm=meta["px_z_mm"],
                                             density=pid.split("/")[1][0] if pid.startswith("oat64") else pid[-1])) + "\n")
                n += 1
        print(f"{n} frames -> {out}")


if __name__ == "__main__":
    main()
