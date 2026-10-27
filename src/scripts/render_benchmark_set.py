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
import fnmatch
import json
from pathlib import Path

import numpy as np
from PIL import Image

from breastsim import devices, phantom, tissue
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
    ap.add_argument("--lesion-views", type=int, default=1, help="frames per lesion")
    ap.add_argument("--on-breast", type=float, default=0.0, metavar="MM",
                    help="normal views only where breast tissue (not just chest-wall muscle) extends at least MM "
                         "around the probe centre, e.g. 19 = half the L12-hf aperture (probe kept on the breast)")
    ap.add_argument("--phantoms", help="comma-separated phantom id patterns (fnmatch), e.g. 'victre/hetero_*'; "
                                       "default: the lesion phantoms (oat64 *l, USCT A-D)")
    ap.add_argument("--devices", action="store_true",
                    help="virtual device per frame (breastsim.devices; own random stream, poses unchanged)")
    args = ap.parse_args()
    out = Path(args.out)
    for sub in ("img", "lesion", "label", "raw"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    p = ImagingParams.from_preset(args.preset, overlay="none", **json.loads(args.set))
    if json.loads(args.probe_set):
        PROBES[p.probe] = dataclasses.replace(PROBES[p.probe], **json.loads(args.probe_set))
    dev_rng = np.random.default_rng(args.seed + 1)
    r = PreviewRenderer()
    if args.phantoms:
        pats = args.phantoms.split(",")
        pids = [i.id for i in phantom.list_phantoms() if any(fnmatch.fnmatch(i.id, q) for q in pats)]
    else:
        pids = [i.id for i in phantom.list_phantoms() if (i.source == "oat64" and i.id.endswith("l")) or i.source == "usct"]
    pids = pids[::args.every]
    n = 0
    with open(out / "meta.jsonl", "w") as meta_f:
        for pid in pids:
            vol = phantom.load(pid, log=lambda *a: None)
            r.set_volume(vol)
            ok = np.isfinite(vol.surface)
            if args.on_breast > 0:
                from scipy import ndimage
                breast = ((vol.label > 0) & ~np.isin(vol.label, phantom.CHEST_WALL_LABELS)).any(0)
                r_px = int(round(args.on_breast / vol.voxel_mm))
                yy, xx = np.mgrid[-r_px:r_px + 1, -r_px:r_px + 1]
                ok &= ndimage.binary_erosion(breast, structure=yy ** 2 + xx ** 2 <= r_px ** 2)
            iu, iv = np.nonzero(ok)
            # lesion views (detached sub-2 mm mass fragments are not targeted)
            views = [("lesion", np.array(l["center_mm"][1:]) + rng.normal(0, 3, 2), l) for l in vol.meta.get("lesions", [])
                     if l["diameter_mm"] >= 2 for _ in range(args.lesion_views)]
            for _ in range(args.normal_per_phantom):
                j = rng.integers(len(iu))
                views.append(("normal", np.array([iu[j], iv[j]]) * vol.voxel_mm, None))
            for view, (u, v), les_t in views:
                pose = ProbePose(u=float(u), v=float(v), rotation=float(rng.uniform(-180, 180)))
                dev = devices.sample_device(dev_rng) if args.devices else None
                pp = ImagingParams.from_preset(args.preset, overlay="none", **{**json.loads(args.set), **dev["params"]}) \
                    if dev else p
                img, meta, lab = r.render(pose, pp)
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
                                             density=pid.split("/")[1][0] if pid.startswith("oat64") else
                                             tissue.VICTRE_DENSITY_BIRADS.get(pid.split("/")[1].split("_")[0], "?")
                                             if pid.startswith("victre") else pid[-1],
                                             **({"device": dev} if dev else {}),
                                             **({f"lesion_{k}": les_t["recipe"][k] for k in ("case", "echo", "margin", "key")}
                                                if les_t and "recipe" in les_t else {}))) + "\n")
                n += 1
        print(f"{n} frames -> {out}")


if __name__ == "__main__":
    main()
