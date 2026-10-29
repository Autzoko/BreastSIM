#!/usr/bin/env python3
"""Insert typed lesions (breastsim.lesions recipes from real BrEaST lesions) into a VICTRE phantom with the
unmodified pipeline. Runs inside the breastsim/victre container, with the repository's src mounted:

    docker run --rm -e OMP_NUM_THREADS=2 --cpus 2 -u $(id -u):$(id -g) -v $VARIANT:/out -v $PWD/src:/src \
        -e PYTHONPATH=/opt/VICTRE_PIPELINE:/src breastsim/victre python3 /src/scripts/victre_lesion_typed.py --out /out --seed 1 \
        --n 4 --type-seed 7

<out>/<seed>/p_<seed>.* is a copy of a base phantom (no pcl_* yet). For each of --n lesions a recipe is
drawn (lesions.sample_recipe("<type-seed>#<i>")), then with documented pipeline interfaces only:
- mass: the breastMass binary with its own full config (breastmass.py; the pipeline template ignores all keys but
  imgRes / complexity / seed / alpha) and the keys of the recipe's margin (MARGIN_KEYS); alpha is rescaled once so
  that the trimmed mass's eq. diameter is the case's in-plane eq. diameter x SECTION_FACTOR[margin];
- the mass is inserted from its bounding box (cube, +2 voxels; documented lesion_file argument): the breastMass ROI
  (~2.7x the diameter) does not fit into a 25-37 mm thick supine breast;
- location: a fibroglandular voxel where the mass grown by --margin mm touches no air / skin / nipple /
  muscle / lesion (as victre_insert_at.py), centre depth below the anterior skin within the recipe's depth
  (x --depth-scale) +-2 mm clipped to --depth, else within --depth; inserted with insert_lesions(locations=...);
- calcifications ("in a mass"): Pipeline.generate_cluster (cube of 0.6 x the mass eq. diameter, 3-12
  calcifications of 1-voxel radius; size in whole mm) inserted at the same location (material 250).
Writes pcl_<seed>.* and insertion.json ("lesions": recipe, morphology, alpha, centre in mm of the array).
"""
import argparse
import json
import subprocess
import time

import h5py
import numpy as np
from scipy import ndimage
from Victre import Constants, Pipeline

from breastsim import lesions
from breastmass import generate_mass
from victre_insert_at import ap_axis

# breastMass keys per BI-RADS margin (documented breastMass parameters, written to its own config: the pipeline's
# template ignores them, see breastmass.py). meanShape 0 = spike-, 1 = bump-like low-frequency modifications;
# meanLFLen / meanLFRad = their relative length / radius; meanInitial = spicule segments (0 = none).
# Central-section irregularity at alpha 5 mm (after a 1-voxel opening): smooth 0.03, default 0.10, bumps with
# LFLen 0.25 0.14, LFRad 0.35 0.13, spikes with LFLen 0.25 0.12 (2026-10-28).
MARGIN_KEYS = {
    "circumscribed": dict(meanInitial=0, stdInitial=0),                                   # near default, no spicules
    "lobulated": dict(meanShape=1.0, meanLFLen=0.25, meanLFRad=0.30, meanInitial=0, stdInitial=0),   # microlobulated
    "indistinct": dict(meanShape=1.0, meanLFLen=0.25, meanInitial=0, stdInitial=0),       # indistinct / angular
    "spiculated": dict(meanShape=0.0, meanLFLen=0.25),                                    # spikes + default spicules
}


# the real eq. diameter is the in-plane section of the annotated (largest) view of the visible lesion: rendered views
# (imaging-resolution masks) through a mass of 3D eq. diameter D give a section of ~0.9 D for compact masses and
# less for bumpy / lobulated ones (t6u, 138 lesion frames, at a factor 1.10: circumscribed 0.97, spiculated 0.95,
# indistinct 0.70, lobulated 0.36 with LFRad 0.35) -> per-margin factors on the target 3D eq. diameter
SECTION_FACTOR = {"circumscribed": 1.13, "spiculated": 1.16, "indistinct": 1.55, "lobulated": 1.55}


def margin_class(r):
    m = r["margin"]
    if "spiculated" in m:
        return "spiculated"
    if "microlobulated" in m:
        return "lobulated"
    return "circumscribed" if m == "circumscribed" else "indistinct"


def trim(body):
    """Mass core (opening with a ball of 0.2 x the radius removes spicules) + what lies within 0.5 x its radius:
    breastMass spicules / fragments can reach 20-30 mm, which no supine breast holds and the imaging mask drops
    anyway. Distance transforms on the bounding box (a morphological opening of a 500^3 spiculated mass cube with
    a ball took hours and 24 GB)."""
    sl = ndimage.find_objects(body.astype(np.uint8))[0]
    b = body[sl]
    r_all = (3 * b.sum() / (4 * np.pi)) ** (1 / 3)
    rb = max(1.0, 0.2 * r_all)
    eroded = ndimage.distance_transform_edt(b) > rb
    core = ndimage.distance_transform_edt(~eroded) <= rb if eroded.any() else b
    comp, n = ndimage.label(core)
    main = comp == 1 + int(np.argmax(ndimage.sum(comp > 0, comp, range(1, n + 1))))
    r_main = (3 * main.sum() / (4 * np.pi)) ** (1 / 3)
    out = np.zeros(body.shape, np.uint8)
    out[sl] = b & (ndimage.distance_transform_edt(~main) <= 0.5 * r_main)
    return out


def pick_location(p, mass, margin_mm, depth, rng, tries=2000):
    """Centre on a fibroglandular voxel where the mass grown by margin_mm touches no air / skin / nipple / muscle /
    lesion. Candidates are drawn among the glandular voxels whose depth below the anterior skin (along the
    chest-wall axis) lies in depth[0], then depth[1] (random order, up to `tries` overlap tests each), so shallow
    placements are found when they exist (drawing among all glandular voxels mostly hit deep ones: centre depth
    median 16 mm for real cases at ~10 mm). (The pipeline's own candidate list p_<seed>.loc does not map into the
    uncompressed phantom array.) Returns (location for insert_lesions, array index of the centre, depth mm)."""
    vox = float(p.mhd["ElementSpacing"][0])
    grown = ndimage.binary_dilation(mass, iterations=max(1, int(round(margin_mm / vox))))
    ph = p._load_phantom_array_from_gzip()
    axis, sign = ap_axis(ph)
    banned = np.append(Constants.FORBIDDEN_OVERLAP, list(Constants.LESION_MATERIALS.values()))
    s = mass.shape
    tis = ph > 0
    n = ph.shape[axis]
    first = np.argmax(np.flip(tis, axis) if sign > 0 else tis, axis=axis)       # skin index per column
    skin = (n - 1 - first) if sign > 0 else first
    gl = np.argwhere(ph == Constants.PHANTOM_MATERIALS["glandular"])
    col = tuple(np.delete(gl, axis, 1).T)
    dep = np.abs(skin[col] - gl[:, axis]) * vox
    for lo_, hi_ in depth:
        sel = np.nonzero((dep >= lo_) & (dep <= hi_))[0]
        for j in sel[rng.permutation(len(sel))[:tries]]:
            ctr = gl[j]
            lo = [int(ctr[i] - s[i] / 2) for i in range(3)]                # as insert_lesions slices
            roi = ph[lo[0]:lo[0] + s[0], lo[1]:lo[1] + s[1], lo[2]:lo[2] + s[2]]
            if min(lo) < 0 or roi.shape != grown.shape or np.isin(roi[grown], banned).any():
                continue
            idx = [int(c) for c in ctr]
            return [idx[0], idx[2], idx[1]], idx, float(dep[j])        # insert_lesions slices axes (c0, c2, c1)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/out")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--type-seed", type=int, default=0)
    ap.add_argument("--depth", type=float, nargs=2, default=(5.0, 22.0), help="allowed centre depth, mm")
    ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--depth-scale", type=float, default=1.0,
                    help="target depth = case depth x this (e.g. 1.25 before a 25 mm AP compression, which brings "
                         "lesions ~20 %% closer to the skin)")
    a = ap.parse_args()
    t0 = time.time()
    p = Pipeline(seed=a.seed, results_folder=a.out)
    vox = float(p.mhd["ElementSpacing"][0])
    rng = np.random.default_rng(a.type_seed)
    out = []
    for i in range(a.n):
        r = lesions.sample_recipe(f"{a.type_seed}#{i}")
        kind = margin_class(r)
        mass_seed = 1000 * a.type_seed + i + 1
        target = (r["eqdiam_mm"] or 10.0) * SECTION_FACTOR[kind]           # 3D eq. diameter of the trimmed mass
        alpha = float(np.clip(target / 2.16, 2.0, 10.0))
        vol = None
        for attempt in range(3):
            # breastMass occasionally builds a huge domain for one seed (spicules / LF modifications far out:
            # a 1381^3 or 1723^3 cube that fails to load, which aborted the whole phantom): retry with another seed
            try:
                for it in range(3):                                         # rescale alpha twice to hit the size
                    vol = trim(generate_mass(f"{a.out}/lesions/breastmass", mass_seed, alpha, vox, **MARGIN_KEYS[kind]))
                    eq = (6 * vol.sum() * vox ** 3 / np.pi) ** (1 / 3)
                    if it < 2:
                        alpha = float(np.clip(alpha * target / max(eq, 0.1), 1.5, 12.0))
                break
            except (ValueError, MemoryError, IndexError, subprocess.CalledProcessError) as e:
                print(json.dumps({"lesion": i, "mass_seed": mass_seed, "retry": repr(e)[:120]}), flush=True)
                vol, mass_seed = None, mass_seed + 500
                alpha = float(np.clip(target / 2.16, 2.0, 10.0))
        if vol is None:
            print(json.dumps({"lesion": i, "skipped": "breastMass failed 3 times"}))
            continue
        sl = ndimage.find_objects(vol)[0]
        side = max(q.stop - q.start for q in sl) + 4                   # cube: insert_lesions needs equal sides
        mass = np.zeros((side,) * 3, bool)
        sub = vol[sl] == 1
        o = [(side - n) // 2 for n in sub.shape]
        mass[o[0]:o[0] + sub.shape[0], o[1]:o[1] + sub.shape[1], o[2]:o[2] + sub.shape[2]] = sub
        crop = f"{a.out}/lesions/typed_{a.seed}_{mass_seed}.h5"
        with h5py.File(crop, "w") as f:
            f.create_dataset("volume", data=mass.astype(np.uint8), compression="gzip")
        d_case = (r["depth_mm"] or 12.0) * a.depth_scale
        want = (max(a.depth[0], d_case - 2), min(a.depth[1], d_case + 2))
        loc = pick_location(p, mass, a.margin, (want, tuple(a.depth)), rng)
        if loc is None:
            print(json.dumps({"lesion": i, "skipped": "no candidate passes the overlap test"}))
            continue
        cand, ctr, depth = loc
        p.insert_lesions(lesion_type=Constants.VICTRE_SPICULATED, n=1, lesion_file=crop, locations=[cand])
        eq = float((6 * mass.sum() * vox ** 3 / np.pi) ** (1 / 3))
        rec = {"recipe": r, "kind": kind, "breastMass": MARGIN_KEYS[kind], "alpha_mm": round(alpha, 2), "mass_seed": mass_seed,
               "eqdiam_mm": round(eq, 2), "centre_depth_mm": round(float(depth), 1), "location_vox": cand,
               "center_mm_zyx": [round(x * vox, 2) for x in ctr], "calcs": 0}
        if r["calc"]:
            n = p.generate_cluster(seed=mass_seed, size=int(round(max(2.0, 0.6 * eq))), nmin=3, nmax=12, smin=vox, smax=vox)
            p.insert_lesions(lesion_type=Constants.VICTRE_CLUSTERCALC, n=1, locations=[cand])
            rec["calcs"] = int(n)
        out.append(rec)
        print(json.dumps({k: rec[k] for k in ("kind", "alpha_mm", "eqdiam_mm", "centre_depth_mm", "calcs")}
                         | {"case": r["case"], "echo": r["echo"], "margin": r["margin"]}), flush=True)
    info = {"kind": "typed", "n": len(out), "alpha_mm": ",".join(str(o["alpha_mm"]) for o in out),
            "type_seed": a.type_seed, "lesions": out, "seconds": round(time.time() - t0, 1)}
    with open(f"{a.out}/{a.seed}/insertion.json", "w") as f:
        json.dump(info, f, indent=1)


if __name__ == "__main__":
    main()
