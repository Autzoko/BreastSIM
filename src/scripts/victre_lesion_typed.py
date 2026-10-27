#!/usr/bin/env python3
"""Insert typed lesions (breastsim.lesions recipes from real BrEaST lesions) into a VICTRE phantom with the
unmodified pipeline. Runs inside the breastsim/victre container, with the repository's src mounted:

    docker run --rm -e OMP_NUM_THREADS=2 --cpus 2 -u $(id -u):$(id -g) -v $VARIANT:/out -v $PWD/src:/src \
        -e PYTHONPATH=/opt/VICTRE_PIPELINE:/src breastsim/victre python3 /src/scripts/victre_lesion_typed.py --out /out --seed 1 \
        --n 4 --type-seed 7

<out>/<seed>/p_<seed>.* is a copy of a base phantom (no pcl_* yet). For each of --n lesions a recipe is
drawn (lesions.sample_recipe("<type-seed>#<i>")), then with documented pipeline interfaces only:
- mass: Pipeline.generate_spiculated with the breastMass parameters of the recipe's morphology
  (victre_insert.py KINDS: circumscribed -> smooth, microlobulated / irregular -> lobulated, spiculated) and
  mean radius alpha = eq. diameter / 2.16 (alpha 5 mm -> 10.8 mm measured), clipped to 2-9 mm;
- the mass is inserted from its bounding box (cube, +2 voxels; documented lesion_file argument): the breastMass ROI
  (~2.7x the diameter) does not fit into a 25-37 mm thick supine breast;
- location: a fibroglandular voxel where the mass grown by --margin mm touches no air / skin / nipple /
  muscle / lesion (as victre_insert_at.py), centre depth below the anterior skin within the recipe's depth
  +-3 mm clipped to --depth, else within --depth, else the shallowest; inserted with
  insert_lesions(locations=...);
- calcifications ("in a mass"): Pipeline.generate_cluster (cube of 0.6 x the mass eq. diameter, 3-12
  calcifications of 1-voxel radius; size in whole mm) inserted at the same location (material 250).
Writes pcl_<seed>.* and insertion.json ("lesions": recipe, morphology, alpha, centre in mm of the array).
"""
import argparse
import json
import time

import h5py
import numpy as np
from scipy import ndimage
from Victre import Constants, Pipeline

from breastsim import lesions
from victre_insert_at import KINDS, ap_axis


def pick_location(p, mass, margin_mm, depth, rng, tries=1500):
    """Centre on a random fibroglandular voxel where the mass grown by margin_mm touches no air / skin / nipple /
    muscle / lesion, preferring centre depths (below the anterior skin, along the chest-wall axis) in depth[0],
    then depth[1], else the shallowest passing one. (The pipeline's own candidate list p_<seed>.loc does not
    map into the uncompressed phantom array: most entries fall outside it.) Returns (location for
    insert_lesions, array index of the centre, depth mm)."""
    vox = float(p.mhd["ElementSpacing"][0])
    grown = ndimage.binary_dilation(mass, iterations=max(1, int(round(margin_mm / vox))))
    ph = p._load_phantom_array_from_gzip()
    axis, sign = ap_axis(ph)
    banned = np.append(Constants.FORBIDDEN_OVERLAP, list(Constants.LESION_MATERIALS.values()))
    s, ok = mass.shape, []
    gl = np.argwhere(ph == Constants.PHANTOM_MATERIALS["glandular"])
    for ctr in gl[rng.permutation(len(gl))[:tries]]:
        lo = [int(ctr[i] - s[i] / 2) for i in range(3)]                # as insert_lesions slices
        roi = ph[lo[0]:lo[0] + s[0], lo[1]:lo[1] + s[1], lo[2]:lo[2] + s[2]]
        if min(lo) < 0 or roi.shape != grown.shape or np.isin(roi[grown], banned).any():
            continue
        idx = [int(c) for c in ctr]
        col = list(idx)
        col[axis] = slice(None)
        tis = np.nonzero(ph[tuple(col)] > 0)[0]
        skin = tis.max() if sign > 0 else tis.min()
        cand = [idx[0], idx[2], idx[1]]                     # insert_lesions slices axes (c0, c2, c1)
        ok.append((cand, idx, abs(skin - idx[axis]) * vox))
    if not ok:
        return None
    for lo_, hi_ in depth:
        inside = [o for o in ok if lo_ <= o[2] <= hi_]
        if inside:
            return inside[rng.integers(len(inside))]
    return min(ok, key=lambda o: o[2])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/out")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--type-seed", type=int, default=0)
    ap.add_argument("--depth", type=float, nargs=2, default=(8.0, 20.0), help="allowed centre depth, mm")
    ap.add_argument("--margin", type=float, default=2.0)
    a = ap.parse_args()
    t0 = time.time()
    p = Pipeline(seed=a.seed, results_folder=a.out)
    vox = float(p.mhd["ElementSpacing"][0])
    rng = np.random.default_rng(a.type_seed)
    out = []
    for i in range(a.n):
        r = lesions.sample_recipe(f"{a.type_seed}#{i}")
        kind = lesions.mass_kind(r)
        cx, shape, spic = KINDS[kind]
        alpha = float(np.clip((r["eqdiam_mm"] or 10.0) / 2.16, 2.0, 9.0))
        params = {**Constants.VICTRE_DEFAULT_SPICULATED_MASS, "complexity": cx, "meanShape": shape,
                  "meanInitial": spic, "stdInitial": 365 if spic else 0, "imgRes": vox}
        p.arguments_spiculated.update(params)
        mass_seed = 1000 * a.type_seed + i + 1
        p.generate_spiculated(seed=mass_seed, size=alpha)
        # the breastMass ROI is ~2.7x the mass diameter, more than a supine breast is thick: insert the same
        # mass from its bounding box (+2 voxels) through the documented lesion_file argument
        with h5py.File(p.lesion_file, "r") as f:
            vol = f["volume"][()]
        sl = ndimage.find_objects((vol == 1).astype(np.uint8))[0]
        side = max(q.stop - q.start for q in sl) + 4                   # cube: insert_lesions needs equal sides
        mass = np.zeros((side,) * 3, bool)
        sub = vol[sl] == 1
        o = [(side - n) // 2 for n in sub.shape]
        mass[o[0]:o[0] + sub.shape[0], o[1]:o[1] + sub.shape[1], o[2]:o[2] + sub.shape[2]] = sub
        crop = f"{a.out}/lesions/typed_{a.seed}_{mass_seed}.h5"
        with h5py.File(crop, "w") as f:
            f.create_dataset("volume", data=mass.astype(np.uint8), compression="gzip")
        d_case = r["depth_mm"] or 12.0
        want = (max(a.depth[0], d_case - 3), min(a.depth[1], d_case + 3))
        loc = pick_location(p, mass, a.margin, (want, tuple(a.depth)), rng)
        if loc is None:
            print(json.dumps({"lesion": i, "skipped": "no candidate passes the overlap test"}))
            continue
        cand, ctr, depth = loc
        p.insert_lesions(lesion_type=Constants.VICTRE_SPICULATED, n=1, lesion_file=crop, locations=[cand])
        eq = float((6 * mass.sum() * vox ** 3 / np.pi) ** (1 / 3))
        rec = {"recipe": r, "kind": kind, "alpha_mm": round(alpha, 2), "mass_seed": mass_seed,
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
