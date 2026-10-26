#!/usr/bin/env python3
"""Insert one breastMass mass into a VICTRE phantom at a chosen candidate location (unmodified pipeline).

Runs inside the breastsim/victre container on <out>/<seed>/p_<seed>.raw.gz:

    docker run --rm -e OMP_NUM_THREADS=2 -u $(id -u):$(id -g) -v $DIR:/out -v $PWD/src/scripts:/scripts \
        breastsim/victre python3 /scripts/victre_insert_at.py --out /out --seed 11 --kind smooth --alpha 5 \
        --mass-seed 7 --loc-seed 3 --depth 8 25

Same mass as victre_insert.py (documented breastMass parameters per morphology, generate_spiculated),
inserted with the documented `locations` argument of Pipeline.insert_lesions. The location is one of
the pipeline's own candidate locations (p_<seed>.loc, in fibroglandular tissue) where the mass, grown by
--margin mm, touches no air / skin / nipple / muscle / lesion voxel (the pipeline's random insertion tests
the whole mass bounding box, 2x the mass size, which keeps every mass >= ~1 diameter from the skin), drawn with
--loc-seed among those whose centre lies --depth mm below the anterior skin (along the chest-wall to
nipple axis), else the shallowest passing one. Why: the random insertion shuffles the candidates with
the phantom seed, so every variant of a phantom lands at the same place, and in the large pendant
(uncompressed) breasts that place can be > 40 mm below every skin point, out of a handheld field of view.
Writes pcl_<seed>.* and insertion.json (+ location, depth).
"""
import argparse
import json
import time

import numpy as np
from Victre import Constants, Pipeline

KINDS = {   # complexity, meanShape (0 spike / 1 bump), spicules (as victre_insert.py)
    "smooth": (0.0, 0.36, 0),
    "lobulated": (1.0, 1.0, 0),
    "spiculated": (1.0, 0.36, 1358),
}


def ap_axis(ph):
    """Chest-wall axis and direction (as breastsim.phantom._orient): the end slice with the largest tissue
    cross-section is the chest wall. Returns (axis, +1 if the nipple is toward higher indices)."""
    best = None
    for ax in range(3):
        occ = (ph > 0).mean(axis=tuple(a for a in range(3) if a != ax))
        idx = np.nonzero(occ)[0]
        lo, hi = occ[idx[0]], occ[idx[-1]]
        if best is None or max(lo, hi) > best[0]:
            best = (max(lo, hi), ax, 1 if lo > hi else -1)
    return best[1], best[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/out")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--kind", choices=KINDS, required=True)
    ap.add_argument("--alpha", type=float, default=5.0, help="breastMass mean radius, mm")
    ap.add_argument("--mass-seed", type=int, default=7)
    ap.add_argument("--loc-seed", type=int, default=0)
    ap.add_argument("--depth", type=float, nargs=2, default=(8.0, 25.0), help="preferred centre depth, mm")
    ap.add_argument("--margin", type=float, default=2.0, help="mm of clearance around the mass")
    a = ap.parse_args()
    cx, shape, spic = KINDS[a.kind]
    params = {**Constants.VICTRE_DEFAULT_SPICULATED_MASS, "complexity": cx, "meanShape": shape,
              "meanInitial": spic, "stdInitial": 365 if spic else 0}
    params.pop("imgRes")                       # the pipeline sets it from the phantom's voxel size
    t0 = time.time()
    p = Pipeline(seed=a.seed, results_folder=a.out, arguments_spiculated=params)
    p.generate_spiculated(seed=a.mass_seed, size=a.alpha)
    import h5py
    from scipy import ndimage
    with h5py.File(p.lesion_file, "r") as f:
        mass = f["volume"][()] == 1
    s = mass.shape
    vox_ = float(p.mhd["ElementSpacing"][0])
    grown = ndimage.binary_dilation(mass, iterations=max(1, int(round(a.margin / vox_))))
    ph = p._load_phantom_array_from_gzip()
    vox = float(p.mhd["ElementSpacing"][0])
    axis, sign = ap_axis(ph)
    banned = np.append(Constants.FORBIDDEN_OVERLAP, list(Constants.LESION_MATERIALS.values()))
    ok = []
    for c in p.candidate_locations:
        # same slicing as insert_lesions(locations=...)
        lo = [int(c[0] - s[0] / 2), int(c[2] - s[2] / 2), int(c[1] - s[1] / 2)]
        if min(lo) < 0:
            continue
        roi = ph[lo[0]:lo[0] + s[0], lo[1]:lo[1] + s[2], lo[2]:lo[2] + s[1]]
        if roi.shape != grown.shape or np.isin(roi[grown], banned).any():
            continue
        ctr = [int(c[0]), int(c[2]), int(c[1])]                     # array index of the mass centre
        idx = list(ctr)
        idx[axis] = slice(None)
        tis = np.nonzero(ph[tuple(idx)] > 0)[0]                      # tissue along the chest-wall axis
        skin = tis.max() if sign > 0 else tis.min()
        ok.append((c, abs(skin - ctr[axis]) * vox))
    if not ok:
        raise SystemExit("no candidate location passes the overlap test")
    rng = np.random.default_rng(a.loc_seed)
    inside = [o for o in ok if a.depth[0] <= o[1] <= a.depth[1]]
    loc, depth = inside[rng.integers(len(inside))] if inside else min(ok, key=lambda o: o[1])
    t1 = time.time()
    p.insert_lesions(lesion_type=Constants.VICTRE_SPICULATED, n=1, locations=[list(loc)])
    info = {"kind": a.kind, "alpha_mm": a.alpha, "n": 1, "mass_seed": a.mass_seed, "loc_seed": a.loc_seed,
            "margin_mm": a.margin,
            "breastMass": {"complexity": cx, "meanShape": shape, "meanInitial": spic},
            "location_vox": [int(x) for x in loc], "centre_depth_mm": round(float(depth), 1),
            "candidates_passing": len(ok), "candidates_in_depth_range": len(inside),
            "mass_seconds": round(t1 - t0, 1), "insert_seconds": round(time.time() - t1, 1)}
    with open(f"{a.out}/{a.seed}/insertion.json", "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
