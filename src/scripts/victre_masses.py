#!/usr/bin/env python3
"""Generate breast masses with the unmodified VICTRE breastMass (Pipeline.generate_spiculated).

Runs inside the breastsim/victre container:

    docker run --rm -u $(id -u):$(id -g) -v $OUT:/out -v $PWD/src/scripts:/scripts breastsim/victre \
        python3 /scripts/victre_masses.py --out /out

Only the documented breastMass parameters are varied (passed through the pipeline's
arguments_spiculated): mean radius `alpha` (mm), `complexity` (scales the surface
irregularity), `meanShape` (low-frequency modifications, 0 = spike, 1 = bump) and the number of
spicules `meanInitial` / `stdInitial` (0 = no spicules). Each mass is saved by the pipeline as
lesions/spiculated/mass_<seed>_size<alpha>.h5; masses.json lists the parameters per file.
"""
import argparse
import glob
import json
import os
import time

from Victre import Constants, Pipeline


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/out")
    ap.add_argument("--voxel", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, default=1)
    args = ap.parse_args()
    # three morphologies x three sizes: smooth (complexity 0, no spicules ~ circumscribed oval/round),
    # lobulated (bump-type low-frequency modifications, no spicules), spiculated (VICTRE default)
    kinds = {"smooth": (0.0, 0.36, 0), "lobulated": (1.0, 1.0, 0), "spiculated": (1.0, 0.36, 1358)}
    grid = [(alpha, *kinds[k], k) for k in kinds for alpha in (2.5, 5.0, 8.0)]   # alpha: mean radius, mm
    out = []
    for k, (alpha, cx, shape, spic, kind) in enumerate(grid):
        for s in range(args.seeds):
            seed = 1000 + 10 * k + s
            folder = f"{args.out}/m{k:02d}"
            params = {**Constants.VICTRE_DEFAULT_SPICULATED_MASS, "imgRes": args.voxel, "complexity": cx,
                      "meanShape": shape, "meanInitial": spic, "stdInitial": 365 if spic else 0}
            t0 = time.time()
            p = Pipeline(seed=seed, results_folder=folder, arguments_spiculated=params)
            p.generate_spiculated(seed=seed, size=alpha)
            f = sorted(glob.glob(f"{folder}/lesions/spiculated/mass_{seed}_size*.h5"))[-1]
            out.append(dict(file=os.path.relpath(f, args.out), seed=seed, kind=kind, alpha=alpha, complexity=cx, meanShape=shape,
                            spicules=spic, voxel=args.voxel, seconds=round(time.time() - t0, 1)))
            print(json.dumps(out[-1]), flush=True)
            with open(f"{args.out}/masses.json", "w") as fh:
                json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
