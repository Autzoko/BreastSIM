#!/usr/bin/env python3
"""Insert breastMass lesions into a VICTRE phantom with the unmodified pipeline (insert_lesions).

Runs inside the breastsim/victre container on a phantom folder <out>/<seed>/p_<seed>.raw.gz:

    docker run --rm -u $(id -u):$(id -g) -v $DIR:/out -v $PWD/src/scripts:/scripts breastsim/victre \
        python3 /scripts/victre_insert.py --out /out --seed 1 --kind smooth --alpha 5 --n 3

The mass is generated at the phantom's voxel size (the pipeline inserts voxels without resampling)
with the documented breastMass parameters of the chosen morphology; the pipeline places it at
candidate locations from p_<seed>.loc that do not overlap forbidden tissues, and writes
pcl_<seed>.raw.gz / .mhd / .loc. insertion.json records what was inserted.
"""
import argparse
import json
import time

from Victre import Constants, Pipeline

KINDS = {   # complexity, meanShape (0 spike / 1 bump), spicules
    "smooth": (0.0, 0.36, 0),
    "lobulated": (1.0, 1.0, 0),
    "spiculated": (1.0, 0.36, 1358),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/out")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--kind", choices=KINDS, required=True)
    ap.add_argument("--alpha", type=float, default=5.0, help="breastMass mean radius, mm")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--mass-seed", type=int, default=7)
    args = ap.parse_args()
    cx, shape, spic = KINDS[args.kind]
    params = {**Constants.VICTRE_DEFAULT_SPICULATED_MASS, "complexity": cx, "meanShape": shape,
              "meanInitial": spic, "stdInitial": 365 if spic else 0}
    params.pop("imgRes")                       # the pipeline sets it from the phantom's voxel size
    t0 = time.time()
    p = Pipeline(seed=args.seed, results_folder=args.out, arguments_spiculated=params)
    p.generate_spiculated(seed=args.mass_seed, size=args.alpha)
    t1 = time.time()
    p.insert_lesions(lesion_type=Constants.VICTRE_SPICULATED, n=args.n)
    info = {"kind": args.kind, "alpha_mm": args.alpha, "n": args.n, "mass_seed": args.mass_seed,
            "breastMass": {"complexity": cx, "meanShape": shape, "meanInitial": spic},
            "mass_seconds": round(t1 - t0, 1), "insert_seconds": round(time.time() - t1, 1)}
    with open(f"{args.out}/{args.seed}/insertion.json", "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
