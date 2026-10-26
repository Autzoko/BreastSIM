#!/usr/bin/env python3
"""Generate VICTRE breast phantoms with the unmodified FDA VICTRE_PIPELINE.

Runs inside the breastsim/victre container (src/docker/victre), where `Victre` is importable:

    docker run --rm -u $(id -u):$(id -g) -v $OUT:/out -v $PWD/src/scripts:/scripts breastsim/victre \
        python3 /scripts/victre_generate.py --seed 1 --density hetero --voxel 0.2 --out /out

Only the anatomy step of Victre.Pipeline is used (generate_phantom); the X-ray steps
(project / reconstruct) are never called. Generation parameters are passed through the
pipeline's own `arguments_generation` interface (the phantom software is designed to be
parameterised: density, voxel size, shape coefficients...). Output: <out>/<seed>/p_<seed>.raw.gz
+ .mhd (+ the generator's log), exactly as the pipeline writes them.
"""
import argparse
import json
import time

from Victre import Constants, Pipeline

DENSITY = {"dense": Constants.VICTRE_DENSE, "hetero": Constants.VICTRE_HETERO,
           "scattered": Constants.VICTRE_SCATTERED, "fatty": Constants.VICTRE_FATTY}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--density", choices=DENSITY, default="hetero")
    ap.add_argument("--voxel", type=float, default=0.2, help="imgRes in mm (VICTRE default 0.05)")
    ap.add_argument("--out", default="/out")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="extra arguments_generation overrides (JSON values), e.g. --set surface_a3=0.9 --set surface_doPtosis=false")
    args = ap.parse_args()

    overrides = {"imgRes": args.voxel}
    for kv in args.set:
        k, v = kv.split("=", 1)
        # keys are the pipeline's template names (surface_a3, surface_doPtosis, ligament_thickness, ...);
        # string.Template silently ignores unknown keys, so a misspelt key would be a no-op
        if k not in DENSITY[args.density]:
            ap.error(f"unknown generation parameter {k!r} (keys of Victre.Constants.VICTRE_*, e.g. surface_a3)")
        # booleans stay the pipeline's "true" / "false" strings
        overrides[k] = v if v in ("true", "false") else json.loads(v)
    t0 = time.time()
    pline = Pipeline(seed=args.seed, results_folder=args.out,
                     arguments_generation={**DENSITY[args.density], **overrides})
    pline.generate_phantom()
    info = {"seed": args.seed, "density": args.density, "overrides": overrides, "seconds": round(time.time() - t0, 1)}
    with open(f"{args.out}/{args.seed}/generation.json", "w") as f:   # read by breastsim.phantom (description)
        json.dump(info, f, indent=1)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
