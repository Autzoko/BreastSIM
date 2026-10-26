#!/usr/bin/env python
"""Lesion variants of VICTRE phantoms with the unmodified pipeline (victre_insert.py in breastsim/victre).

    python src/scripts/victre_lesion_variants.py --bases ds_fatty/11 ds_dense/14 ... --per-base 3

For each base phantom <BIG>/victre/<variant>/<seed> (never modified) and k < per-base, the base
files are copied to <BIG>/victre/<variant>_les<k>/<seed>/ and victre_insert.py inserts n breastMass
masses there (writes pcl_<seed>; phantom.list_phantoms then registers victre/<variant>_les<k>_<seed>
with the lesions). Per variant, drawn with a fixed seed: morphology (smooth / lobulated / spiculated,
the documented breastMass settings in victre_insert.KINDS), mean radius alpha (log-uniform
2.5-8 mm: eq. diameters ~5-16 mm around the clinical ~10 mm) and the mass seed.
Docker containers run with OMP_NUM_THREADS = --cpus (breastMass / breastPhantomMain open an OpenMP
region per sample; with the default 48 threads under a 2-CPU quota they spin and run ~50x slower).
"""
import argparse
import json
import shutil
import subprocess
import time
import zlib
from pathlib import Path

import numpy as np

from breastsim import paths

KINDS = ("smooth", "lobulated", "spiculated")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bases", nargs="+", required=True, help="<variant>/<seed> under $BREASTSIM_BIG/victre")
    ap.add_argument("--per-base", type=int, default=3)
    ap.add_argument("--n", type=int, default=2, help="masses per phantom")
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--cpus", type=int, default=2)
    a = ap.parse_args()
    scripts = Path(__file__).resolve().parent
    jobs = []
    for base in a.bases:
        variant, seed = base.split("/")
        src = paths.VICTRE / variant / seed
        for k in range(a.per_base):
            rng = np.random.default_rng(zlib.crc32(f"{base}#les{k}".encode()))
            kind = KINDS[rng.integers(3)]
            alpha = round(float(np.exp(rng.uniform(np.log(2.5), np.log(8.0)))), 2)
            dst = paths.VICTRE / f"{variant}_les{k}" / seed
            if (dst / "insertion.json").exists():
                continue
            dst.mkdir(parents=True, exist_ok=True)
            for f in src.glob(f"p_{seed}.*"):
                if f.suffix != ".vti":
                    shutil.copy2(f, dst / f.name)
            shutil.copy2(src / "generation.json", dst / "generation.json")
            cmd = ["docker", "run", "--rm", "--name", f"ds_ins_{variant}_les{k}_{seed}", "--cpus", str(a.cpus),
                   "-e", f"OMP_NUM_THREADS={a.cpus}", "-u", f"{_id('-u')}:{_id('-g')}",
                   "-v", f"{dst.parent}:/out", "-v", f"{scripts}:/scripts", "breastsim/victre",
                   "python3", "/scripts/victre_insert.py", "--out", "/out", "--seed", seed, "--kind", kind,
                   "--alpha", str(alpha), "--n", str(a.n), "--mass-seed", str(int(rng.integers(1, 10_000)))]
            jobs.append((f"{variant}_les{k}/{seed}", cmd))
    running = []
    while jobs or running:
        while jobs and len(running) < a.parallel:
            name, cmd = jobs.pop(0)
            log = open(paths.VICTRE / f"{name.replace('/', '_')}.log", "w")
            running.append((name, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), time.time()))
            print(f"start {name}: {' '.join(cmd[-10:])}", flush=True)
        for r in list(running):
            if r[1].poll() is not None:
                print(f"done {r[0]} exit {r[1].returncode} in {time.time() - r[2]:.0f} s", flush=True)
                running.remove(r)
        time.sleep(10)


def _id(flag):
    return subprocess.check_output(["id", flag], text=True).strip()


if __name__ == "__main__":
    main()
