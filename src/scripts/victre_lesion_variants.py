#!/usr/bin/env python
"""Lesion variants of VICTRE phantoms with the unmodified pipeline (victre_insert.py in breastsim/victre).

    python src/scripts/victre_lesion_variants.py --bases ds_fatty/11 ds_dense/14 ... --per-base 3

For each base phantom <BIG>/victre/<variant>/<seed> (never modified) and k < per-base, the base
files are copied to <BIG>/victre/<variant>_les<k>/<seed>/ and victre_insert.py inserts n breastMass
mass there (writes pcl_<seed>; phantom.list_phantoms then registers victre/<variant>_les<k>_<seed>
with the lesions). Per variant, drawn with a fixed seed: morphology (smooth / lobulated / spiculated,
the documented breastMass settings in victre_insert.KINDS), mean radius alpha (log-uniform
2.5-8 mm: eq. diameters ~5-16 mm around the clinical ~10 mm) and the mass seed.
One mass per phantom: insert_lesions tests the mass's bounding box (~2x its diameter) against skin /
muscle / nipple / other lesions and restarts the whole insertion when one fails; with two 10 mm masses
the second box overlapped the first at every candidate and it retried up to 1000 times. A run that
exceeds --timeout is stopped and retried with alpha x 0.75 (up to two times).
Docker containers run with OMP_NUM_THREADS = --cpus (breastMass / breastPhantomMain open an OpenMP
region per sample; with the default 48 threads under a 2-CPU quota they spin and run ~50x slower).
"""
import argparse
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
    ap.add_argument("--n", type=int, default=1, help="masses per phantom")
    ap.add_argument("--timeout", type=float, default=180,
                    help="s per insertion run before retrying smaller (successful runs take 10-30 s)")
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
            jobs.append(dict(name=f"{variant}_les{k}/{seed}", dst=dst, seed=seed, kind=kind, alpha=alpha, tries=0,
                             mass_seed=int(rng.integers(1, 10_000))))
    running = []
    while jobs or running:
        while jobs and len(running) < a.parallel:
            j = jobs.pop(0)
            cname = f"ds_ins_{j['name'].replace('/', '_')}"
            cmd = ["docker", "run", "--rm", "--name", cname, "--cpus", str(a.cpus), "-e", f"OMP_NUM_THREADS={a.cpus}",
                   "-u", f"{_id('-u')}:{_id('-g')}", "-v", f"{j['dst'].parent}:/out", "-v", f"{scripts}:/scripts",
                   "breastsim/victre", "python3", "/scripts/victre_insert.py", "--out", "/out", "--seed", j["seed"],
                   "--kind", j["kind"], "--alpha", str(j["alpha"]), "--n", str(a.n), "--mass-seed", str(j["mass_seed"])]
            log = open(paths.VICTRE / f"{cname}.log", "w")
            running.append((j, cname, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), time.time()))
            print(f"start {j['name']}: {j['kind']} alpha {j['alpha']} mass seed {j['mass_seed']}", flush=True)
        for r in list(running):
            j, cname, proc, t0 = r
            if proc.poll() is None and time.time() - t0 > a.timeout:
                subprocess.run(["docker", "kill", cname], capture_output=True)   # our own container, by name
                proc.wait()
            if proc.poll() is not None:
                running.remove(r)
                ok = (j["dst"] / "insertion.json").exists()
                print(f"{'done' if ok else 'FAILED'} {j['name']} exit {proc.returncode} in {time.time() - t0:.0f} s", flush=True)
                if not ok and j["tries"] < 2:
                    for f in (j["dst"].parent / "lesions").glob(f"spiculated/*_{j['mass_seed']}*"):
                        f.unlink()   # only this job's mass (variants of other seeds share the folder)
                    jobs.append({**j, "alpha": round(j["alpha"] * 0.75, 2), "tries": j["tries"] + 1})
        time.sleep(10)


def _id(flag):
    return subprocess.check_output(["id", flag], text=True).strip()


if __name__ == "__main__":
    main()
