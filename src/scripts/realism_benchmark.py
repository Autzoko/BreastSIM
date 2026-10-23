#!/usr/bin/env python
"""Realism benchmark of simulated breast US against the public datasets (breastsim.realism).

    python src/scripts/realism_benchmark.py extract                      # real datasets, once (cached)
    python src/scripts/realism_benchmark.py extract --sim outputs/realism/sim/v5
    python src/scripts/realism_benchmark.py report v5 [v3 ...]           # scores per feature and version

Feature tables are cached as outputs/realism/features/<name>_{images,lesions}.csv. The report
prints, per feature, each version's score against every real dataset (W1 distance divided by
the median W1 between real datasets; <= 1: within the real between-dataset spread) and writes
outputs/realism/report.csv.
"""
import argparse
from pathlib import Path

import pandas as pd

from breastsim import paths, realism

FEAT = paths.OUTPUTS / "realism" / "features"


def load(name):
    return tuple(pd.read_csv(FEAT / f"{name}_{k}.csv") for k in ("images", "lesions"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--sim", help="simulated set folder (render_benchmark_set.py); default: all real datasets")
    e.add_argument("--only", help="comma-separated real datasets")
    rp = sub.add_parser("report")
    rp.add_argument("versions", nargs="+", help="simulated set names (folder names under outputs/realism/sim)")
    args = ap.parse_args()
    FEAT.mkdir(parents=True, exist_ok=True)

    if args.cmd == "extract":
        jobs = [(Path(args.sim).name, realism.sim(args.sim))] if args.sim else \
               [(k, f()) for k, f in realism.READERS.items() if not args.only or k in args.only.split(",")]
        for name, samples in jobs:
            print(f"[realism] {name}", flush=True)
            imgs, lesions = realism.extract(samples, name, log=lambda m: print(m, flush=True))
            imgs.to_csv(FEAT / f"{name}_images.csv", index=False)
            lesions.to_csv(FEAT / f"{name}_lesions.csv", index=False)
            print(f"[realism] {name}: {len(imgs)} images, {len(lesions)} lesions", flush=True)
        return

    real = {k: load(k) for k in realism.READERS if (FEAT / f"{k}_images.csv").exists()}
    tables = []
    for v in args.versions:
        si, sl = load(v)
        for kind, idx in (("image", 0), ("lesion", 1)):
            t = realism.score({k: r[idx] for k, r in real.items()}, (si, sl)[idx], kind)
            t.insert(0, "version", v)
            tables.append(t)
    rep = pd.concat(tables, ignore_index=True)
    rep.to_csv(paths.OUTPUTS / "realism" / "report.csv", index=False)
    wide = rep.pivot_table(index=["part", "group", "feature"], columns="version", values="score_mean").round(2)
    med = rep.drop_duplicates(["part", "group", "feature"]).set_index(["part", "group", "feature"]).filter(like="real_median")
    wide["real_medians (" + " / ".join(c.split(":")[1] for c in med.columns) + ")"] = med.apply(
        lambda r: " / ".join("-" if pd.isna(v) else f"{v:.3g}" for v in r), axis=1)
    sim_med = rep.pivot_table(index=["part", "group", "feature"], columns="version", values="sim_median")
    for v in args.versions:
        wide[f"median:{v}"] = sim_med[v].map(lambda x: f"{x:.3g}")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    print(wide.to_string())
    print("\nmedian score per part / group (imaging: used to tune the imaging chain; content: phantom-driven):")
    print(rep.groupby(["version", "part", "group"]).score_mean.median().unstack([1, 2]).round(2).to_string())


if __name__ == "__main__":
    main()
