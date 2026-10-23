#!/usr/bin/env python
"""Image statistics (breastsim.imstats) of the public breast US datasets and of simulated frames.

    python src/scripts/real_image_stats.py --out outputs/real_stats
    python src/scripts/real_image_stats.py --sim outputs/density_grid --out outputs/sim_stats --only SIM

Writes stats.json (one record per image) and summary.csv (medians per dataset / group).
Lesion masks are dilated by 15 px and excluded; BrEaST and simulated frames are also
analysed after resampling to 0.1 mm/px (BrEaST lists its pixel size, BUSI / BUS-BRA do not).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from breastsim import imstats
from breastsim.realdata import abus, breast, busbra, busi, sims

TARGET_MM = 0.1


def run(name, items, calibrated=False):
    res, nerr = [], 0
    for f, g, bad, extra in items:
        scale = extra.get("px_mm", TARGET_MM) / TARGET_MM if calibrated else 1.0
        try:
            r = imstats.analyse(g, imstats.tissue_mask(g, bad), scale)
        except Exception as e:  # keep going; report the first few
            nerr += 1
            if nerr <= 5:
                print(name, f, repr(e), flush=True)
            r = None
        if r:
            res.append(dict(dataset=name + ("@0.1mm" if calibrated else ""), file=str(f), **extra, **r))
    print(f"{name}{'@0.1mm' if calibrated else ''}: {len(res)} images", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="outputs/real_stats")
    ap.add_argument("--sim", nargs="*", default=[], help="folders of simulated frames")
    ap.add_argument("--only", help="comma-separated subset of BUSI,BUS-BRA,BrEaST,TDSC-ABUS,SIM")
    args = ap.parse_args()
    want = set(args.only.split(",")) if args.only else {"BUSI", "BUS-BRA", "BrEaST", "TDSC-ABUS", "SIM"}
    res = []
    if "BUSI" in want:
        res += run("BUSI", busi())
    if "BUS-BRA" in want:
        res += run("BUS-BRA", busbra())
    if "BrEaST" in want:
        res += run("BrEaST", breast()) + run("BrEaST", breast(), calibrated=True)
    if "TDSC-ABUS" in want:
        res += run("TDSC-ABUS", abus())
    if "SIM" in want:
        for s in args.sim:
            res += run(f"SIM:{Path(s).name}", sims(s)) + run(f"SIM:{Path(s).name}", sims(s), calibrated=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "stats.json").write_text(json.dumps(res))
    d = pd.DataFrame(res)
    d["key"] = d.dataset + ":" + d.group.astype(str)
    scal = [c for c in d.columns if c not in ("dataset", "file", "group", "key", "px_mm")
            and np.issubdtype(d[c].dtype, np.number)]
    t = d.groupby("key")[scal].median().round(3)
    t.insert(0, "n", d.groupby("key").size())
    t.to_csv(out / "summary.csv")
    print(t.to_string())


if __name__ == "__main__":
    main()
