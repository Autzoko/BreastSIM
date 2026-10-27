#!/usr/bin/env python
"""Lesion conspicuity of real and simulated sets (breastsim.realism.conspicuity, CONSPICUITY_MIN).

    python src/scripts/lesion_visibility.py les_v3 les_v4 [--panel outputs/phantom_realism/invisible.png]

Prints the conspicuity percentiles of the real datasets (the 2nd percentile over BrEaST + BUSI lesions is the
lower bound used for simulated positive frames), and per simulated set and lesion echo type the number of
lesions below the bound. --panel: frames below the bound in the FIRST set, side by side with the same frame
(same file name) in the other sets, with the lesion outline.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from breastsim import paths, realism

FEAT = paths.OUTPUTS / "realism" / "features"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sets", nargs="+")
    ap.add_argument("--panel")
    a = ap.parse_args()
    real = []
    for d in ("BrEaST", "BUSI", "BUS-BRA", "TDSC-ABUS"):
        t = pd.read_csv(FEAT / f"{d}_lesions.csv").dropna(subset=["contrast_sd", "edge_sharpness"])
        c = realism.conspicuity(t)
        print(f"{d:10s} n={len(t):5d} conspicuity p1/p2/p5/p50 = {np.percentile(c, [1, 2, 5, 50]).round(3)}")
        if d in ("BrEaST", "BUSI"):
            real.append(c)
    print(f"BrEaST+BUSI p2 = {np.percentile(np.concatenate(real), 2):.3f} (realism.CONSPICUITY_MIN = {realism.CONSPICUITY_MIN})")
    for s in a.sets:
        t = pd.read_csv(FEAT / f"{s}_lesions.csv").dropna(subset=["contrast_sd", "edge_sharpness"])
        t["low"] = realism.conspicuity(t) < realism.CONSPICUITY_MIN
        by = t.groupby(t.get("lesion_echo", pd.Series("all", index=t.index))).low.agg(["sum", "size"])
        print(f"{s}: {int(t.low.sum())} / {len(t)} lesions below the bound;", {k: f"{int(r['sum'])}/{int(r['size'])}" for k, r in by.iterrows()})
    if a.panel:
        t = pd.read_csv(FEAT / f"{a.sets[0]}_lesions.csv").dropna(subset=["contrast_sd", "edge_sharpness"])
        low = t[realism.conspicuity(t) < realism.CONSPICUITY_MIN].id.unique()[:8]
        import json
        key = lambda m: (m["phantom"], round(m["u"], 3), round(m["v"], 3), round(m["rotation"], 3))
        metas = {s: {json.loads(l)["file"]: json.loads(l) for l in
                     (paths.OUTPUTS / "realism" / "sim" / s / "meta.jsonl").read_text().splitlines()} for s in a.sets}
        by_pose = {s: {key(m): f for f, m in metas[s].items()} for s in a.sets}
        rows = []
        for f0 in low:
            tiles = []
            for s in a.sets:                                   # the same pose in every set (file numbers may shift)
                f = by_pose[s].get(key(metas[a.sets[0]][f0]))
                if f is None:
                    tiles.append(np.zeros((330, 300, 3), np.uint8))
                    continue
                root = paths.OUTPUTS / "realism" / "sim" / s
                g = np.asarray(Image.open(root / "img" / f).convert("L"))
                les = np.asarray(Image.open(root / "lesion" / f).convert("L")) > 127
                rgb = np.repeat(g[..., None], 3, -1)
                from scipy import ndimage
                rgb[les & ~ndimage.binary_erosion(les, iterations=2)] = (60, 255, 60)
                tiles.append(np.asarray(Image.fromarray(rgb).resize((300, 330))))
            rows.append(np.concatenate(tiles, 1))
        if rows:
            Image.fromarray(np.concatenate(rows, 0)).save(a.panel)
            print(a.panel, list(low))


if __name__ == "__main__":
    main()
