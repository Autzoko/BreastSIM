#!/usr/bin/env python
"""Image statistics (breastsim.imstats) of the public breast US datasets and of simulated frames.

    python src/scripts/real_image_stats.py --out outputs/real_stats
    python src/scripts/real_image_stats.py --sim outputs/density_grid --out outputs/sim_stats --only SIM

Writes stats.json (one record per image) and summary.csv (medians per dataset / group).
Lesion masks are dilated by 15 px and excluded; BrEaST and simulated frames are also
analysed after resampling to 0.1 mm/px (BrEaST lists its pixel size, BUSI / BUS-BRA do not).
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi

from breastsim import imstats, paths

TARGET_MM = 0.1


def load_gray(f):
    a = np.asarray(Image.open(f).convert("RGB")).astype(np.float32)
    return a.mean(-1), (a.max(-1) - a.min(-1)) > 12          # gray, coloured annotations


def mask_of(files, shape):
    bad = np.zeros(shape, bool)
    for mf in files:
        bad |= ndi.binary_dilation(np.asarray(Image.open(mf).convert("L")) > 127, iterations=15)
    return bad


def busi():
    for f in sorted(glob.glob(str(paths.REAL / "busi/extracted/Dataset_BUSI_with_GT/*/*).png"))):
        g, col = load_gray(f)
        yield f, g, col | mask_of(glob.glob(f[:-4] + "_mask*.png"), g.shape), dict(group=Path(f).parent.name)


def busbra(every=3):
    for f in sorted(glob.glob(str(paths.REAL / "busbra/BUSBRA/Images/*.png")))[::every]:
        g, col = load_gray(f)
        mf = paths.REAL / "busbra/BUSBRA/Masks" / Path(f).name.replace("bus_", "mask_")
        yield f, g, col | mask_of([mf] if mf.exists() else [], g.shape), dict(group="all")


def breast():
    root = paths.REAL / "breast_lesions_usg"
    meta = pd.read_excel(next(root.glob("BrEaST-Lesions-USG-clinical-data-*.xlsx"))).set_index("Image_filename")
    for fn, row in meta.iterrows():
        f = root / "BrEaST-Lesions_USG-images_and_masks" / fn
        g, col = load_gray(f)
        yield (str(f), g, col | mask_of(glob.glob(str(f)[:-4] + "_*.png"), g.shape),
               dict(group=str(row.Tissue_composition), px_mm=float(row.Pixel_size) * 10))   # Pixel_size in cm


def abus(n_vol=10, n_slices=5):
    """TDSC-ABUS volumes (z = sweep, y = depth, x = along the transducer); native B-mode planes."""
    import zipfile
    import tempfile
    zs = sorted((paths.REAL / "abus_tdsc/Train/DATA").glob("*.zip"))
    with zipfile.ZipFile(zs[0]) as z, tempfile.TemporaryDirectory() as tmp:
        for name in sorted(n for n in z.namelist() if n.endswith(".nrrd"))[:n_vol]:
            v, _ = imstats.read_nrrd(z.extract(name, tmp))
            for k in np.linspace(0.25, 0.75, n_slices) * v.shape[0]:
                g = v[int(k)].astype(np.float32)
                yield f"{name}#{int(k)}", g, np.zeros(g.shape, bool), dict(group="abus")


def sims(root):
    """Preview grids (left half = B-mode, 0.078 mm) and k-Wave frames (0.05 mm, 12 px caption)."""
    for f in sorted(glob.glob(str(Path(root) / "*_[12].png"))):
        g, _ = load_gray(f)
        g = g[:, : g.shape[1] // 2 - 4]
        s = Path(f).stem.split("_")
        yield f, g, np.zeros(g.shape, bool), dict(group=f"{s[0]}-{s[1]}", px_mm=0.078)
    for f in sorted(glob.glob(str(Path(root) / "*_kwave.png"))):
        g, _ = load_gray(f)
        g[:12] = 0
        yield f, g, np.zeros(g.shape, bool), dict(group="kwave", px_mm=0.05)


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
