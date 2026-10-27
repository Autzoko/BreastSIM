#!/usr/bin/env python
"""Pack the real breast-US datasets into one HDF5 per dataset for segmentation training / evaluation.

    python src/train/prep_real.py --out $BIG/realseg --size 256

Images with a lesion only (BUSI / BrEaST "normal" images and TDSC-ABUS normal planes are left out).
Gray (mean of RGB, as breastsim.realism reads them) and lesion mask resized to size x size
(bilinear / nearest). Fixed splits (seeded, grouped by case so one patient never spans splits):
  BUS-BRA   official 5-fold-cv.csv, kFold 1 = test (patients grouped by the dataset authors)
  BrEaST    20 % of CaseIDs test
  BUSI      20 % of images test, stratified benign / malignant (no patient ids are published)
  TDSC-ABUS lesion plane of the 100 training volumes, 20 % of volumes test; each 154 mm sweep frame is cut
            to a 42 mm window (as the handheld images) at a random offset that contains the lesion
Datasets: img (N, S, S) uint8, mask (N, S, S) uint8, split (N,) "train" | "test", meta (N,) JSON.
Meta per image for breakdowns: "pathology" (benign / malignant: BUSI folder, BUS-BRA Pathology, BrEaST
Classification, TDSC-ABUS label) and "cystic" (BUS-BRA histology "cyst"; BrEaST echogenicity "anechoic" or
"complex cystic/solid"; None where the dataset has no such annotation).
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image

from breastsim import paths, realism


def resize(a, size, mode):
    return np.asarray(Image.fromarray(a).resize((size, size), mode))


def abus_window(g, les, px_mm, rng, width_mm=42.0):
    w = int(round(width_mm / px_mm))
    cols = np.nonzero(les.any(0))[0]
    lo, hi = max(0, cols.max() + 1 - w), min(cols.min(), g.shape[1] - w)
    x0 = int(rng.integers(lo, hi + 1)) if hi >= lo else max(0, int(cols.mean()) - w // 2)
    return g[:, x0:x0 + w], les[:, x0:x0 + w]


def group_of(name, s):
    if name == "BUS-BRA":
        return Path(s["id"]).stem                 # bus_0001-l; split from 5-fold-cv.csv
    if name == "BrEaST":
        return Path(s["id"]).stem                 # case001 (one image per CaseID)
    if name == "TDSC-ABUS":
        return s["id"].split("#")[0]              # volume
    return Path(s["id"]).name


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--datasets", default="BUSI,BUS-BRA,BrEaST,TDSC-ABUS")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    folds = pd.read_csv(paths.REAL / "busbra/BUSBRA/5-fold-cv.csv").set_index("ID").kFold
    histology = pd.read_csv(paths.REAL / "busbra/BUSBRA/bus_data.csv").set_index("ID").Histology.str.lower()
    breast_x = pd.read_excel(next((paths.REAL / "breast_lesions_usg").glob("BrEaST-Lesions-USG-clinical-data-*.xlsx")))
    echogenicity = breast_x.set_index(breast_x.Image_filename.str.replace(".png", "", regex=False)).Echogenicity
    for name in a.datasets.split(","):
        rows = []
        rng = np.random.default_rng(a.seed)
        reader = realism.READERS[name]
        for s in (reader(normal_slices=0) if name == "TDSC-ABUS" else reader()):
            if not s["lesion"].any():
                continue
            g = np.clip(s["gray"], 0, 255).astype(np.uint8)
            if name == "TDSC-ABUS":   # 154 mm wide sweep frame -> 42 mm window (handheld field of view) at a
                g, s["lesion"] = abus_window(g, s["lesion"], s["px_mm"], rng)   # random offset containing the lesion
            rows.append(dict(img=resize(g, a.size, Image.BILINEAR),
                             mask=resize(s["lesion"].astype(np.uint8), a.size, Image.NEAREST),
                             meta=dict(id=str(s["id"]), case=group_of(name, s), img_shape=list(g.shape),
                                       px_mm=s["px_mm"], lesion_frac=round(float(s["lesion"].mean()), 4),
                                       **{k: str(v) for k, v in s["info"].items()})))
            m = rows[-1]["meta"]
            m["pathology"] = m.get("group")
            m["cystic"] = (histology.get(m["case"]) == "cyst" if name == "BUS-BRA" else
                           echogenicity.get(m["case"]) in ("anechoic", "complex cystic/solid") if name == "BrEaST" else None)
        rng = np.random.default_rng(a.seed)
        if name == "BUS-BRA":
            test = {r["meta"]["case"] for r in rows if folds.get(r["meta"]["case"]) == 1}
        elif name == "BUSI":       # stratified by class (info "group": benign / malignant)
            test = set()
            for cls in ("benign", "malignant"):
                g = sorted(r["meta"]["case"] for r in rows if r["meta"]["group"] == cls)
                test |= set(rng.choice(g, int(round(0.2 * len(g))), replace=False))
        else:
            g = sorted({r["meta"]["case"] for r in rows})
            test = set(rng.choice(g, int(round(0.2 * len(g))), replace=False))
        split = ["test" if r["meta"]["case"] in test else "train" for r in rows]
        with h5py.File(out / f"{name}.h5", "w") as f:
            f.create_dataset("img", data=np.stack([r["img"] for r in rows]), compression="gzip")
            f.create_dataset("mask", data=np.stack([r["mask"] for r in rows]), compression="gzip")
            f["split"] = np.array(split, dtype="S5")
            f["meta"] = np.array([json.dumps(r["meta"]) for r in rows], dtype=h5py.string_dtype())
            f.attrs["size"] = a.size
        print(f"{name}: {len(rows)} lesion images, {split.count('train')} train / {split.count('test')} test, "
              f"lesion fraction median {np.median([r['meta']['lesion_frac'] for r in rows]):.3f}", flush=True)


if __name__ == "__main__":
    main()
