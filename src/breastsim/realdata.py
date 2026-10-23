"""Readers of the public breast US datasets under data/real, for image statistics.

Each reader yields (id, gray image float32 (H, W), excluded mask bool (H, W), info dict).
Excluded = coloured annotations and lesion masks dilated by 15 px. info["group"] is the
dataset's subgroup (BUSI class, BrEaST tissue composition); info["px_mm"] the pixel size
when the dataset gives it (BrEaST only; TDSC-ABUS volumes carry unit spacing).
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi

from . import imstats, paths


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


READERS = {"BUSI": busi, "BUS-BRA": busbra, "BrEaST": breast, "TDSC-ABUS": abus}
