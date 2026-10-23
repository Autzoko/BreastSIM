#!/usr/bin/env python
"""Lesion statistics of the public breast US datasets and of simulated frames, on the same footing.

    python src/scripts/lesion_stats.py --out outputs/lesion_stats            # real + simulated
    python src/scripts/lesion_stats.py --only SIM --out outputs/lesion_stats_sim

Per lesion (connected component of the expert / label mask, >= 50 px):
- size: area, equivalent diameter, width (lateral) and height (depth) extents, in px and in mm
  where the pixel size is known (BrEaST, TDSC-ABUS spacing from the challenge, simulation);
- shape: height / width (> 1 = taller than wide), principal-axis angle from horizontal,
  solidity (area / convex hull), circularity (4 pi A / P^2), margin irregularity (std / mean of
  the boundary's distance to the centroid);
- position: depth of the lesion centre below the top of the tissue (mm where known);
- appearance: median gray inside minus a surrounding ring (0.15-0.5 equivalent diameters out),
  the same in units of the ring std, internal / ring std, boundary gradient / ring gradient;
- posterior: median gray below the lesion (0.5 diameter, central 60 % of its columns) minus the
  same depth band beside it (enhancement > 0, shadowing < 0).
TDSC-ABUS: 3D components of the mask volumes; extents in mm and the 2D features on the native
B-mode plane through the lesion centroid.
"""
import argparse
import glob
import json
import zipfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from skimage.measure import perimeter
from skimage.morphology import convex_hull_image

from breastsim import imstats, paths, tissue
from breastsim.realdata import load_gray

ABUS_SPACING = (0.476, 0.073, 0.2)   # mm (sweep, depth, lateral), as in breastsim.presets (TDSC-ABUS 2023)


def features(g, les, px_mm=None, tissue_m=None):
    """2D features of one lesion mask `les` in gray image g (0-255)."""
    rows, cols = np.nonzero(les)
    area = float(les.sum())
    d = 2 * np.sqrt(area / np.pi)
    h, w = rows.max() - rows.min() + 1, cols.max() - cols.min() + 1
    cy, cx = rows.mean(), cols.mean()
    mu20, mu02, mu11 = ((cols - cx) ** 2).mean(), ((rows - cy) ** 2).mean(), ((cols - cx) * (rows - cy)).mean()
    angle = 0.5 * np.degrees(np.arctan2(2 * mu11, mu20 - mu02))         # 0 = long axis horizontal
    edge = les & ~ndi.binary_erosion(les)
    er, ec = np.nonzero(edge)
    rad = np.hypot(er - cy, ec - cx)
    hull = convex_hull_image(les)
    valid = tissue_m if tissue_m is not None else np.ones_like(les)
    near = ndi.binary_dilation(les, iterations=max(1, int(0.15 * d)))
    far = ndi.binary_dilation(les, iterations=max(2, int(0.5 * d)))
    ring = far & ~near & valid
    out = dict(area_px=area, eqdiam_px=d, width_px=float(w), height_px=float(h), h_over_w=float(h / w),
               angle_deg=float(abs(angle)), solidity=float(area / hull.sum()),
               circularity=float(4 * np.pi * area / max(perimeter(les), 1.0) ** 2),
               irregularity=float(rad.std() / rad.mean()))
    if px_mm:
        out.update(eqdiam_mm=d * px_mm, width_mm=w * px_mm, height_mm=h * px_mm)
        top = np.nonzero(valid[:, int(cx)])[0]
        if len(top):
            out["depth_mm"] = float((cy - top.min()) * px_mm)
    if ring.sum() > 30:
        gin, gr = g[les], g[ring]
        out.update(contrast=float(np.median(gin) - np.median(gr)),
                   contrast_sd=float((np.median(gin) - np.median(gr)) / max(gr.std(), 1e-3)),
                   inner_heterogeneity=float(gin.std() / max(gr.std(), 1e-3)))
        gm = np.hypot(*np.gradient(ndi.gaussian_filter(g, 1.0)))
        out["edge_sharpness"] = float(gm[edge].mean() / max(gm[ring].mean(), 1e-3))
    # posterior band vs the same depth band beside the lesion
    b0, b1 = rows.max() + 1, min(g.shape[0], rows.max() + 1 + max(3, int(0.5 * d)))
    c0, c1 = cols.min() + int(0.2 * w), cols.max() + 1 - int(0.2 * w)
    s0, s1 = max(0, cols.min() - int(0.5 * d)), min(g.shape[1], cols.max() + 1 + int(0.5 * d))
    if b1 - b0 >= 3 and c1 > c0:
        below = g[b0:b1, c0:c1][valid[b0:b1, c0:c1]]
        side_m = np.zeros(g.shape, bool)
        side_m[b0:b1, s0:cols.min()] = True
        side_m[b0:b1, cols.max() + 1:s1] = True
        side = g[side_m & valid & ~far]
        if below.size > 20 and side.size > 20:
            out["posterior"] = float(np.median(below) - np.median(side))
    return out


def components(mask, min_px=50):
    lab, n = ndi.label(mask)
    for k in range(1, n + 1):
        m = lab == k
        if m.sum() >= min_px:
            yield m


def real_2d():
    """(dataset, id, gray, [lesion masks], px_mm, info) for BUSI, BUS-BRA, BrEaST."""
    for f in sorted(glob.glob(str(paths.REAL / "busi/extracted/Dataset_BUSI_with_GT/*/*).png"))):
        group = Path(f).parent.name
        if group == "normal":
            continue
        g, _ = load_gray(f)
        m = np.zeros(g.shape, bool)
        for mf in glob.glob(f[:-4] + "_mask*.png"):
            m |= np.asarray(Image.open(mf).convert("L")) > 127
        yield "BUSI", f, g, m, None, dict(pathology=group)
    meta = pd.read_csv(paths.REAL / "busbra/BUSBRA/bus_data.csv").set_index("ID")
    for f in sorted(glob.glob(str(paths.REAL / "busbra/BUSBRA/Images/*.png"))):
        g, _ = load_gray(f)
        mf = paths.REAL / "busbra/BUSBRA/Masks" / Path(f).name.replace("bus_", "mask_")
        row = meta.loc[Path(f).stem]
        yield "BUS-BRA", f, g, np.asarray(Image.open(mf).convert("L")) > 127, None, \
            dict(pathology=row.Pathology, birads=str(row.BIRADS))
    root = paths.REAL / "breast_lesions_usg"
    cm = pd.read_excel(next(root.glob("BrEaST-Lesions-USG-clinical-data-*.xlsx")))
    for _, row in cm.iterrows():
        if not isinstance(row.Mask_tumor_filename, str):
            continue
        f = root / "BrEaST-Lesions_USG-images_and_masks" / row.Image_filename
        g, _ = load_gray(f)
        m = np.zeros(g.shape, bool)
        for mf in row.Mask_tumor_filename.split("&"):
            if (root / "BrEaST-Lesions_USG-images_and_masks" / mf.strip()).exists():
                m |= np.asarray(Image.open(root / "BrEaST-Lesions_USG-images_and_masks" / mf.strip()).convert("L")) > 127
        yield "BrEaST", str(f), g, m, float(row.Pixel_size) * 10, dict(
            pathology=row.Classification, shape=row.Shape, margin=row.Margin, echogenicity=row.Echogenicity,
            posterior_label=row.Posterior_features, composition=row.Tissue_composition)


def abus_lesions():
    labels = pd.read_csv(paths.REAL / "abus_tdsc/Train/labels.csv")
    dz = zipfile.ZipFile(next((paths.REAL / "abus_tdsc/Train/MASK").glob("*.zip")))
    data_zips = [zipfile.ZipFile(z) for z in sorted((paths.REAL / "abus_tdsc/Train/DATA").glob("*.zip"))]
    with tempfile.TemporaryDirectory() as tmp:
        for _, row in labels.iterrows():
            dname, mname = Path(row.data_path.replace("\\", "/")).name, Path(row.mask_path.replace("\\", "/")).name
            zf = next(z for z in data_zips if dname in z.namelist())
            v, _ = imstats.read_nrrd(zf.extract(dname, tmp))
            m, _ = imstats.read_nrrd(dz.extract(mname, tmp))
            for les in components(m > 0, min_px=200):
                zz, yy, xx = np.nonzero(les)
                k = int(round(zz.mean()))
                ext = [(a.max() - a.min() + 1) * s for a, s in zip((zz, yy, xx), ABUS_SPACING)]
                # native B-mode plane resampled to isotropic 0.2 mm (depth 0.073 -> 0.2 mm)
                sy = ABUS_SPACING[1] / ABUS_SPACING[2]
                g = ndi.zoom(v[k].astype(np.float32), (sy, 1.0), order=1)
                m2 = ndi.zoom(les[k].astype(np.float32), (sy, 1.0), order=0) > 0.5
                yield "TDSC-ABUS", f"{dname}#{k}", g, m2, ABUS_SPACING[2], dict(
                    pathology="malignant" if row.label == "M" else "benign", volume_mm3=float(les.sum() * np.prod(ABUS_SPACING)),
                    ext_sweep_mm=ext[0], ext_depth_mm=ext[1], ext_lateral_mm=ext[2])
            Path(tmp, dname).unlink()
            Path(tmp, mname).unlink()


def sim_lesions(preset="breast", rotations=(0.0, 60.0, 120.0)):
    """Preview frames through every lesion of the lesion phantoms, at three probe rotations."""
    from breastsim import phantom
    from breastsim.probe import ProbePose
    from breastsim.realtime.renderer import ImagingParams, PreviewRenderer
    r = PreviewRenderer()
    p = ImagingParams.from_preset(preset, overlay="none")
    pids = [i.id for i in phantom.list_phantoms() if (i.source == "oat64" and i.id.endswith("l")) or i.source == "usct"]
    for pid in pids:
        vol = phantom.load(pid, log=lambda *a: None)
        r.set_volume(vol)
        for les in vol.meta.get("lesions", []):
            u, v = les["center_mm"][1:]
            for rot in rotations:
                img, meta, lab = r.render(ProbePose(u=float(u), v=float(v), rotation=rot), p)
                m = np.isin(lab, tissue.LESIONS)
                if not m.any():
                    continue
                # the component under the probe centre column (the targeted lesion)
                lab_c, n = ndi.label(m)
                ids = np.unique(lab_c[:, lab.shape[1] // 2 - 5:lab.shape[1] // 2 + 5])
                ids = ids[ids > 0]
                if not len(ids):
                    continue
                les_m = np.isin(lab_c, ids)
                if les_m[:, [0, -1]].any() or les_m[-1].any():
                    continue                                          # cut by the image border
                yield "SIM", f"{pid}#{les['id']}@{rot:.0f}", img[..., 0].astype(np.float32), les_m, meta["px_z_mm"], dict(
                    pathology="malignant", source=pid.split("/")[0], posterior_drawn=les.get("posterior"),
                    alpha_drawn=les.get("attenuation_db_cm_mhz"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="outputs/lesion_stats")
    ap.add_argument("--only", help="comma-separated subset of REAL2D,ABUS,SIM")
    args = ap.parse_args()
    want = set(args.only.split(",")) if args.only else {"REAL2D", "ABUS", "SIM"}
    gens = ([real_2d()] if "REAL2D" in want else []) + ([abus_lesions()] if "ABUS" in want else []) + \
           ([sim_lesions()] if "SIM" in want else [])
    rows = []
    for gen in gens:
        for ds, fid, g, m, px, info in gen:
            tm = imstats.tissue_mask(g)
            for les in components(m):
                try:
                    rows.append(dict(dataset=ds, file=fid, **info, **features(g, les, px, tm)))
                except Exception as e:  # keep going
                    print(ds, fid, repr(e), flush=True)
            if len(rows) % 200 == 0:
                print(ds, len(rows), flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d = pd.DataFrame(rows)
    d.to_csv(out / "lesions.csv", index=False)
    num = [c for c in d.columns if np.issubdtype(d[c].dtype, np.number)]
    t = d.groupby(["dataset", "pathology"])[num].median().round(3)
    t.insert(0, "n", d.groupby(["dataset", "pathology"]).size())
    t.to_csv(out / "summary.csv")
    print(t.T.to_string())


if __name__ == "__main__":
    main()
