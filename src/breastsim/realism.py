"""Realism benchmark: features of real and simulated breast US images and their distances.

A simulated set is compared with every public dataset feature by feature. The yardstick is the
spread between the real datasets themselves (different hospitals, scanners and protocols): for
each feature, score = W1(sim, real_k) / median W1(real_i, real_j), with W1 the 1-Wasserstein
distance of the per-image (or per-lesion) values. A score <= 1 means the simulation differs from a
real dataset no more than real datasets differ from each other.

Samples carry the gray image, the lesion mask, a validity mask (no burnt-in colour annotations)
and the pixel size when known (BrEaST, TDSC-ABUS challenge spacing, simulation). Features in
physical units are computed only when the pixel size is known (`SCALED`); texture features use
the image resampled to 0.1 mm/px.

Feature groups, chosen for segmentation / detection / classification:
- image: gray-level distribution, local mean echogenicity at the mm scale (tissue interleaving), depth trend, noise floor, texture spectrum and orientation,
  speckle size and contrast in structure-free patches (by depth third), column banding;
- anatomy: depth layering (bright / dark band transitions and thickness, lateral continuity),
  thin bright ridges (fascia, ligaments: density, length, orientation), hypoechoic distractors
  outside lesions (mass-like dark blobs);
- lesion: size, shape, orientation, depth, contrast, internal echotexture, margin sharpness,
  halo, posterior features, and how well the mask boundary sits on the image edge.
"""
import glob
import gzip
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.measure import perimeter
from skimage.morphology import convex_hull_image

from . import paths

TARGET_MM = 0.1                                   # texture features at 0.1 mm/px
ABUS_SPACING = (0.476, 0.073, 0.2)                # TDSC-ABUS 2023 (sweep, depth, lateral), mm (breastsim.presets)
# which part of the simulation a feature reflects: "imaging" features are used to tune the imaging
# chain; "content" features depend mostly on the phantom (anatomy, lesion morphology) and are only
# reported, so that the imaging chain is never tuned to stand in for missing anatomy
CONTENT = {"tex_aniso", "tex_coherence", "tex_horiz_frac", "tex_mottling", "tex_psd_mid", "banding",
           "layer_transitions_per_cm", "layer_thickness_mm", "layer_continuity", "ridge_density", "ridge_length_mm",
           "ridge_horiz_frac", "distractors_per_cm2", "eqdiam_mm", "depth_mm", "h_over_w", "angle_deg", "solidity",
           "circularity", "irregularity", "rel_area", "halo_sd", "lm_std_05", "lm_frac_05", "lm_std_15", "lm_frac_15",
           "lm_p90p10_15"}
SCALED = {"tex_psd_high", "tex_psd_mid", "tex_slope", "tex_aniso", "tex_horiz_frac", "tex_coherence", "tex_mottling",
          "spk_axial_mm_top", "spk_axial_mm_mid", "spk_axial_mm_deep", "spk_lateral_mm_top", "spk_lateral_mm_mid",
          "spk_lateral_mm_deep", "banding", "layer_transitions_per_cm", "layer_thickness_mm", "layer_continuity",
          "ridge_density", "ridge_length_mm", "ridge_horiz_frac", "distractors_per_cm2", "eqdiam_mm", "depth_mm"}


# ------------------------------------------------------------------ samples
def read_nrrd(path):
    """Minimal NRRD reader (raw / gzip encodings) -> (array in (z, y, x) order, header dict)."""
    with open(path, "rb") as f:
        hdr = {}
        while True:
            line = f.readline().decode().strip()
            if not line:
                break
            if ":" in line and not line.startswith("#"):
                k, v = line.split(":", 1)
                hdr[k.strip()] = v.strip()
        raw = f.read()
    if hdr.get("encoding") == "gzip":
        raw = gzip.decompress(raw)
    dt = {"unsigned char": np.uint8, "uchar": np.uint8, "short": np.int16,
          "unsigned short": np.uint16, "float": np.float32}[hdr["type"]]
    return np.frombuffer(raw, dt).reshape([int(s) for s in hdr["sizes"].split()][::-1]), hdr


def load_gray(f):
    """Gray image and mask of coloured annotation pixels (calipers, markers)."""
    a = np.asarray(Image.open(f).convert("RGB")).astype(np.float32)
    return a.mean(-1), (a.max(-1) - a.min(-1)) > 12


def _mask(files, shape):
    m = np.zeros(shape, bool)
    for mf in files:
        if Path(mf).exists():
            m |= np.asarray(Image.open(mf).convert("L")) > 127
    return m


def busi():
    for f in sorted(glob.glob(str(paths.REAL / "busi/extracted/Dataset_BUSI_with_GT/*/*).png"))):
        g, col = load_gray(f)
        yield dict(id=f, gray=g, lesion=_mask(glob.glob(f[:-4] + "_mask*.png"), g.shape), bad=col, px_mm=None,
                   info=dict(group=Path(f).parent.name))


def busbra():
    meta = pd.read_csv(paths.REAL / "busbra/BUSBRA/bus_data.csv").set_index("ID")
    for f in sorted(glob.glob(str(paths.REAL / "busbra/BUSBRA/Images/*.png"))):
        g, col = load_gray(f)
        mf = paths.REAL / "busbra/BUSBRA/Masks" / Path(f).name.replace("bus_", "mask_")
        yield dict(id=f, gray=g, lesion=_mask([mf], g.shape), bad=col, px_mm=None,
                   info=dict(group=meta.loc[Path(f).stem].Pathology, device=meta.loc[Path(f).stem].Device))


def breast():
    root = paths.REAL / "breast_lesions_usg"
    meta = pd.read_excel(next(root.glob("BrEaST-Lesions-USG-clinical-data-*.xlsx")))
    img_dir = root / "BrEaST-Lesions_USG-images_and_masks"
    for _, row in meta.iterrows():
        f = img_dir / row.Image_filename
        g, col = load_gray(f)
        les = _mask([img_dir / m.strip() for m in str(row.Mask_tumor_filename).split("&")], g.shape) \
            if isinstance(row.Mask_tumor_filename, str) else np.zeros(g.shape, bool)
        other = _mask(glob.glob(str(f)[:-4] + "_other*.png"), g.shape)
        yield dict(id=str(f), gray=g, lesion=les, bad=col | other, px_mm=float(row.Pixel_size) * 10,   # cm -> mm
                   info=dict(group=row.Classification, composition=row.Tissue_composition, shape=row.Shape,
                             margin=row.Margin, posterior_label=row.Posterior_features, echo=row.Echogenicity,
                             halo=row.Halo, calcifications=row.Calcifications))


def abus(n_vol=100, normal_slices=2):
    """TDSC-ABUS training volumes: the native B-mode plane through each lesion centroid plus
    `normal_slices` planes away from it, resampled to isotropic 0.2 mm (depth 0.073 mm -> 0.2 mm)."""
    labels = pd.read_csv(paths.REAL / "abus_tdsc/Train/labels.csv")
    mz = zipfile.ZipFile(next((paths.REAL / "abus_tdsc/Train/MASK").glob("*.zip")))
    dzips = [zipfile.ZipFile(z) for z in sorted((paths.REAL / "abus_tdsc/Train/DATA").glob("*.zip"))]
    sy = ABUS_SPACING[1] / ABUS_SPACING[2]
    with tempfile.TemporaryDirectory() as tmp:
        for _, row in labels.head(n_vol).iterrows():
            dn, mn = Path(row.data_path.replace("\\", "/")).name, Path(row.mask_path.replace("\\", "/")).name
            zf = next(z for z in dzips if dn in z.namelist())
            v, _ = read_nrrd(zf.extract(dn, tmp))
            m, _ = read_nrrd(mz.extract(mn, tmp))
            zz = np.nonzero((m > 0).any((1, 2)))[0]
            ks = [int(zz.mean())] if len(zz) else []
            free = [k for k in np.linspace(0.2, 0.8, 9).astype(float) * v.shape[0] if not len(zz) or
                    abs(k - zz.mean()) > 20]
            ks += [int(k) for k in free[:normal_slices]]
            for k in ks:
                g = ndi.zoom(v[k].astype(np.float32), (sy, 1.0), order=1)
                les = ndi.zoom((m[k] > 0).astype(np.float32), (sy, 1.0), order=0) > 0.5
                yield dict(id=f"{dn}#{k}", gray=g, lesion=les, bad=np.zeros(g.shape, bool), px_mm=ABUS_SPACING[2],
                           info=dict(group="malignant" if row.label == "M" else "benign"))
            Path(tmp, dn).unlink()
            Path(tmp, mn).unlink()


def sim(root):
    """Simulated set written by src/scripts/render_benchmark_set.py: img/, lesion/, meta.jsonl."""
    root = Path(root)
    for line in (root / "meta.jsonl").read_text().splitlines():
        m = json.loads(line)
        g = np.asarray(Image.open(root / "img" / m["file"]).convert("L")).astype(np.float32)
        les = np.asarray(Image.open(root / "lesion" / m["file"]).convert("L")) > 127
        yield dict(id=m["file"], gray=g, lesion=les, bad=np.zeros(g.shape, bool), px_mm=m["px_mm"],
                   info={k: v for k, v in m.items() if k not in ("file", "px_mm")})


READERS = {"BUSI": busi, "BUS-BRA": busbra, "BrEaST": breast, "TDSC-ABUS": abus}


# ------------------------------------------------------------------ helpers
def tissue_mask(g, bad=None):
    """Tissue pixels: the largest non-black region, eroded, minus saturated text and `bad`."""
    body = ndi.binary_opening(ndi.uniform_filter(g, 9) > 6, iterations=3)
    lab, n = ndi.label(body)
    if n:
        body = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    body = ndi.binary_erosion(body, iterations=6)
    sat = ndi.binary_dilation(g >= 250, iterations=2)
    return body & ~sat & (~bad if bad is not None else True)


def depth_profile(g, m, bins=10):
    rows = np.nonzero(m.any(1))[0]
    r0, r1 = rows.min(), rows.max()
    out = []
    for b in range(bins):
        a, z = r0 + (r1 - r0) * b // bins, r0 + (r1 - r0) * (b + 1) // bins
        s = g[a:z][m[a:z]]
        out.append(float(np.median(s)) if s.size else np.nan)
    return out


def _fwhm(ac):
    c = len(ac) // 2
    v = ac[c:] / ac[c]
    if not (v < 0.5).any():
        return np.nan
    k = np.argmax(v < 0.5)
    return 2 * (k - 1 + (v[k - 1] - 0.5) / max(v[k - 1] - v[k], 1e-6))


T = 64                                               # texture tile, px at 0.1 mm
_HANN = np.outer(np.hanning(T), np.hanning(T))
_fy, _fx = np.meshgrid(np.fft.fftfreq(T), np.fft.fftfreq(T), indexing="ij")
_FR = np.hypot(_fy, _fx)
_ANG = np.degrees(np.arctan2(np.abs(_fx), np.abs(_fy)))   # 0 = axial frequency (horizontal structures)


def _tiles(m, step=T // 2):
    H, W = m.shape
    for i in range(0, H - T + 1, step):
        for j in range(0, W - T + 1, step):
            if m[i:i + T, j:j + T].mean() > 0.98:
                yield i, j


def _structure(g, sigma=6):
    gs = ndi.gaussian_filter(g, 1.5)
    gx, gz = ndi.sobel(gs, 1), ndi.sobel(gs, 0)
    J = [ndi.gaussian_filter(a, sigma) for a in (gx * gx, gz * gz, gx * gz)]
    coh = np.sqrt((J[0] - J[1]) ** 2 + 4 * J[2] ** 2) / (J[0] + J[1] + 1e-6)
    theta = np.abs(0.5 * np.degrees(np.arctan2(2 * J[2], J[1] - J[0])))      # 0 = horizontal structure
    return coh, theta


# ------------------------------------------------------------------ features
def image_features(g, m, scaled):
    """Per-image features on tissue mask m (lesions excluded). g, m at 0.1 mm/px when `scaled`."""
    v = g[m]
    if v.size < 4000:
        return None
    prof = depth_profile(g, m)
    out = dict(gray_p05=np.percentile(v, 5), gray_p50=np.median(v), gray_p95=np.percentile(v, 95), gray_std=v.std(),
               gray_skew=((v - v.mean()) ** 3).mean() / v.std() ** 3, dark_frac=(v < 20).mean(),
               depth_trend=np.polyfit(np.linspace(0, 1, 10), np.nan_to_num(prof, nan=np.nanmean(prof)), 1)[0])
    dark = v[v <= np.percentile(v, 5)]
    out["noise_floor_cv"] = dark.std() / max(dark.mean(), 1.0)
    # local mean echogenicity at the mm scale (interleaving of tissues): gray level averaged over the
    # speckle with a masked Gaussian of sigma 0.5 / 1.5 mm (px assumed 0.1 mm when the pixel size is
    # unknown: BUSI / BUS-BRA are ~0.08-0.12 mm/px); spread as gray-level std, as a share of the
    # pixel std, and the 10-90 % range
    mf = m.astype(np.float32)
    for name, s in (("05", 5), ("15", 15)):
        den = ndi.gaussian_filter(mf, s)
        sel = m & (den > 0.9)
        if sel.sum() > 500:
            lm = (ndi.gaussian_filter(g * mf, s) / np.maximum(den, 1e-6))[sel]
            out[f"lm_std_{name}"] = lm.std()
            out[f"lm_frac_{name}"] = lm.std() / max(v.std(), 1e-6)
            if name == "15":
                out["lm_p90p10_15"] = np.percentile(lm, 90) - np.percentile(lm, 10)
    if not scaled:
        return out
    # texture: tile spectra, orientation, mottling
    psd = np.zeros((T, T))
    mot, n = [], 0
    for i, j in _tiles(m):
        t = g[i:i + T, j:j + T]
        F = np.abs(np.fft.fft2((t - t.mean()) * _HANN)) ** 2
        psd += F / F[_FR > 0].sum()
        mot.append(t.reshape(4, 16, 4, 16).mean((1, 3)).std() / max(t.std(), 1e-6))
        n += 1
    if n >= 3:
        psd /= n
        sel, mid = (_FR >= 1 / 32) & (_FR <= 1 / 4), (_FR >= 1 / 32) & (_FR < 1 / 8)
        coh, theta = _structure(g)
        w = coh[m]
        out.update(tex_psd_high=psd[(_FR >= 1 / 4)].sum(), tex_psd_mid=psd[(_FR >= 1 / 16) & (_FR < 1 / 4)].sum(),
                   tex_slope=np.polyfit(np.log(_FR[sel]), np.log(psd[sel]), 1)[0],
                   tex_aniso=psd[mid & (_ANG < 20)].mean() / psd[mid & (_ANG > 70)].mean(),
                   tex_coherence=w.mean(), tex_horiz_frac=(w * (theta[m] < 20)).sum() / w.sum(),
                   tex_mottling=np.mean(mot))
        # speckle in structure-free tiles (low orientation coherence, flat local mean), by depth third
        rows = np.nonzero(m.any(1))[0]
        r0, r1 = rows.min(), rows.max()
        lm = ndi.uniform_filter(g, 16)
        for name, (a, b) in zip(("top", "mid", "deep"), ((0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1))):
            fa, fl = [], []
            for i, j in _tiles(m):
                if not (r0 + a * (r1 - r0) <= i + T / 2 < r0 + b * (r1 - r0)):
                    continue
                if coh[i:i + T, j:j + T].mean() > 0.6 or lm[i:i + T, j:j + T].std() > 0.6 * g[i:i + T, j:j + T].std():
                    continue
                t = g[i:i + T, j:j + T]
                d = t - ndi.gaussian_filter(t, 8)
                ac = np.fft.fftshift(np.real(np.fft.ifft2(np.abs(np.fft.fft2(d)) ** 2)))
                fa.append(_fwhm(ac[:, T // 2]))
                fl.append(_fwhm(ac[T // 2, :]))
            if len(fa) >= 2:
                out[f"spk_axial_mm_{name}"] = np.nanmedian(fa) * TARGET_MM
                out[f"spk_lateral_mm_{name}"] = np.nanmedian(fl) * TARGET_MM
    # column banding (lower two thirds, > 5 mm trend removed)
    rows = np.nonzero(m.any(1))[0]
    sl = slice(rows.min() + (rows.max() - rows.min()) // 3, rows.max())
    gg, mm = g[sl], m[sl]
    ok = mm.sum(0) > 20
    if ok.sum() > 120:
        cm = np.where(ok, (gg * mm).sum(0) / np.maximum(mm.sum(0), 1), np.nan)
        idx = np.nonzero(ok)[0]
        c = cm[idx.min():idx.max() + 1]
        c = np.where(np.isnan(c), np.nanmean(c), c)
        k = int(5 / TARGET_MM)
        out["banding"] = np.std((c - np.convolve(c, np.ones(k) / k, "same"))[k:-k])
    # depth layering: bright / dark band transitions of the 1 mm-smoothed profile along each column
    sm = ndi.gaussian_filter(np.where(m, g, np.nan_to_num(np.median(v))), 10)
    trans, thick, cols = [], [], np.nonzero(m.sum(0) > 100)[0][::10]
    for j in cols:
        p = sm[m[:, j], j]
        s = np.sign(p - np.median(p))
        ch = np.nonzero(np.diff(s) != 0)[0]
        trans.append(len(ch) / (len(p) * TARGET_MM / 10))
        if len(ch) > 1:
            thick.extend(np.diff(ch) * TARGET_MM)
    if trans:
        out["layer_transitions_per_cm"] = np.median(trans)
        out["layer_thickness_mm"] = np.median(thick) if thick else np.nan
    lag = int(5 / TARGET_MM)
    pairs = [(j, j + lag) for j in cols if j + lag < g.shape[1] and m[:, j].sum() > 100 and m[:, j + lag].sum() > 100]
    if pairs:
        cc = []
        for a, b in pairs:
            both = m[:, a] & m[:, b]
            if both.sum() > 100:
                cc.append(np.corrcoef(sm[both, a], sm[both, b])[0, 1])
        out["layer_continuity"] = np.nanmedian(cc) if cc else np.nan
    # thin bright ridges (fascia, ligaments): Hessian ridge measure at 0.3 mm
    H = hessian_matrix(g, sigma=3, order="rc", use_gaussian_derivatives=True)
    l1 = hessian_matrix_eigvals(H)[1]                                    # most negative = bright ridge
    ridge = (-l1 > np.percentile(-l1[m], 97)) & m & (g > np.percentile(v, 70))
    lab, nr = ndi.label(ridge, np.ones((3, 3)))
    if nr:
        sizes = ndi.sum(ridge, lab, range(1, nr + 1))
        objs = ndi.find_objects(lab)
        ext = np.array([max(o[0].stop - o[0].start, o[1].stop - o[1].start) for o in objs]) * TARGET_MM
        hz = np.array([(o[1].stop - o[1].start) > 2 * (o[0].stop - o[0].start) for o in objs])
        big = sizes >= 10
        area_cm2 = m.sum() * TARGET_MM ** 2 / 100
        out.update(ridge_density=big.sum() / area_cm2, ridge_length_mm=np.median(ext[big]) if big.any() else np.nan,
                   ridge_horiz_frac=hz[big].mean() if big.any() else np.nan)
    # hypoechoic distractors: dark blobs of mass-like size outside lesions
    lm1 = ndi.gaussian_filter(np.where(m, g, np.median(v)), 10)
    mad = np.median(np.abs(v - np.median(v)))
    blob = (lm1 < np.median(v) - 1.5 * mad) & m
    lab, nb = ndi.label(blob)
    if nb:
        areas = ndi.sum(blob, lab, range(1, nb + 1)) * TARGET_MM ** 2
        out["distractors_per_cm2"] = ((areas >= 4) & (areas <= 400)).sum() / (m.sum() * TARGET_MM ** 2 / 100)
    return {k: float(x) for k, x in out.items()}


def lesion_features(g, les, valid, px_mm):
    """Per-lesion features (mask component `les`) on gray image g."""
    rows, cols = np.nonzero(les)
    area = float(les.sum())
    d = 2 * np.sqrt(area / np.pi)
    h, w = rows.max() - rows.min() + 1, cols.max() - cols.min() + 1
    cy, cx = rows.mean(), cols.mean()
    mu20, mu02, mu11 = ((cols - cx) ** 2).mean(), ((rows - cy) ** 2).mean(), ((cols - cx) * (rows - cy)).mean()
    edge = les & ~ndi.binary_erosion(les)
    er, ec = np.nonzero(edge)
    rad = np.hypot(er - cy, ec - cx)
    out = dict(h_over_w=h / w, angle_deg=abs(0.5 * np.degrees(np.arctan2(2 * mu11, mu20 - mu02))),
               solidity=area / convex_hull_image(les).sum(), circularity=4 * np.pi * area / max(perimeter(les), 1.0) ** 2,
               irregularity=rad.std() / rad.mean(), rel_area=area / max(valid.sum(), 1))
    if px_mm:
        out["eqdiam_mm"] = d * px_mm
        top = np.nonzero(valid[:, int(cx)])[0]
        if len(top):
            out["depth_mm"] = (cy - top.min()) * px_mm
    near = ndi.binary_dilation(les, iterations=max(1, int(0.15 * d)))
    far = ndi.binary_dilation(les, iterations=max(2, int(0.5 * d)))
    ring, rim = far & ~near & valid, near & ~les & valid
    if ring.sum() > 30:
        gin, gr = g[les], g[ring]
        sd = max(gr.std(), 1e-3)
        out.update(contrast_sd=(np.median(gin) - np.median(gr)) / sd, inner_heterogeneity=gin.std() / sd)
        if rim.sum() > 10:
            out["halo_sd"] = (np.median(g[rim]) - np.median(gr)) / sd
        gm = np.hypot(*np.gradient(ndi.gaussian_filter(g, 1.0)))
        out["edge_sharpness"] = gm[edge].mean() / max(gm[ring].mean(), 1e-3)
        # mask boundary on the image edge: gradient on the boundary vs 3 px inside / outside
        inner = ndi.binary_erosion(les, iterations=3)
        shell_in = inner & ~ndi.binary_erosion(inner)
        outer = ndi.binary_dilation(les, iterations=3)
        shell_out = outer & ~ndi.binary_erosion(outer) & valid
        if shell_in.sum() > 5 and shell_out.sum() > 5:
            out["mask_edge_alignment"] = gm[edge].mean() / max(0.5 * (gm[shell_in].mean() + gm[shell_out].mean()), 1e-3)
        b0, b1 = rows.max() + 1, min(g.shape[0], rows.max() + 1 + max(3, int(0.5 * d)))
        c0, c1 = cols.min() + int(0.2 * w), cols.max() + 1 - int(0.2 * w)
        s0, s1 = max(0, cols.min() - int(0.5 * d)), min(g.shape[1], cols.max() + 1 + int(0.5 * d))
        if b1 - b0 >= 3 and c1 > c0:
            below = g[b0:b1, c0:c1][valid[b0:b1, c0:c1]]
            side = np.zeros(g.shape, bool)
            side[b0:b1, s0:cols.min()] = True
            side[b0:b1, cols.max() + 1:s1] = True
            side = g[side & valid & ~far]
            if below.size > 20 and side.size > 20:
                out["posterior_sd"] = (np.median(below) - np.median(side)) / sd
    return {k: float(x) for k, x in out.items()}


# lesion conspicuity: interior contrast against the surrounding ring plus the boundary cue (gradient on the
# mask edge above the ring's); real lesions are all visible (annotated by radiologists): the lower bound is
# the 2nd percentile over BrEaST + BUSI lesions (src/scripts/lesion_visibility.py, 2026-10-27)
CONSPICUITY_MIN = 0.25


def conspicuity(t):
    """Conspicuity of lesion feature rows (dict or DataFrame with contrast_sd, edge_sharpness)."""
    return np.abs(t["contrast_sd"]) + np.maximum(t["edge_sharpness"] - 1.0, 0)


def extract(samples, dataset, log=print):
    """Image and lesion feature tables of an iterable of samples."""
    imgs, lesions = [], []
    for k, s in enumerate(samples):
        g, les, px = s["gray"], s["lesion"], s["px_mm"]
        valid = tissue_mask(g, s["bad"])
        # lesions: the largest mask component and any other of at least a quarter of its area (2D
        # sections of spiculated masses leave detached spicule fragments)
        lab, n = ndi.label(les)
        sizes = ndi.sum(les, lab, range(1, n + 1)) if n else []
        for c in range(1, n + 1):
            comp = lab == c
            if sizes[c - 1] >= max(50, 0.25 * max(sizes)):
                try:
                    lesions.append(dict(dataset=dataset, id=s["id"], **s["info"], **lesion_features(g, comp, valid, px)))
                except Exception as e:  # degenerate masks
                    log(f"  lesion {s['id']}: {e!r}")
        m = valid & ~ndi.binary_dilation(les, iterations=15)
        scaled = px is not None
        if scaled and abs(px / TARGET_MM - 1) > 0.02:
            z = px / TARGET_MM
            g, m = ndi.zoom(g, z, order=1), ndi.zoom(m.astype(np.float32), z, order=0) > 0.5
        f = image_features(g, m, scaled)
        if f:
            imgs.append(dict(dataset=dataset, id=s["id"], **s["info"], **f))
        if (k + 1) % 100 == 0:
            log(f"  {dataset}: {k + 1} images")
    return pd.DataFrame(imgs), pd.DataFrame(lesions)


# ------------------------------------------------------------------ scoring
def w1(a, b):
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    q = np.linspace(0, 1, 201)
    return float(np.abs(np.quantile(a, q) - np.quantile(b, q)).mean())


def score(real, sim, kind, min_n=10):
    """Per-feature realism scores of `sim` against each real dataset in `real` (dict name -> table)."""
    feats = sorted({c for t in list(real.values()) + [sim] for c in t.columns
                    if pd.api.types.is_numeric_dtype(t[c])})
    rows = []
    for f in feats:
        have = {k: t[f].dropna() for k, t in real.items() if f in t and t[f].notna().sum() >= min_n}
        s = sim[f].dropna() if f in sim else pd.Series(dtype=float)
        if len(have) < 2 or len(s) < min_n:
            continue
        names = sorted(have)
        yard = np.median([w1(have[a], have[b]) for i, a in enumerate(names) for b in names[i + 1:]])
        row = dict(group=kind, part="content" if f in CONTENT else "imaging", feature=f, sim_median=s.median(),
                   yardstick=yard)
        for k in names:
            row[f"real_median:{k}"] = have[k].median()
            row[f"score:{k}"] = w1(s, have[k]) / max(yard, 1e-9)
        row["score_mean"] = np.mean([row[f"score:{k}"] for k in names])
        row["score_min"] = np.min([row[f"score:{k}"] for k in names])
        rows.append(row)
    return pd.DataFrame(rows)
