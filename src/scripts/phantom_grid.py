#!/usr/bin/env python
"""Phantom-content diagnostics: image grids, label statistics and breast thickness.

    python src/scripts/phantom_grid.py grid outputs/realism/sim/vh1 [--n 16]   # 4x4 grid, lesion outlines, label stats
    python src/scripts/phantom_grid.py real BrEaST [--n 16]                       # same grid of real images
    python src/scripts/phantom_grid.py geom victre/hetero_1 [...]                 # skin-to-chest-wall thickness
    python src/scripts/phantom_grid.py crops outputs/realism/sim/x --out c.png     # lesion crops per echo type, real vs sim

`grid` reads a set written by render_benchmark_set.py (img/, lesion/, label/, meta.jsonl) and writes
<set>/grid.png and <set>/label_stats.json. Label statistics per frame (simulation only, from the
tissue labels under the image):
  muscle_visible   pectoralis (label 40) present in the frame
  muscle_depth_mm  median depth of the first muscle pixel per column (frames with muscle)
  subq_fat_mm      median depth of the first glandular pixel per column (premammary fat + skin)
  lig_horiz_frac   share of ligament (label 88) orientation energy within 20 deg of horizontal
  bnd_horiz_frac   same for all fat / gland / ligament boundaries
`geom` measures, over the breast footprint (u, v), skin height minus the top of the muscle layer
(the thickness a probe pressing along -A sees) and prints percentiles.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from breastsim import paths, tissue


def _orient_frac(mask, sigma=2.0):
    """Share of the structure-tensor energy of `mask` oriented within 20 deg of horizontal."""
    f = ndi.gaussian_filter(mask.astype(np.float32), 1.0)
    gx, gz = ndi.sobel(f, 1), ndi.sobel(f, 0)
    J = [ndi.gaussian_filter(a, sigma) for a in (gx * gx, gz * gz, gx * gz)]
    energy = np.sqrt((J[0] - J[1]) ** 2 + 4 * J[2] ** 2)
    theta = np.abs(0.5 * np.degrees(np.arctan2(2 * J[2], J[1] - J[0])))      # 0 = horizontal structure
    sel = mask if mask.any() else np.ones_like(mask)
    e = energy[ndi.binary_dilation(sel, iterations=2)]
    t = theta[ndi.binary_dilation(sel, iterations=2)]
    return float((e * (t < 20)).sum() / max(e.sum(), 1e-9))


def label_stats(lab, px):
    out = {}
    mus = lab == tissue.MUSCLE
    out["muscle_visible"] = float(mus.any(0).mean() > 0.2)
    if mus.any(0).mean() > 0.2:
        out["muscle_depth_mm"] = float(np.median(np.argmax(mus, 0)[mus.any(0)]) * px)
    gl = np.isin(lab, (tissue.GLAND, tissue.TDLU))
    if gl.any(0).mean() > 0.2:
        out["subq_fat_mm"] = float(np.median(np.argmax(gl, 0)[gl.any(0)]) * px)
    lig = lab == tissue.LIGAMENT
    if lig.sum() > 50:
        out["lig_horiz_frac"] = _orient_frac(lig)
    b = np.isin(lab, (tissue.GLAND, tissue.TDLU, tissue.LIGAMENT))
    if b.any() and (~b).any():
        out["bnd_horiz_frac"] = _orient_frac(b)
    for name, labs in (("fat", (tissue.FAT,)), ("gland", (tissue.GLAND, tissue.TDLU)), ("ligament", (tissue.LIGAMENT,)),
                       ("muscle", (tissue.MUSCLE,))):
        out[f"frac_{name}"] = float(np.isin(lab, labs).mean())
    return out


def montage(tiles, path, cols=4, w=300, captions=None):
    hs = [int(round(t.shape[0] * w / t.shape[1])) for t in tiles]
    h = max(hs)
    rows = (len(tiles) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * (w + 4), rows * (h + 4)), (40, 40, 40))
    d = ImageDraw.Draw(canvas)
    for k, t in enumerate(tiles):
        im = Image.fromarray(t).resize((w, hs[k]), Image.BILINEAR)
        x, y = (k % cols) * (w + 4), (k // cols) * (h + 4)
        canvas.paste(im, (x, y))
        if captions:
            d.text((x + 3, y + 2), captions[k], fill=(255, 255, 0))
    canvas.save(path)


def outline(gray, les):
    rgb = np.repeat(gray[..., None], 3, -1).astype(np.uint8)
    if les.any():
        edge = les & ~ndi.binary_erosion(les, iterations=2)
        rgb[edge] = (60, 255, 60)
    return rgb


def cmd_grid(a):
    root = Path(a.set)
    meta = [json.loads(l) for l in (root / "meta.jsonl").read_text().splitlines()]
    rng = np.random.default_rng(0)
    pick = sorted(rng.choice(len(meta), min(a.n, len(meta)), replace=False)) if len(meta) > a.n else range(len(meta))
    tiles, caps, stats = [], [], []
    for k, m in enumerate(meta):
        lab = np.asarray(Image.open(root / "label" / m["file"]))
        s = label_stats(lab, m["px_mm"])
        stats.append({"file": m["file"], "phantom": m["phantom"], "view": m["view"], **s})
        if k in pick:
            g = np.asarray(Image.open(root / "img" / m["file"]).convert("L"))
            les = np.asarray(Image.open(root / "lesion" / m["file"]).convert("L")) > 127
            tiles.append(outline(g, les))
            caps.append(f"{m['phantom'].split('/')[-1]} {m['view'][0]} r{m['rotation']:.0f}")
    montage(tiles, root / "grid.png", captions=caps)
    keys = sorted({k for s in stats for k in s if k not in ("file", "phantom", "view")})
    summ = {k: round(float(np.nanmedian([s.get(k, np.nan) for s in stats])), 3) for k in keys}
    summ["muscle_visible"] = round(float(np.mean([s["muscle_visible"] for s in stats])), 3)
    (root / "label_stats.json").write_text(json.dumps({"summary": summ, "frames": stats}, indent=1))
    print(json.dumps({"set": root.name, "n": len(stats), **summ}))


def cmd_real(a):
    from breastsim import realism
    tiles, caps = [], []
    rng = np.random.default_rng(a.seed)
    samples = list(realism.READERS[a.dataset]())
    for k in rng.choice(len(samples), a.n, replace=False):
        s = samples[k]
        tiles.append(outline(np.clip(s["gray"], 0, 255), s["lesion"]))
        caps.append(Path(s["id"]).name[:24])
    out = paths.OUTPUTS / "phantom_realism" / f"real_{a.dataset}_{a.seed}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    montage(tiles, out, captions=caps)
    print(out)


def _crop(g, les, size=150, pad=0.5):
    rows, cols = np.nonzero(les)
    h, w = np.ptp(rows) + 1, np.ptp(cols) + 1
    side = int(max(h, w) * (1 + 2 * pad))
    cy, cx = int(rows.mean()), int(cols.mean())
    y0, x0 = max(0, cy - side // 2), max(0, cx - side // 2)
    sub = outline(np.clip(g, 0, 255)[y0:y0 + side, x0:x0 + side], les[y0:y0 + side, x0:x0 + side])
    return np.asarray(Image.fromarray(sub).resize((size, size), Image.BILINEAR))


def cmd_crops(a):
    """Lesion crops per echo pattern: real BrEaST (top row of each type) vs simulated sets (bottom row)."""
    from breastsim import realism
    types = ["anechoic", "hypoechoic", "isoechoic", "hyperechoic", "heterogeneous", "complex cystic/solid"]
    rng = np.random.default_rng(0)
    real = {t: [] for t in types}
    for s_ in realism.breast():
        t = s_["info"].get("echo")
        if t in real and s_["lesion"].any():
            real[t].append(_crop(s_["gray"], s_["lesion"]))
    sim = {t: [] for t in types}
    for root in map(Path, a.sets):
        for l in (root / "meta.jsonl").read_text().splitlines():
            m = json.loads(l)
            t = m.get("lesion_echo")
            les = np.asarray(Image.open(root / "lesion" / m["file"]).convert("L")) > 127
            if t in sim and les.sum() > 50:
                g = np.asarray(Image.open(root / "img" / m["file"]).convert("L")).astype(np.float32)
                sim[t].append(_crop(g, les))
    tiles, caps = [], []
    for t in types:
        for src, d in (("real", real[t]), ("sim", sim[t])):
            pick = [d[i] for i in rng.permutation(len(d))[:a.n]]
            for k in range(a.n):
                tiles.append(pick[k] if k < len(pick) else np.zeros((150, 150, 3), np.uint8))
                caps.append(f"{src} {t[:12]}" if k == 0 else "")
    montage(tiles, Path(a.out), cols=a.n, w=150, captions=caps)
    print(a.out, {t: (len(real[t]), len(sim[t])) for t in types})


def cmd_geom(a):
    from breastsim import phantom
    for pid in a.ids:
        vol = phantom.load(pid, log=lambda *x: None, chest_wall=False)
        lab, vox = vol.label, vol.voxel_mm
        mus = lab == tissue.MUSCLE
        any_m = mus.any(0)
        top_m = np.where(any_m, (lab.shape[0] - 1 - np.argmax(mus[::-1], 0) + 0.5) * vox, 0.0)
        surf = vol.surface
        ok = np.isfinite(surf)
        thick = (surf - top_m)[ok]
        lig = lab == tissue.LIGAMENT
        q = np.percentile(thick, [10, 50, 90])
        r = {"id": pid, "footprint_cm2": round(ok.sum() * vox ** 2 / 100, 1), "thick_p10_p50_p90_mm": q.round(1).tolist(),
             "frac_thick_lt_40mm": round(float((thick < 40).mean()), 3), "max_mm": round(float(thick.max()), 1),
             "lig_frac": round(float(lig.sum() / max((lab > 0).sum(), 1)), 4), "extent_mm": [round(e, 1) for e in vol.extent_mm]}
        print(json.dumps(r))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("grid")
    g.add_argument("set")
    g.add_argument("--n", type=int, default=16)
    r = sub.add_parser("real")
    r.add_argument("dataset")
    r.add_argument("--n", type=int, default=16)
    r.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("crops")
    c.add_argument("sets", nargs="+")
    c.add_argument("--out", required=True)
    c.add_argument("--n", type=int, default=6)
    m = sub.add_parser("geom")
    m.add_argument("ids", nargs="+")
    a = ap.parse_args()
    {"grid": cmd_grid, "real": cmd_real, "geom": cmd_geom, "crops": cmd_crops}[a.cmd](a)


if __name__ == "__main__":
    main()
