#!/usr/bin/env python
"""Fit a scan preset's display chain (TGC slope, gain, dynamic range, gray gamma) to a real dataset.

    python src/scripts/calibrate_display.py --preset breast --dataset BrEaST
    python src/scripts/calibrate_display.py --preset busi --dataset BUSI
    python src/scripts/calibrate_display.py --preset busbra --dataset BUS-BRA --no-profile   # lesion-centred crops

1. Renders preview frames over all four BI-RADS densities (oat64 and USCT phantoms, lesion and
   normal-tissue views, random positions / rotations) with TGC off and keeps the linear echo
   intensity (cached in outputs/calib/<preset>/frames.npz).
2. Takes the real targets on tissue pixels (lesions dilated and excluded, as in
   breastsim.imstats): the gray-level distribution (every image weighted equally) and the
   median depth profile (10 bins from the skin to the bottom of the tissue region).
3. Minimises W1(gray histograms) + mean |depth-profile difference| (both in gray levels) over
   (tgc_db_cm_mhz, gain_db, dynamic_range_db, gray_gamma), Nelder-Mead from a grid of starts.

The physics (attenuation, speckle, PSF) is untouched: only the scanner-side display chain is
fitted, as a sonographer sets TGC / gain / DR / map for these datasets.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage as ndi
from scipy.optimize import minimize

from breastsim import imstats, paths, phantom, realdata, tissue
from breastsim.probe import PROBES, ProbePose
from breastsim.realtime.renderer import ImagingParams, PreviewRenderer, _conv_along, _gauss_kernel

BINS = np.arange(257) - 0.5


def real_targets(dataset, limit=None):
    hists, profs = [], []
    for k, (_, g, bad, _) in enumerate(realdata.READERS[dataset]()):
        if limit and k >= limit:
            break
        m = imstats.tissue_mask(g, bad)
        if m.sum() < 4000:
            continue
        h = np.histogram(g[m], BINS)[0]
        hists.append(h / h.sum())
        profs.append(imstats.depth_profile(g, m))
    return np.mean(hists, 0), np.nanmedian(np.array(profs), 0), len(hists)


def render_frames(preset, n_oat=4, poses_oat=3, poses_usct=6, seed=0):
    """Frames with TGC off: (intensity (H, W), tissue mask, alpha power, ref) per frame."""
    rng = np.random.default_rng(seed)
    r = PreviewRenderer()
    p = ImagingParams.from_preset(preset, overlay="none", tgc_db_cm_mhz=0.0)
    frames = []
    for cls in "ABCD":
        oat = sorted(i.id for i in phantom.list_phantoms() if i.source == "oat64" and i.id.endswith("l")
                     and i.id.split("/")[1][0] == cls and phantom.is_cached(i.id))
        pids = [(pid, poses_oat) for pid in rng.choice(oat, min(n_oat, len(oat)), replace=False)]
        pids.append((f"usct/{cls}", poses_usct))
        for pid, n in pids:
            vol = phantom.load(pid, log=lambda *a: None)
            r.set_volume(vol)
            iu, iv = np.nonzero(np.isfinite(vol.surface))
            les = vol.meta.get("lesions", [])
            for k in range(n):
                if les and k % 2 == 0:              # half of the views through a lesion
                    u, v = np.array(les[rng.integers(len(les))]["center_mm"][1:]) + rng.normal(0, 3, 2)
                else:
                    j = rng.integers(len(iu))
                    u, v = iu[j] * vol.voxel_mm, iv[j] * vol.voxel_mm
                pose = ProbePose(u=float(u), v=float(v), rotation=float(rng.uniform(-180, 180)))
                _, _, lab = r.render(pose, p)
                inten = r.last_intensity.cpu().numpy()
                m = (lab != tissue.BACKGROUND) & ~ndi.binary_dilation(np.isin(lab, tissue.LESIONS), iterations=15)
                m = m & ndi.binary_erosion(np.ones_like(m), iterations=6, border_value=0)
                if m.sum() > 4000:
                    frames.append(dict(inten=inten, mask=m, power=vol.alpha_power, ref=r.last_ref, pid=pid, cls=cls))
        print(f"  {cls}: {len(frames)} frames", flush=True)
    return frames, r


class SimSet:
    """Pre-smoothed frames; display(theta) applies TGC, log compression and the gray map."""

    def __init__(self, frames, r, preset):
        p = ImagingParams.from_preset(preset)
        self.r, self.spec, self.p = r, PROBES[p.probe], p
        ks = _gauss_kernel(p.speckle_reduction, r.device) if p.speckle_reduction > 0 else None
        self.items = []
        for f in frames:
            a = torch.from_numpy(f["inten"]).to(r.device).sqrt()
            if ks is not None:                       # smoothing commutes with the per-row TGC gain
                a = _conv_along(_conv_along(a, ks, 0), ks, 1)
            H = a.shape[0]
            z = (torch.arange(H, device=r.device, dtype=torch.float32) + 0.5) * (p.depth_mm / H)
            db0 = 10 * torch.log10(a ** 2 / f["ref"] + 1e-12)
            self.items.append((db0, z, f["power"], torch.from_numpy(f["mask"]).to(r.device)))

    def gray(self, theta, item):
        tgc, gain, dr, gamma = theta
        db0, z, power, m = item
        g_tgc = 10 * torch.log10(self.r.tgc_gain(self.spec, power, z, tgc))
        lev = torch.clamp((db0 + g_tgc[:, None] + gain) / dr + 1, 0, 1) ** gamma
        return lev * 255, m

    def stats(self, theta):
        hists, profs = [], []
        for it in self.items:
            g, m = self.gray(theta, it)
            g, m = g.cpu().numpy(), m.cpu().numpy()
            h = np.histogram(g[m], BINS)[0]
            hists.append(h / h.sum())
            profs.append(imstats.depth_profile(g, m))
        return np.mean(hists, 0), np.nanmedian(np.array(profs), 0)


def w1(h1, h2):
    return float(np.abs(np.cumsum(h1) - np.cumsum(h2)).sum())     # gray levels


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="breast")
    ap.add_argument("--dataset", default="BrEaST", choices=list(realdata.READERS))
    ap.add_argument("--no-profile", action="store_true", help="fit the histogram only (cropped datasets)")
    ap.add_argument("--rerender", action="store_true")
    args = ap.parse_args()
    out = paths.OUTPUTS / "calib" / args.preset
    out.mkdir(parents=True, exist_ok=True)

    print(f"[calib] real targets: {args.dataset}", flush=True)
    h_real, p_real, n_real = real_targets(args.dataset)
    cache = out / "frames.npz"
    if cache.exists() and not args.rerender:
        z = np.load(cache, allow_pickle=True)
        frames, r = list(z["frames"]), PreviewRenderer()
    else:
        print(f"[calib] rendering frames with preset {args.preset}", flush=True)
        frames, r = render_frames(args.preset)
        np.savez(cache, frames=np.array(frames, dtype=object))
    sims = SimSet(frames, r, args.preset)
    use_prof = not args.no_profile

    def loss(theta):
        tgc, gain, dr, gamma = theta
        if not (0 <= tgc <= 1.5 and -60 <= gain <= 20 and 20 <= dr <= 100 and 0.3 <= gamma <= 4):
            return 1e6
        h, prof = sims.stats(theta)
        return w1(h, h_real) + (float(np.nanmean(np.abs(prof - p_real))) if use_prof else 0.0)

    p0 = ImagingParams.from_preset(args.preset)
    theta0 = (p0.tgc_db_cm_mhz, p0.gain_db, p0.dynamic_range_db, p0.gray_gamma)
    starts = sorted(itertools.product((0.2, 0.5, 0.8), (-25, -10), (45, 65), (1.0, 1.8)), key=loss)[:4]
    best = min((minimize(loss, s, method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-3, maxiter=400))
                for s in starts), key=lambda res: res.fun)
    theta = best.x
    h0, pr0 = sims.stats(theta0)
    h1, pr1 = sims.stats(theta)
    mean = lambda h: float((h * np.arange(256)).sum())
    report = dict(
        preset=args.preset, dataset=args.dataset, n_real=n_real, n_frames=len(frames), fit_profile=use_prof,
        before=dict(zip(("tgc_db_cm_mhz", "gain_db", "dynamic_range_db", "gray_gamma"), map(float, theta0)),
                    loss=loss(theta0), w1=w1(h0, h_real), mean_gray=mean(h0), profile=pr0.round(1).tolist()),
        after=dict(zip(("tgc_db_cm_mhz", "gain_db", "dynamic_range_db", "gray_gamma"), map(float, theta)),
                   loss=float(best.fun), w1=w1(h1, h_real), mean_gray=mean(h1), profile=pr1.round(1).tolist()),
        real=dict(mean_gray=mean(h_real), profile=p_real.round(1).tolist()))
    (out / f"fit_{args.dataset}.json").write_text(json.dumps(report, indent=1))
    np.savez(out / f"hist_{args.dataset}.npz", real=h_real, before=h0, after=h1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
