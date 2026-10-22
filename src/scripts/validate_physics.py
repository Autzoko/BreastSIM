#!/usr/bin/env python
"""Quantitative checks of the preview imaging model, with PyMUST as reference.

Tests (homogeneous / simple numerical phantoms):
1. speckle statistics: envelope SNR (Rayleigh: 1.91) in fully developed speckle
2. speckle size: lateral / axial -6 dB widths of the envelope autocorrelation,
   preview vs PyMUST (SIMUS + DAS) for the same probe (L11-5v)
3. attenuation: slope of mean echo level vs depth = -2 * alpha * f^y * 8.686 dB/m
4. anechoic cyst: interior contrast and posterior enhancement

    python src/scripts/validate_physics.py --out outputs/validation      # add --no-pymust to skip the reference
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from breastsim import phantom, tissue
from breastsim.probe import PROBES, ProbePose
from breastsim.realtime.renderer import ImagingParams, PreviewRenderer

VOX = 0.2


def block_volume(alpha=0.0, cyst_radius=0.0, size=(60, 70, 70)):
    """Homogeneous glandular block (a, u, v mm), optional anechoic cyst 20 mm below the top."""
    shape = tuple(int(s / VOX) for s in size)
    label = np.full(shape, tissue.GLAND, np.uint8)
    if cyst_radius:
        a, u, v = np.meshgrid(*[np.arange(n) * VOX for n in shape], indexing="ij")
        c = (size[0] - 20, size[1] / 2, size[2] / 2)
        label[(a - c[0]) ** 2 + (u - c[1]) ** 2 + (v - c[2]) ** 2 < cyst_radius ** 2] = tissue.CYST
    z = np.full(shape, 1.5, np.float16)
    al = np.where(label == tissue.CYST, 0.0, alpha).astype(np.float16)
    vol = phantom.Volume("validation/block", label, z, al, VOX, 1.0,
                         np.full(shape[1:], size[0], np.float32), np.zeros(shape[1:], bool), {})
    return vol


def stats(env, px_x, px_z):
    """Envelope SNR and -6 dB (0.5 of the normalised autocorrelation) speckle widths in mm."""
    snr = env.mean() / env.std()
    i = env ** 2 - (env ** 2).mean()
    f = np.fft.fft2(i, s=(2 * i.shape[0], 2 * i.shape[1]))
    ac = np.fft.ifft2(np.abs(f) ** 2).real
    ac /= ac[0, 0]

    def width(profile, px):
        k = np.argmax(profile < 0.5)
        if k == 0:
            return float("nan")
        # linear interpolation of the half-maximum crossing, two-sided width
        x = (k - 1) + (profile[k - 1] - 0.5) / (profile[k - 1] - profile[k])
        return float(2 * x * px)
    return {"envelope_snr": round(float(snr), 3),
            "speckle_lateral_mm": round(width(ac[0, :i.shape[1]], px_x), 3),
            "speckle_axial_mm": round(width(ac[:i.shape[0], 0], px_z), 3)}


def preview(vol, depth=30.0, **kw):
    r = PreviewRenderer()
    r.set_volume(vol)
    p = ImagingParams.from_preset("l11_5v", overlay="none", speckle_reduction=0, specular=0, conform=False,
                                  tgc_db_cm_mhz=0, shadow_blur_mm=0, depth_mm=depth, **kw)
    _, meta, labels = r.render(ProbePose(u=vol.label.shape[1] * VOX / 2, v=vol.label.shape[2] * VOX / 2), p)
    return np.sqrt(r.last_intensity.cpu().numpy()), meta, labels


def pymust_reference(region, density=40, angles=7):
    """Compounded plane-wave SIMUS + DAS on random 2D scatterers (L11-5v)."""
    import pymust
    rng = np.random.default_rng(0)
    param = pymust.getparam("L11-5v")
    param.fnumber = 2.0
    (x0, x1), (z0, z1) = region
    n = int(density * (x1 - x0 + 4) * (z1 - z0 + 4))           # scatterers per mm^2 x area (+margin)
    xs = rng.uniform(x0 - 2, x1 + 2, n) * 1e-3
    zs = rng.uniform(z0 - 2, z1 + 2, n) * 1e-3
    rc = rng.standard_normal(n)
    xi, zi = np.meshgrid(np.arange(x0, x1, 0.05) * 1e-3, np.arange(z0, z1, 0.025) * 1e-3)
    opt = pymust.utils.Options()
    opt.ParPool = True
    iq_sum = 0
    for a in np.deg2rad(np.linspace(-6, 6, angles)):
        tx = pymust.txdelay(param, a)
        rf, _ = pymust.simus(xs[None], np.zeros((1, n)), zs[None], rc[None], tx, param, opt)
        iq = pymust.rf2iq(rf, param)
        m = pymust.dasmtx(1j * np.ones(iq.shape), xi, zi, tx, param)
        iq_sum = iq_sum + (m @ iq.flatten(order="F")).reshape(xi.shape, order="F")
    return np.abs(iq_sum), n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="outputs/validation")
    ap.add_argument("--no-pymust", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = {}
    spec = PROBES["L11-5v"]

    # 1-2. speckle statistics and size, window 15-25 mm deep, central 16 mm
    env, meta, _ = preview(block_volume(), px_x_mm=0.05, px_z_mm=0.025)
    px, pz = meta["px_x_mm"], meta["px_z_mm"]
    w = env[int(15 / pz):int(25 / pz), int((meta["width_mm"] / 2 - 8) / px):int((meta["width_mm"] / 2 + 8) / px)]
    rep["preview_speckle"] = stats(w, px, pz)
    lam = spec.wavelength * 1e3
    rep["theory"] = {"rayleigh_snr": 1.91, "lambda_mm": round(lam, 3),
                     "note": "speckle cell ~ PSF size: lateral ~ lambda*F#, axial ~ pulse length / 2"}
    if not args.no_pymust:
        t0 = time.time()
        env_ref, n = pymust_reference(((-8, 8), (15, 25)))
        rep["pymust_speckle"] = {**stats(env_ref, 0.05, 0.025), "scatterers": n, "seconds": round(time.time() - t0, 1)}

    # 3. attenuation slope: alpha = 0.5 dB/cm/MHz (y = 1) in Np/m/MHz
    alpha_np = 0.5 * 100 / 8.686
    env_a, meta_a, _ = preview(block_volume(alpha=alpha_np), depth=40)
    pz = meta_a["px_z_mm"]
    prof = 10 * np.log10((env_a ** 2).mean(axis=1))
    zz = (np.arange(len(prof)) + 0.5) * pz
    sel = (zz > 5) & (zz < 35)
    slope = np.polyfit(zz[sel], prof[sel], 1)[0] * 1e3              # dB/m
    expect = -2 * alpha_np * spec.fc / 1e6 * 8.686
    rep["attenuation"] = {"measured_db_per_cm": round(slope / 100, 2), "expected_db_per_cm": round(expect / 100, 2)}

    # 4. anechoic cyst (r = 4 mm) 20 mm deep in attenuating gland: contrast and posterior enhancement
    env_c, meta_c, labels = preview(block_volume(alpha=alpha_np, cyst_radius=4), depth=40)
    db = 20 * np.log10(env_c + 1e-9)
    cyst = np.isin(labels, (tissue.CYST,))
    rows, cols = np.nonzero(cyst)
    inside = db[cyst].mean()
    side = db[rows.min():rows.max(), :max(1, cols.min() - 40)].mean()
    below = db[rows.max() + 30:rows.max() + 100, cols.min() + 10:cols.max() - 10].mean()
    below_side = db[rows.max() + 30:rows.max() + 100, :max(1, cols.min() - 40)].mean()
    path = 2 * 4.0 * 1e-3 * 0.85                                     # ~mean chord through the cyst, m
    rep["cyst"] = {"contrast_db": round(inside - side, 1),
                   "posterior_enhancement_db": round(below - below_side, 1),
                   "expected_enhancement_db_approx": round(2 * alpha_np * spec.fc / 1e6 * path * 8.686, 1)}

    (out / "report.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = 3 if "pymust_speckle" in rep else 2
        fig, ax = plt.subplots(1, n, figsize=(4 * n, 4))
        ax[0].imshow(20 * np.log10(w / w.max()), cmap="gray", vmin=-40, aspect=pz / px)
        ax[0].set_title("preview speckle")
        ax[1].imshow(db, cmap="gray", vmin=db.max() - 50, aspect=meta_c["px_z_mm"] / meta_c["px_x_mm"])
        ax[1].set_title("cyst test")
        if n == 3:
            ax[2].imshow(20 * np.log10(env_ref / env_ref.max()), cmap="gray", vmin=-40, aspect=0.5)
            ax[2].set_title("PyMUST speckle")
        fig.tight_layout()
        fig.savefig(out / "validation.png", dpi=110)
    except Exception as e:  # figure is optional
        print("figure skipped:", e)


if __name__ == "__main__":
    main()
