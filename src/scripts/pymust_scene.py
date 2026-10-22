#!/usr/bin/env python
"""Render one probe pose with PyMUST (SIMUS + DAS) and with the real-time preview, side by side.

Prototype of the high-fidelity path: scatterers come from the shared lattice
(breastsim.scatter) inside the elevation slab, scaled by tissue echogenicity
and by the heterogeneous two-way attenuation along their line (spectral LUT at
the scatterer depth); SIMUS adds the true field physics (diffraction,
sidelobes, elevation focusing, frequency-dependent PSF); plane-wave
compounding + delay-and-sum beamforming as on a research scanner.

    python src/scripts/pymust_scene.py usct/A --lesion 0 --out outputs/pymust_scene.png
"""
import argparse
import os
import time

# one BLAS thread per PyMUST worker process (avoids oversubscription on many-core hosts)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch

from breastsim import phantom, scatter, tissue
from breastsim.probe import PROBES, ProbePose, skin_offsets
from breastsim.realtime.renderer import ImagingParams, PreviewRenderer, spectral_attenuation_lut


def scene_scatterers(vol, pose, spec, depth, slab_mm, keep_frac=1.0, conform=True):
    """Lattice scatterers in the elevation slab, in probe coordinates (x, y, z) m, with amplitudes."""
    origin, axes = pose.frame(vol.surface, vol.voxel_mm, spec)
    lat, ax, el = axes
    width = spec.aperture * 1e3
    xs = np.arange(-width / 2 - 2, width / 2 + 2, 0.07)
    offs = skin_offsets(vol.surface, vol.voxel_mm, origin, axes, xs, conform=conform)
    zs = np.arange(0, depth, 0.07)
    ys = np.arange(-slab_mm, slab_mm + 1e-6, 0.07)
    X, Z, Y = np.meshgrid(xs, zs, ys, indexing="ij")
    O = np.broadcast_to(offs[:, None, None], X.shape)
    pts = origin + X[..., None] * lat + (Z + O)[..., None] * ax + Y[..., None] * el
    idx = scatter.unique_cells(torch.from_numpy(pts.reshape(-1, 3)).float())
    seed = __import__("zlib").crc32(vol.id.encode()) % 100_000           # same seed as the preview
    if keep_frac < 1:  # deterministic thinning of the shared lattice; amplitudes rescaled to keep intensity
        u = scatter._uniform(scatter._hash(*idx.unbind(-1), seed, 9))
        idx = idx[u < keep_frac]
    amp, pos = scatter.from_index(idx, seed=seed)
    amp = amp / np.sqrt(keep_frac)
    pos, amp = pos.numpy().astype(np.float64), amp.numpy()
    rel = pos - origin
    x, y, zr = rel @ lat, rel @ el, rel @ ax
    z = zr - np.interp(x, xs, offs)                                        # depth along its own line
    keep = (np.abs(x) < width / 2 + 2) & (z > 0) & (z < depth) & (np.abs(y) <= slab_mm)
    x, y, z, pos, amp = x[keep], y[keep], z[keep], pos[keep], amp[keep]
    a0 = np.array([vol.a0_mm, 0.0, 0.0])
    iv = np.clip(np.round((pos - a0) / vol.voxel_mm).astype(int), 0, np.array(vol.label.shape) - 1)
    lab = vol.label[iv[:, 0], iv[:, 1], iv[:, 2]]
    echo = tissue.lut(tissue.ECHOGENICITY)[lab]
    # heterogeneous attenuation: integrate alpha along each scatterer's line (0.1 mm steps)
    A = np.zeros(len(z))
    steps = np.arange(0, depth, 0.1)
    xq = np.round(x / 0.25) * 0.25                                         # share line integrals per 0.25 mm column
    for xc in np.unique(xq):
        m = xq == xc
        line = origin + xc * lat + (steps[:, None] + np.interp(xc, xs, offs)) * ax
        li = np.clip(np.round((line - a0) / vol.voxel_mm).astype(int), 0, np.array(vol.label.shape) - 1)
        cum = np.cumsum(vol.alpha[li[:, 0], li[:, 1], li[:, 2]].astype(np.float64)) * 0.1e-3
        A[m] = np.interp(z[m], steps, cum)
    Ag, E, _, _ = spectral_attenuation_lut(spec, vol.alpha_power)
    a = amp * echo * np.sqrt(np.interp(A, Ag, E))
    return x * 1e-3, y * 1e-3, z * 1e-3, a, lab


def run_pymust(x, y, z, rc, depth, width, angles, workers):
    import pymust
    param = pymust.getparam("L11-5v")
    param.fnumber = 1.7
    opt = pymust.utils.Options()
    opt.ParPool = True
    opt.setParPool(workers, "process")
    xi, zi = np.meshgrid(np.arange(-width / 2, width / 2, 0.1) * 1e-3, np.arange(0.5, depth, 0.05) * 1e-3)
    iq_sum = 0
    for k, a in enumerate(np.deg2rad(angles)):
        t0 = time.time()
        tx = pymust.txdelay(param, a)
        rf, _ = pymust.simus(x[None], y[None], z[None], rc[None], tx, param, opt)
        iq = pymust.rf2iq(rf, param)
        m = pymust.dasmtx(1j * np.ones(iq.shape), xi, zi, tx, param)
        iq_sum = iq_sum + (m @ iq.flatten(order="F")).reshape(xi.shape, order="F")
        print(f"  angle {k + 1}/{len(angles)}: {time.time() - t0:.0f} s", flush=True)
    tgc = np.exp(2 * 0.5 * 100 / 8.686 * param.fc / 1e6 * zi[:, :1])     # same 0.5 dB/cm/MHz TGC as the preview
    return np.abs(iq_sum) * tgc, xi, zi


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phantom")
    ap.add_argument("--lesion", type=int, default=0)
    ap.add_argument("--du", type=float, default=0.0, help="probe offset from the lesion along u, mm")
    ap.add_argument("--rotation", type=float, default=0.0)
    ap.add_argument("--depth", type=float, default=35.0)
    ap.add_argument("--slab", type=float, default=0.4, help="half elevation slab of scatterers, mm")
    ap.add_argument("--keep", type=float, default=0.15, help="fraction of lattice scatterers simulated")
    ap.add_argument("--angles", type=int, default=3)
    ap.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    ap.add_argument("--out", default="outputs/pymust_scene.png")
    args = ap.parse_args()

    vol = phantom.load(args.phantom)
    les = vol.meta["lesions"][args.lesion] if vol.meta.get("lesions") else None
    u, v = (les["center_mm"][1] + args.du, les["center_mm"][2]) if les else np.array(vol.label.shape[1:]) * vol.voxel_mm / 2
    pose = ProbePose(u=u, v=v, rotation=args.rotation)
    spec = PROBES["L11-5v"]
    width = spec.aperture * 1e3

    t0 = time.time()
    x, y, z, rc, lab = scene_scatterers(vol, pose, spec, args.depth, args.slab, args.keep)
    t1 = time.time()
    print(f"{len(x)} scatterers in {t1 - t0:.0f} s; running PyMUST ({args.workers} workers)", flush=True)
    env, xi, zi = run_pymust(x, y, z, rc, args.depth, width, np.linspace(-6, 6, args.angles), args.workers)
    t2 = time.time()
    print(f"{len(x)} scatterers ({t1 - t0:.0f} s), PyMUST {args.angles} angles {t2 - t1:.0f} s")

    r = PreviewRenderer()
    r.set_volume(vol)
    p = ImagingParams.from_preset("l11_5v", depth_mm=args.depth, overlay="none", speckle_reduction=0)
    prev, meta, labels = r.render(pose, p)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    db = 20 * np.log10(env / np.percentile(env, 99) + 1e-6)
    ext = [-width / 2, width / 2, args.depth, 0]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.2))
    axs[0].imshow(db, cmap="gray", vmin=-50, vmax=0, extent=[xi.min() * 1e3, xi.max() * 1e3, zi.max() * 1e3, zi.min() * 1e3])
    axs[0].set_title(f"PyMUST SIMUS + DAS ({args.angles} plane waves)\n{len(x)} scatterers, {t2 - t1:.0f} s")
    axs[1].imshow(prev[..., 0], cmap="gray", extent=ext)
    axs[1].set_title(f"Real-time preview (same pose)\n{meta['render_ms']:.0f} ms")
    cl = tissue.color_lut()[labels]
    axs[2].imshow(cl, extent=ext)
    axs[2].set_title("Ground truth tissue labels")
    for a in axs:
        a.set_xlabel("lateral (mm)")
    axs[0].set_ylabel("depth (mm)")
    fig.suptitle(f"{args.phantom}  lesion #{args.lesion}  u={u:.1f} v={v:.1f} mm  rot={args.rotation}°  L11-5v 7.6 MHz")
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print("->", args.out)


if __name__ == "__main__":
    main()
