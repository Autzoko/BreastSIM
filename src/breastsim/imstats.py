"""Pixel- and frequency-domain statistics of B-mode images (real or simulated).

Used to compare simulated frames with public breast US datasets on the same footing.
All statistics are computed on tissue pixels only: background (outside the sector),
burnt-in annotations and (dilated) lesion masks are excluded by the caller's mask.

Per image:
- gray levels: mean, percentiles, std, skewness, fraction of near-black pixels;
- depth profile: median gray in 10 equal depth bins of the tissue region;
- speckle size: FWHM of the axial / lateral autocorrelation of 64x64 tiles after
  removing slow trends (Gaussian sigma 8 px);
- power spectrum of the tiles: band energies, log-log slope, horizontal anisotropy
  (energy of axially varying = horizontally oriented structures over laterally varying
  ones, 8-32 px periods);
- structure tensor (sigma 6 px): mean coherence and the coherence-weighted fraction of
  near-horizontal structures;
- mottling: std of 16x16 block means over the tile std.
"""
import gzip

import numpy as np
from scipy import ndimage as ndi

T = 64                       # tile size, px
_HANN = np.outer(np.hanning(T), np.hanning(T))
_fy, _fx = np.meshgrid(np.fft.fftfreq(T), np.fft.fftfreq(T), indexing="ij")
_FR = np.hypot(_fy, _fx)
_ANG = np.degrees(np.arctan2(np.abs(_fx), np.abs(_fy)))       # 0 = axial frequency
_RB = np.linspace(0, 0.5, 17)


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
    sizes = [int(s) for s in hdr["sizes"].split()]
    return np.frombuffer(raw, dt).reshape(sizes[::-1]), hdr


def tissue_mask(g, bad=None):
    """Tissue pixels of a gray image: the largest non-black region, eroded, minus saturated
    annotation pixels and `bad` (e.g. dilated lesion masks, coloured overlays)."""
    body = ndi.binary_opening(ndi.uniform_filter(g, 9) > 6, iterations=3)
    lab, n = ndi.label(body)
    if n:
        body = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    body = ndi.binary_erosion(body, iterations=6)
    sat = ndi.binary_dilation(g >= 250, iterations=2)
    return body & ~sat & (~bad if bad is not None else True)


def _tiles(g, m, step=T // 2):
    H, W = g.shape
    for i in range(0, H - T + 1, step):
        for j in range(0, W - T + 1, step):
            if m[i:i + T, j:j + T].mean() > 0.98:
                yield g[i:i + T, j:j + T]


def _fwhm(ac):
    c = len(ac) // 2
    v = ac[c:] / ac[c]
    if not (v < 0.5).any():
        return np.nan
    k = np.argmax(v < 0.5)
    return 2 * (k - 1 + (v[k - 1] - 0.5) / max(v[k - 1] - v[k], 1e-6))


def depth_profile(g, m, bins=10):
    rows = np.nonzero(m.any(1))[0]
    r0, r1 = rows.min(), rows.max()
    out = []
    for b in range(bins):
        a, z = r0 + (r1 - r0) * b // bins, r0 + (r1 - r0) * (b + 1) // bins
        s = g[a:z][m[a:z]]
        out.append(float(np.median(s)) if s.size else np.nan)
    return out


def analyse(g, m, scale=1.0):
    """Statistics of gray image g (0-255) on mask m; scale resamples first (e.g. to 0.1 mm/px)."""
    g = np.asarray(g, np.float32)
    if scale != 1.0:
        g = ndi.zoom(g, scale, order=1)
        m = ndi.zoom(m.astype(np.float32), scale, order=0) > 0.5
    v = g[m]
    if v.size < 4000:
        return None
    out = dict(mean=float(v.mean()), p05=float(np.percentile(v, 5)), p50=float(np.median(v)),
               p95=float(np.percentile(v, 95)), std=float(v.std()),
               skew=float(((v - v.mean()) ** 3).mean() / v.std() ** 3), dark_frac=float((v < 20).mean()),
               hist=np.histogram(v, 64, (0, 256))[0].tolist(), depth_profile=depth_profile(g, m))
    psd = np.zeros((T, T))
    fa, fl, mot = [], [], []
    for t in _tiles(g, m):
        d = t - ndi.gaussian_filter(t, 8)
        ac = np.fft.fftshift(np.real(np.fft.ifft2(np.abs(np.fft.fft2(d)) ** 2)))
        fa.append(_fwhm(ac[:, T // 2]))
        fl.append(_fwhm(ac[T // 2, :]))
        F = np.abs(np.fft.fft2((t - t.mean()) * _HANN)) ** 2
        psd += F / F[_FR > 0].sum()
        bm = t.reshape(4, 16, 4, 16).mean((1, 3))
        mot.append(bm.std() / max(t.std(), 1e-6))
    n = len(fa)
    out["n_tiles"] = n
    if n < 3:
        return out
    psd /= n
    band = lambda a, b: float(psd[(_FR >= a) & (_FR < b)].sum())
    sel = (_FR >= 1 / 32) & (_FR <= 1 / 4)
    mid = (_FR >= 1 / 32) & (_FR < 1 / 8)
    gs = ndi.gaussian_filter(g, 1.5)
    gx, gz = ndi.sobel(gs, 1), ndi.sobel(gs, 0)
    J = [ndi.gaussian_filter(a, 6) for a in (gx * gx, gz * gz, gx * gz)]
    coh = np.sqrt((J[0] - J[1]) ** 2 + 4 * J[2] ** 2) / (J[0] + J[1] + 1e-6)
    theta = np.abs(0.5 * np.degrees(np.arctan2(2 * J[2], J[1] - J[0])))   # 0 = horizontal structure
    w = coh[m]
    out.update(speckle_axial_px=float(np.nanmedian(fa)), speckle_lateral_px=float(np.nanmedian(fl)),
               speckle_aspect=float(np.nanmedian(np.array(fl) / np.array(fa))),
               psd_low=band(0, 1 / 16), psd_mid=band(1 / 16, 1 / 4), psd_high=band(1 / 4, 0.75),
               psd_slope=float(np.polyfit(np.log(_FR[sel]), np.log(psd[sel]), 1)[0]),
               aniso_horiz=float(psd[mid & (_ANG < 20)].mean() / psd[mid & (_ANG > 70)].mean()),
               radial_psd=[float(psd[(_FR >= _RB[k]) & (_FR < _RB[k + 1])].mean()) for k in range(16)],
               coherence=float(w.mean()), horiz_frac=float((w * (theta[m] < 20)).sum() / w.sum()),
               mottling=float(np.mean(mot)))
    return out
