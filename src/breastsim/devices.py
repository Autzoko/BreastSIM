"""Virtual devices: per-frame scanner variation spanning the real datasets.

The display chain (TGC slope and cap, gain, dynamic range, gray map) is fitted separately to each
public dataset with src/scripts/calibrate_display.py (same rendered frames, same bounds as the
joint `breast` preset fit: gamma <= 1.6, TGC cap <= 60 dB). A virtual device draws a convex
combination of these fits (Dirichlet weights; concentration < 1 keeps most draws near one dataset),
so every sampled chain lies within the span of chains that reproduce a real dataset. The probe
centre frequency / bandwidth and the vendor smoothing are drawn within the ranges of the scanners
behind the datasets (presets.py: 10-14 MHz linears, 6-15 MHz ML6-15 / ABUS; smoothing of the
scanner presets 0.4-1.2 px). The default `breast` preset itself is unchanged.

    dev = sample_device(rng)                       # dict
    p = ImagingParams.from_preset("breast", **dev["params"])
"""
import dataclasses

import numpy as np

from .probe import PROBES

# calibrate_display.py --preset breast --dataset <name> --gamma-max 1.6 --cap-max 60 (2026-10-27,
# oat64 + USCT frames; BUS-BRA without depth profile: lesion-centred crops)
DISPLAY_FITS = {   # loss (W1 gray histogram + |depth profile|, gray levels): joint breast preset -> own fit
    "BrEaST": dict(tgc_db_cm_mhz=0.55, gain_db=-27.74, dynamic_range_db=61.24, gray_gamma=1.60, tgc_max_db=40.65),  # 24.8 -> 8.9
    "TDSC-ABUS": dict(tgc_db_cm_mhz=0.36, gain_db=-13.85, dynamic_range_db=49.85, gray_gamma=1.60, tgc_max_db=49.10),  # 18.3 -> 13.0
    "BUSI": dict(tgc_db_cm_mhz=0.28, gain_db=-2.42, dynamic_range_db=42.07, gray_gamma=1.60, tgc_max_db=34.67),  # 39.0 -> 11.7
    "BUS-BRA": dict(tgc_db_cm_mhz=0.21, gain_db=-5.61, dynamic_range_db=49.55, gray_gamma=1.60, tgc_max_db=47.47),  # 13.0 -> 8.7
}   # gray_gamma sits at its 1.6 bound in every fit (as in the joint fit)

PROBE_RANGES = {"fc_mhz": (8.5, 13.0), "bandwidth": (0.6, 0.9)}
SMOOTHING_PX = (0.4, 1.2)
CONCENTRATION = 0.5


def probe_variant(base, fc_mhz, bandwidth):
    """Registers (once) and returns the name of `base` with another centre frequency / bandwidth.
    Values are rounded (0.25 MHz, 0.05) so the renderer's per-probe lookup tables stay few."""
    fc, bw = round(fc_mhz * 4) / 4, round(bandwidth * 20) / 20
    name = f"{base}@{fc:.2f}MHz/{bw:.2f}"
    if name not in PROBES:
        PROBES[name] = dataclasses.replace(PROBES[base], name=name, fc=fc * 1e6, bandwidth=bw)
    return name


def sample_device(rng, base_probe="L12-hf", fits=None):
    """Display chain + probe + smoothing of one virtual device. Returns
    {"params": ImagingParams overrides, "weights": {dataset: w}, "fc_mhz", "bandwidth"}."""
    fits = fits or DISPLAY_FITS
    names = sorted(fits)
    w = rng.dirichlet(np.full(len(names), CONCENTRATION))
    params = {k: float(sum(wi * fits[n][k] for wi, n in zip(w, names))) for k in fits[names[0]]}
    fc, bw = (float(rng.uniform(*PROBE_RANGES[k])) for k in ("fc_mhz", "bandwidth"))
    params["probe"] = probe_variant(base_probe, fc, bw)
    params["speckle_reduction"] = float(rng.uniform(*SMOOTHING_PX))
    return {"params": params, "weights": dict(zip(names, np.round(w, 3).tolist())),
            "fc_mhz": round(fc * 4) / 4, "bandwidth": round(bw * 20) / 20}
