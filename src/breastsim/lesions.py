"""Lesion appearance from real annotated lesions: type sampling and per-lesion content.

A lesion "recipe" is one annotated BrEaST lesion (lesion_types.json, built by src/scripts/lesion_types.py):
its joint BI-RADS descriptors (pathology, shape, margin, echo pattern, posterior features, halo,
calcifications) and the features measured on its image (size, depth, contrast...). Recipes are drawn with
tempered type weights (rare types moderately up-weighted). Geometry follows the recipe at insertion
(src/scripts/victre_lesion_typed.py: breastMass morphology, size, calcification cluster); acoustics follow
it at load time (apply_content), deterministically per phantom / lesion:

- internal echogenicity: lesion backscatter amplitude = ratio x rms echogenicity of the host tissue 1-3 mm
  around it; the ratio is the inverse of the rendered contrast curve (CONTRAST_CURVE, measured on simulated
  lesions with breastsim.realism.lesion_features) at the recipe's measured contrast (contrast_sd on the real
  image); circumscribed lesions are homogeneous, non-circumscribed / heterogeneous ones keep the backscatter
  pattern of the host tissue they replaced (entrapped fat / stroma) at that mean level;
- complex cystic / solid: fluid (anechoic) parts, ~CYSTIC_CORE_FRACTION of the lesion volume, where the
  replaced host tissue was least echogenic (irregular, anatomy-shaped); without a host pattern the deepest voxels;
- non-circumscribed margins: an infiltration transition zone of TRANSITION_MM on both sides of the boundary
  where tumour and host backscatter mix by volume fraction (linear in intensity with signed distance);
- echogenic halo (annotated halo): host tissue within HALO_REL_WIDTH x diameter outside the lesion gets
  HALO_FACTOR x its backscatter amplitude; circumscribed masses without halo get a thin echogenic pseudocapsule
  (CAPSULE_MM, CAPSULE_FACTOR);
- posterior features: the recipe's annotated feature (phantom.lesion_acoustics).
Masks never change: everything acts on the backscatter factor (Volume.echo_mod) and the lesion attenuation.
"""
import json
import zlib
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import tissue

TYPES_FILE = Path(__file__).with_name("lesion_types.json")
# rendered contrast_sd (breastsim.realism) vs lesion / host backscatter amplitude ratio, breast preset;
# median over 28 typed VICTRE masses x 2 poses (supine hetero / scattered phantoms, homogeneous, capsule 1.6, no
# posterior feature; src/scripts/lesion_calibrate.py, 2026-10-27). Monotonic.
CONTRAST_CURVE = {"ratio": [0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0],
                  "contrast_sd": [-2.638, -2.488, -2.051, -1.497, -1.066, -0.625, -0.161, 0.379, 0.962]}
# fluid (cystic core): ratio at the median rendered contrast of real anechoic BrEaST lesions (-2.05 -> ~0.13;
# real cysts are not black: clutter / noise fill them)
FLUID_RATIO = 0.13
CYSTIC_CORE_FRACTION = 0.4
TRANSITION_MM = 0.4            # 0.8 mm gave edge_sharpness 0.99 vs real 1.16-1.29 (non-circumscribed)
HALO_REL_WIDTH = 0.15           # realism.lesion_features measures the halo over 0.15 x diameter
HALO_FACTOR = 1.5
# circumscribed masses: "thin echogenic pseudocapsule" (Stavros et al., Radiology 1995;196:123, benign criteria);
# also what puts the real-mask rim above the surroundings (BrEaST halo_sd ~0.2 in every type)
CAPSULE_MM = 0.4
CAPSULE_FACTOR = 1.4            # 1.6 gave halo_sd 0.34-0.54 vs real ~0.23
POSTERIOR = {"no": "none", "shadowing": "shadowing", "enhancement": "enhancement", "combined": "combined"}


@lru_cache(maxsize=1)
def load_types():
    return json.loads(TYPES_FILE.read_text())


def sample_recipe(key, echo=None):
    """Deterministic recipe for a string key (e.g. "<phantom id>#<lesion id>" or an insertion seed),
    optionally restricted to one echo pattern (e.g. "anechoic" for a simple cyst geometry)."""
    cases = [c for c in load_types()["cases"] if echo is None or c["echo"] == echo]
    w = np.array([c["weight"] for c in cases], float)
    rng = np.random.default_rng(zlib.crc32(str(key).encode()))
    return dict(cases[rng.choice(len(cases), p=w / w.sum())])


def mass_kind(r):
    """breastMass morphology for a recipe's margin / shape (victre_insert.py KINDS)."""
    m = r["margin"]
    if "spiculated" in m:
        return "spiculated"
    if "microlobulated" in m or r["shape"] == "irregular":
        return "lobulated"
    return "smooth"


def echo_ratio(r):
    if r.get("echo_ratio") is not None:          # explicit (calibration)
        return float(r["echo_ratio"])
    c = r.get("contrast_sd")
    if c is None:
        c = {"hypoechoic": -0.9, "heterogeneous": -0.5, "isoechoic": -0.45, "hyperechoic": 0.3,
             "complex cystic/solid": -0.8}.get(r["echo"], -0.9)
    x, y = CONTRAST_CURVE["ratio"], CONTRAST_CURVE["contrast_sd"]
    return float(np.exp(np.interp(c, y, np.log(x))))


def apply_content(vol):
    """Per-lesion echogenicity, cystic core, transition zone and halo into vol.echo_mod (in place)."""
    from scipy import ndimage
    lesions = vol.meta.get("lesions") or []
    if vol.lesion_id is None or not lesions:
        return vol
    if vol.echo_mod is None:
        vol.echo_mod = np.ones(vol.label.shape, np.float16)
    e_lut = tissue.lut(tissue.ECHOGENICITY)
    vox, a0 = vol.voxel_mm, int(round(-vol.a0_mm / vol.voxel_mm))
    for les in lesions:
        r = les.get("recipe")
        if r is None:
            continue
        (b0, b1), (u0, u1), (v0, v1) = les["bbox_mm"]
        d = les["diameter_mm"]
        pad = int(np.ceil(max(4.0, HALO_REL_WIDTH * d + TRANSITION_MM + 1) / vox))
        sl = tuple(slice(max(0, int(lo / vox) + off - pad), int(hi / vox) + off + pad + 1)
                   for (lo, hi), off in (((b0, b1), a0), ((u0, u1), 0), ((v0, v1), 0)))
        lab, ids = vol.label[sl], vol.lesion_id[sl]
        mod = vol.echo_mod[sl].astype(np.float32)
        L = ids == les["id"] + 1
        host = (ids == 0) & (lab != tissue.BACKGROUND)
        d_out = ndimage.distance_transform_edt(~L) * vox
        d_in = ndimage.distance_transform_edt(L) * vox
        amp = e_lut[lab] * mod                                     # current backscatter amplitude
        # host level in the ring realism.lesion_features compares against (0.15-0.5 x diameter, >= 1 mm),
        # as a log mean (the display is logarithmic; an rms is dominated by bright gland next to fat)
        shell = host & (d_out > max(1.0, 0.15 * d)) & (d_out <= max(3.0, 0.5 * d)) & (amp > 0)
        e_host = float(np.exp(np.mean(np.log(amp[shell])))) if shell.any() else 0.6
        solid = L & (lab != tissue.CALC)
        main = {v: k for k, v in tissue.NAMES.items()}.get(les["type"], tissue.TUMOR)
        if (r["margin"] != "circumscribed" or r["echo"] == "heterogeneous") and solid.any():
            # infiltrating / heterogeneous: internal pattern of the host tissue the mass replaced (typed VICTRE
            # insertions keep it in echo_mod) or the lesion's own sub-labels, at the target mean level
            pat = amp / np.sqrt(np.mean(amp[solid] ** 2))
        else:
            # circumscribed: homogeneous except sub-labels (e.g. a necrotic core keeps its echogenicity
            # relative to the lesion's main tissue)
            pat = e_lut[lab] / max(e_lut[main], 1e-6)
        target = np.where(solid, echo_ratio(r) * e_host * pat, amp)
        if r["echo"] == "complex cystic/solid" and solid.any():
            hv = amp[solid]
            if hv.std() > 0.1 * hv.mean():
                # fluid spaces where the replaced host tissue was least echogenic (fat lobules): irregular,
                # anatomy-shaped cystic parts; at most ~60 % of the lesion
                core = solid & (amp <= np.quantile(hv, CYSTIC_CORE_FRACTION))
                if core.sum() > 0.6 * solid.sum():
                    core &= d_in > np.median(d_in[solid])
            else:                                                  # no host pattern: deepest voxels
                core = solid & (d_in > np.quantile(d_in[solid], 1 - CYSTIC_CORE_FRACTION))
            target[core] = FLUID_RATIO * e_host
        if r["margin"] != "circumscribed":                         # infiltration transition zone
            sd = np.where(L, -d_in, d_out)
            t = np.clip((sd + TRANSITION_MM) / (2 * TRANSITION_MM), 0, 1)
            zone = (np.abs(sd) < TRANSITION_MM) & (solid | host)
            a_les, a_host = echo_ratio(r) * e_host, np.where(L, e_host, amp)
            target[zone] = np.sqrt((1 - t[zone]) * a_les ** 2 + t[zone] * a_host[zone] ** 2)
        if r.get("halo"):
            rim = host & (d_out > 0) & (d_out <= HALO_REL_WIDTH * d)
            target[rim] *= HALO_FACTOR
        elif r["margin"] == "circumscribed":                       # thin echogenic pseudocapsule
            rim = host & (d_out > 0) & (d_out <= CAPSULE_MM)
            target[rim] *= CAPSULE_FACTOR
        base = e_lut[lab]
        upd = (solid | host) & (base > 0)
        mod[upd] = target[upd] / base[upd]
        vol.echo_mod[sl] = mod.astype(np.float16)
    return vol
