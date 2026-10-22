"""Phantom registry and canonical volumes.

Every source is converted to one canonical form:
    label  uint8   (A, U, V)  canonical tissue codes (see tissue.py)
    z      float16 (A, U, V)  acoustic impedance, MRayl
    alpha  float16 (A, U, V)  attenuation coefficient, Np/m/MHz^y
Axis A runs anterior-posterior with the chest wall at index 0 and the nipple
at the far end; U, V span the coronal plane. World coordinates are mm,
voxel i sits at i * voxel_mm.

Preview volumes are resampled to ~0.2 mm and cached under
$BREASTSIM_BIG/cache/preview/<id>/ as .npy files.
"""
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

from . import paths, tissue

PREVIEW_VOXEL_MM = 0.2
WATER_Z, WATER_ALPHA = 1.5, 0.025

# Li et al. 2021, power-law exponent per BI-RADS type.
USCT_ALPHA_POWER = {"A": 1.1151, "B": 1.1642, "C": 1.2563, "D": 1.3635}
OA_BREAST_LABELS = {2: tissue.GLAND, 3: tissue.FAT, 4: tissue.SKIN, 5: tissue.VEIN}


@dataclass
class PhantomInfo:
    id: str            # e.g. "usct/A", "oat64/A40867720l", "oab/Neg_07_Left"
    source: str        # usct | oat64 | oab
    files: dict        # role -> Path
    description: str


@dataclass
class Volume:
    id: str
    label: np.ndarray
    z: np.ndarray
    alpha: np.ndarray
    voxel_mm: float
    alpha_power: float
    surface: np.ndarray = None       # (U, V) skin height along A in mm, NaN outside
    lesion_proj: np.ndarray = None   # (U, V) bool, lesion present along A
    meta: dict = field(default_factory=dict)

    @property
    def extent_mm(self):
        return tuple(n * self.voxel_mm for n in self.label.shape)


# ----------------------------------------------------------------- registry
def list_phantoms():
    out = []
    for f in sorted((paths.DATA / "nbp3d_usct").glob("*/*_label.mat")):
        t = f.parent.name
        out.append(PhantomInfo(f"usct/{t}", "usct", {
            "label": f, "sos": f.with_name(f.name.replace("_label", "_sos")),
            "density": f.with_name(f.name.replace("_label", "_density")),
            "aa": f.with_name(f.name.replace("_label", "_aa"))},
            f"USCT NBP, BI-RADS {t}, tumor"))
    state = paths.DATA / "nbp3d_oat64" / ".fetch_state.json"
    finished = set(json.loads(state.read_text())) if state.exists() else set()
    for f in sorted((paths.DATA / "nbp3d_oat64").glob("*/*/*_label.mat")):
        acou = f.with_name(f.name.replace("_label", "_acou"))
        if f"{f.parent.name}.zip" not in finished or not acou.exists():
            continue  # still being extracted by fetch_data.py
        name = f.name.split("_")[0]
        kind = "lesion" if name.endswith("l") else "healthy"
        out.append(PhantomInfo(f"oat64/{name}", "oat64", {"label": f, "acou": acou},
                               f"OAT NBP, BI-RADS {name[0]}, {kind}"))
    for f in sorted((paths.DATA / "oa_breast").glob("*.h5")):
        out.append(PhantomInfo(f"oab/{f.stem}", "oab", {"phan": f}, "OA-Breast (MRI-derived), healthy"))
    return out


def get_info(pid):
    for p in list_phantoms():
        if p.id == pid:
            return p
    raise KeyError(pid)


# ------------------------------------------------------------------ loaders
def _factor(voxel):
    return max(1, int(round(PREVIEW_VOXEL_MM / voxel)))


def _block_mean(a, k):
    if k == 1:
        return a.astype(np.float32)
    s = [n // k * k for n in a.shape]
    a = a[:s[0], :s[1], :s[2]].astype(np.float32)
    return a.reshape(s[0] // k, k, s[1] // k, k, s[2] // k, k).mean(axis=(1, 3, 5))


def _stride(a, k):
    s = [n // k * k for n in a.shape]
    return a[:s[0]:k, :s[1]:k, :s[2]:k]


def _read(path, key):
    with h5py.File(path, "r") as h:
        return h[key][()]


def _load_usct(info):
    voxel = float(np.array(_read(info.files["label"].with_name(
        info.files["label"].name.replace("_label", "_info")), "voxel_size")).ravel()[0])
    k = _factor(voxel)
    label = _stride(_read(info.files["label"], "label"), k)
    c = _block_mean(_read(info.files["sos"], "sos"), k)
    z = c * _block_mean(_read(info.files["density"], "density"), k) / 1e6
    del c
    alpha = _block_mean(_read(info.files["aa"], "aa"), k)
    t = info.id.split("/")[1]
    return label, z, alpha, voxel * k, USCT_ALPHA_POWER.get(t, tissue.DEFAULT_ALPHA_POWER)


def _load_oat64(info):
    with h5py.File(info.files["label"], "r") as h:
        voxel = float(np.array(h["voxel_size"]).ravel()[0])
        k = _factor(voxel)
        label = _stride(h["label"][()], k)
        if "label_pa" in h:
            pa = _stride(h["label_pa"][()], k) > 0
            label[pa & ~np.isin(label, (tissue.TUMOR, tissue.NECROSIS))] = tissue.ANGIOGENESIS
    with h5py.File(info.files["acou"], "r") as h:
        c = _block_mean(h["sound_speed"][()], k) * 1e3                    # mm/us -> m/s
        z = c * _block_mean(h["density"][()], k) * 1e6 / 1e6              # g/mm^3 -> kg/m^3, -> MRayl
        del c
        alpha = _block_mean(h["alpha_coeff"][()], k) * 1e3 / 8.686       # dB/(MHz^y mm) -> Np/(m MHz^y)
        power = float(np.array(h["alpha_power"]).ravel()[0])
    return label, z, alpha, voxel * k, power


def _load_oab(info):
    phan = _read(info.files["phan"], "phan")
    label = np.zeros_like(phan)
    for src, dst in OA_BREAST_LABELS.items():
        label[phan == src] = dst
    c, rho, a = (tissue.lut({k: v[i] for k, v in tissue.DEFAULT_ACOUSTIC.items()}) for i in range(3))
    return label, c[label] * rho[label] / 1e6, a[label], 0.2, tissue.DEFAULT_ALPHA_POWER


LOADERS = {"usct": _load_usct, "oat64": _load_oat64, "oab": _load_oab}


# --------------------------------------------------------------- canonical
def _orient(label):
    """Axis order / flips that put AP first with the chest wall at index 0.

    The chest wall is the end of the breast with the largest cross-section;
    the nipple end tapers to a few voxels.
    """
    best = None
    for ax in range(3):
        other = tuple(a for a in range(3) if a != ax)
        occ = (label > 0).mean(axis=other)
        idx = np.nonzero(occ)[0]
        lo, hi = occ[idx[0]], occ[idx[-1]]
        score = max(lo, hi)
        if best is None or score > best[0]:
            best = (score, ax, hi > lo)
    _, ax, flip = best
    order = (ax,) + tuple(a for a in range(3) if a != ax)
    return order, flip


def canonicalize(label, z, alpha, margin=4):
    order, flip = _orient(label)
    arrs = [np.transpose(a, order) for a in (label, z, alpha)]
    if flip:
        arrs = [a[::-1] for a in arrs]
    nz = np.nonzero(arrs[0] > 0)
    lo = [max(0, int(i.min()) - margin) for i in nz]
    hi = [int(i.max()) + 1 + margin for i in nz]
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    label, z, alpha = (np.ascontiguousarray(a[sl]) for a in arrs)
    # pad so the nipple end and the sides are surrounded by water
    pad = [(0, margin)] + [(margin, margin)] * 2
    label = np.pad(label, pad)
    z = np.pad(z, pad, constant_values=WATER_Z)
    alpha = np.pad(alpha, pad, constant_values=WATER_ALPHA)
    water = label == tissue.BACKGROUND
    z[water], alpha[water] = WATER_Z, WATER_ALPHA
    return label, z, alpha, {"axis_order": order, "flip": bool(flip), "crop_lo": lo}


def _surface(label, voxel):
    tissue_mask = label > 0
    any_t = tissue_mask.any(axis=0)
    last = label.shape[0] - 1 - np.argmax(tissue_mask[::-1], axis=0)
    surf = (last + 0.5).astype(np.float32) * voxel
    surf[~any_t] = np.nan
    return surf


# -------------------------------------------------------------------- cache
def cache_dir(pid):
    return paths.CACHE / "preview" / pid.replace("/", "__")


def build_cache(pid, log=print):
    info = get_info(pid)
    t0 = time.time()
    log(f"[phantom] building preview cache for {pid}")
    label, z, alpha, voxel, power = LOADERS[info.source](info)
    label, z, alpha, ori = canonicalize(label, z, alpha)
    out = cache_dir(pid)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "label.npy", label)
    np.save(out / "z.npy", z.astype(np.float16))
    np.save(out / "alpha.npy", alpha.astype(np.float16))
    np.save(out / "surface.npy", _surface(label, voxel))
    np.save(out / "lesion_proj.npy", np.isin(label, tissue.LESIONS).any(axis=0))
    labels, counts = np.unique(label, return_counts=True)
    meta = {"id": pid, "source": info.source, "description": info.description, "voxel_mm": voxel,
            "alpha_power": power, "shape": list(label.shape), "orientation": ori,
            "labels": {int(l): int(c) for l, c in zip(labels, counts)},
            "mean_z": {int(l): float(z[label == l].mean()) for l in labels},
            "mean_alpha": {int(l): float(alpha[label == l].mean()) for l in labels},
            "built": time.strftime("%Y-%m-%d %H:%M:%S"), "build_seconds": round(time.time() - t0, 1)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    log(f"[phantom] {pid}: shape {label.shape} @ {voxel} mm in {meta['build_seconds']} s")
    return out


def load(pid, log=print):
    d = cache_dir(pid)
    if not (d / "meta.json").exists():
        build_cache(pid, log)
    meta = json.loads((d / "meta.json").read_text())
    return Volume(pid, np.load(d / "label.npy"), np.load(d / "z.npy"), np.load(d / "alpha.npy"),
                  meta["voxel_mm"], meta["alpha_power"], np.load(d / "surface.npy"),
                  np.load(d / "lesion_proj.npy"), meta)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="List phantoms / build preview caches")
    ap.add_argument("ids", nargs="*", help="phantom ids to cache (default: list)")
    ap.add_argument("--all", action="store_true", help="cache every phantom")
    args = ap.parse_args()
    ids = [p.id for p in list_phantoms()] if args.all else args.ids
    if not ids:
        for p in list_phantoms():
            print(f"{p.id:24s} {'cached' if (cache_dir(p.id) / 'meta.json').exists() else '      '}  {p.description}")
    for pid in ids:
        build_cache(pid)
