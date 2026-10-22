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
import contextlib
import fcntl
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

from . import paths, tissue

PREVIEW_VOXEL_MM = 0.2
CACHE_VERSION = 4   # bump to rebuild caches after changing the conversion
WATER_Z, WATER_ALPHA, WATER_C = 1.5, 0.025, 1500.0

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
    c: np.ndarray = None              # speed of sound, m/s (float16)
    lesion_id: np.ndarray = None      # uint8, lesion index + 1 inside lesions, 0 elsewhere

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


# Thin or small structures that must survive downsampling (later wins).
PRIORITY_LABELS = (tissue.LIGAMENT, tissue.SKIN, tissue.EPIDERMIS, tissue.DUCT) + tissue.LESIONS


def _downsample_labels(a, k):
    """Nearest-neighbour labels, except that thin structures (ligaments, skin)
    and lesions win their block, so they are not lost at preview resolution."""
    out = _stride(a, k).copy()
    if k == 1:
        return out
    s = [n // k * k for n in a.shape]
    blocks = a[:s[0], :s[1], :s[2]].reshape(s[0] // k, k, s[1] // k, k, s[2] // k, k)
    for lab in PRIORITY_LABELS:
        hit = (blocks == lab).any(axis=(1, 3, 5))
        if hit.any():
            out[hit] = lab
    return out


def _read(path, key):
    with h5py.File(path, "r") as h:
        return h[key][()]


def _load_usct(info):
    voxel = float(np.array(_read(info.files["label"].with_name(
        info.files["label"].name.replace("_label", "_info")), "voxel_size")).ravel()[0])
    k = _factor(voxel)
    label = _downsample_labels(_read(info.files["label"], "label"), k)
    c = _block_mean(_read(info.files["sos"], "sos"), k)
    z = c * _block_mean(_read(info.files["density"], "density"), k) / 1e6
    alpha = _block_mean(_read(info.files["aa"], "aa"), k)
    t = info.id.split("/")[1]
    return label, z, alpha, c, voxel * k, USCT_ALPHA_POWER.get(t, tissue.DEFAULT_ALPHA_POWER)


def _load_oat64(info):
    with h5py.File(info.files["label"], "r") as h:
        voxel = float(np.array(h["voxel_size"]).ravel()[0])
        k = _factor(voxel)
        label = _downsample_labels(h["label"][()], k)
        if "label_pa" in h:
            pa = _stride(h["label_pa"][()], k) > 0
            label[pa & ~np.isin(label, (tissue.TUMOR, tissue.NECROSIS))] = tissue.ANGIOGENESIS
    with h5py.File(info.files["acou"], "r") as h:
        c = _block_mean(h["sound_speed"][()], k) * 1e3                    # mm/us -> m/s
        z = c * _block_mean(h["density"][()], k) * 1e6 / 1e6              # g/mm^3 -> kg/m^3, -> MRayl
        alpha = _block_mean(h["alpha_coeff"][()], k) * 1e3 / 8.686       # dB/(MHz^y mm) -> Np/(m MHz^y)
        power = float(np.array(h["alpha_power"]).ravel()[0])
    return label, z, alpha, c, voxel * k, power


def _load_oab(info):
    phan = _read(info.files["phan"], "phan")
    label = np.zeros_like(phan)
    for src, dst in OA_BREAST_LABELS.items():
        label[phan == src] = dst
    c, rho, a = (tissue.lut({k: v[i] for k, v in tissue.DEFAULT_ACOUSTIC.items()}) for i in range(3))
    return label, c[label] * rho[label] / 1e6, a[label], c[label], 0.2, tissue.DEFAULT_ALPHA_POWER


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


def canonicalize(label, z, alpha, c=None, margin=4):
    """Reorder to (A, U, V) with the chest wall at A=0, crop, pad with water.

    Returns (label, z, alpha, ori), or (label, z, alpha, c, ori) when c is given.
    """
    order, flip = _orient(label)
    fields = [(label, 0), (z, WATER_Z), (alpha, WATER_ALPHA)] + ([(c, WATER_C)] if c is not None else [])
    arrs = [np.transpose(a, order) for a, _ in fields]
    if flip:
        arrs = [a[::-1] for a in arrs]
    nz = np.nonzero(arrs[0] > 0)
    lo = [max(0, int(i.min()) - margin) for i in nz]
    hi = [int(i.max()) + 1 + margin for i in nz]
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    # pad so the nipple end and the sides are surrounded by water
    pad = [(0, margin)] + [(margin, margin)] * 2
    out = [np.pad(np.ascontiguousarray(a[sl]), pad, constant_values=w) for a, (_, w) in zip(arrs, fields)]
    water = out[0] == tissue.BACKGROUND
    for a, (_, w) in zip(out[1:], fields[1:]):
        a[water] = w
    return (*out, {"axis_order": order, "flip": bool(flip), "crop_lo": lo})


def lesion_acoustics(pid, lesions):
    """Per-lesion posterior feature and attenuation slope, deterministic per phantom/lesion.

    Drawn at load time (not cached) from tissue.POSTERIOR_FEATURE_PROBS, so the
    distribution can be tuned without rebuilding caches. Adds "posterior" and
    "attenuation_db_cm_mhz" to each lesion dict.
    """
    import zlib
    codes = {v: k for k, v in tissue.NAMES.items()}
    for les in lesions:
        rng = np.random.default_rng(zlib.crc32(f"{pid}#{les['id']}".encode()))
        main = codes.get(les["type"], tissue.TUMOR)
        probs = tissue.POSTERIOR_FEATURE_PROBS.get(main, tissue.POSTERIOR_FEATURE_PROBS[tissue.TUMOR])
        feats = list(probs)
        feat = feats[rng.choice(len(feats), p=np.array(list(probs.values())) / sum(probs.values()))]
        lo, hi = tissue.CYST_ATTENUATION_DB_CM_MHZ if main == tissue.CYST else tissue.POSTERIOR_ATTENUATION_DB_CM_MHZ[feat]
        les["posterior"] = feat
        les["attenuation_db_cm_mhz"] = round(float(lo + rng.random() * (hi - lo)), 3)
    return lesions


def _surface(label, voxel):
    tissue_mask = label > 0
    any_t = tissue_mask.any(axis=0)
    last = label.shape[0] - 1 - np.argmax(tissue_mask[::-1], axis=0)
    surf = (last + 0.5).astype(np.float32) * voxel
    surf[~any_t] = np.nan
    return surf


def find_lesions(label, voxel, min_voxels=8, return_ids=False):
    """Connected lesion components: type, centroid (a, u, v) mm, size.

    With return_ids, also returns a uint8 volume holding lesion id + 1 (0 elsewhere)."""
    from scipy import ndimage
    mask = np.isin(label, tissue.LESIONS)
    comps, n = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    out = []
    ids = np.zeros(label.shape, np.uint8) if return_ids else None
    for i, sl in enumerate(ndimage.find_objects(comps), 1):
        sub = comps[sl] == i
        nvox = int(sub.sum())
        if nvox < min_voxels or len(out) >= 254:
            continue
        if return_ids:
            ids[sl][sub] = len(out) + 1
        idx = np.argwhere(sub) + [s.start for s in sl]
        labs, cnt = np.unique(label[sl][sub], return_counts=True)
        kinds = [tissue.NAMES[int(l)] for l in labs[np.argsort(-cnt)]]
        vol_mm3 = nvox * voxel ** 3
        out.append({
            "id": len(out), "type": kinds[0], "parts": kinds,
            "center_mm": [float(c) for c in idx.mean(0) * voxel],   # voxel i is centred at i * voxel
            "bbox_mm": [[float(s.start * voxel), float(s.stop * voxel)] for s in sl],
            "volume_mm3": round(vol_mm3, 1), "diameter_mm": round((6 * vol_mm3 / np.pi) ** (1 / 3), 2),
        })
    return (out, ids) if return_ids else out


def _mesh(mask, voxel, factor, sigma=1.0):
    """Smoothed marching-cubes surface; vertices in (a, u, v) mm."""
    from scipy import ndimage
    from skimage import measure
    if factor > 1:
        s = [n // factor * factor for n in mask.shape]
        mask = mask[:s[0], :s[1], :s[2]].reshape(s[0] // factor, factor, s[1] // factor, factor,
                                                  s[2] // factor, factor).mean(axis=(1, 3, 5))
    f = ndimage.gaussian_filter(np.pad(mask.astype(np.float32), 1), sigma)
    if f.max() < 0.5:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint32)
    verts, faces, _, _ = measure.marching_cubes(f, 0.5)
    verts = ((verts - 1) * factor + (factor - 1) / 2) * voxel
    return verts.astype(np.float32), faces.astype(np.uint32)


def build_meshes(label, voxel, lesions):
    meshes = {}
    meshes["skin"] = _mesh(label > 0, voxel, factor=4)
    meshes["gland"] = _mesh(label == tissue.GLAND, voxel, factor=3, sigma=0.7)
    for les in lesions:
        (a0, a1), (u0, u1), (v0, v1) = [[int(x / voxel) for x in b] for b in les["bbox_mm"]]
        pad = 2
        lo = [max(0, a0 - pad), max(0, u0 - pad), max(0, v0 - pad)]
        sub = np.isin(label[lo[0]:a1 + pad, lo[1]:u1 + pad, lo[2]:v1 + pad], tissue.LESIONS)
        v, f = _mesh(sub, voxel, factor=1, sigma=0.6)
        meshes[f"lesion{les['id']}"] = (v + np.array(lo, np.float32) * voxel, f)
    return meshes


# -------------------------------------------------------------------- cache
def cache_dir(pid):
    return paths.CACHE / "preview" / pid.replace("/", "__")


def build_cache(pid, log=print):
    info = get_info(pid)
    t0 = time.time()
    log(f"[phantom] {pid}: [1/4] reading source files ({info.source})")
    label, z, alpha, c, voxel, power = LOADERS[info.source](info)
    log(f"[phantom] {pid}: [2/4] reorienting and writing preview volume")
    label, z, alpha, c, ori = canonicalize(label, z, alpha, c)
    log(f"[phantom] {pid}: [3/4] finding lesions")
    lesions, lesion_id = find_lesions(label, voxel, return_ids=True)
    out = cache_dir(pid)
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").unlink(missing_ok=True)   # meta.json last = cache complete
    np.save(out / "label.npy", label)
    np.save(out / "z.npy", z.astype(np.float16))
    np.save(out / "alpha.npy", alpha.astype(np.float16))
    np.save(out / "c.npy", c.astype(np.float16))
    np.save(out / "lesion_id.npy", lesion_id)
    np.save(out / "surface.npy", _surface(label, voxel))
    np.save(out / "lesion_proj.npy", np.isin(label, tissue.LESIONS).any(axis=0))
    log(f"[phantom] {pid}: [4/4] building 3D surfaces")
    meshes = build_meshes(label, voxel, lesions)
    np.savez_compressed(out / "meshes.npz", **{f"{k}_{p}": a for k, va in meshes.items()
                                                for p, a in zip(("verts", "faces"), va)})
    labels, counts = np.unique(label, return_counts=True)
    meta = {"version": CACHE_VERSION, "id": pid, "source": info.source, "description": info.description,
            "voxel_mm": voxel, "alpha_power": power, "shape": list(label.shape), "orientation": ori,
            "labels": {int(l): int(c) for l, c in zip(labels, counts)},
            "mean_z": {int(l): float(z[label == l].mean()) for l in labels},
            "mean_alpha": {int(l): float(alpha[label == l].mean()) for l in labels},
            "mean_c": {int(l): float(c[label == l].mean()) for l in labels},
            "lesions": lesions, "meshes": {k: [len(v[0]), len(v[1])] for k, v in meshes.items()},
            "built": time.strftime("%Y-%m-%d %H:%M:%S"), "build_seconds": round(time.time() - t0, 1)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    log(f"[phantom] {pid}: shape {label.shape} @ {voxel} mm, {len(lesions)} lesion(s), "
        f"{meta['build_seconds']} s")
    return out


def is_cached(pid):
    f = cache_dir(pid) / "meta.json"
    return f.exists() and json.loads(f.read_text()).get("version") == CACHE_VERSION


@contextlib.contextmanager
def _build_lock(pid):
    """Cross-process lock so concurrent builders of one phantom wait for each other."""
    d = cache_dir(pid)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / ".build.lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def ensure_cache(pid, log=print):
    """Build the preview cache unless it is current (safe to call concurrently)."""
    if is_cached(pid):
        return
    with _build_lock(pid):
        if not is_cached(pid):     # another process may have finished while we waited
            build_cache(pid, log)


def load(pid, log=print):
    d = cache_dir(pid)
    ensure_cache(pid, log)
    meta = json.loads((d / "meta.json").read_text())
    lesion_acoustics(pid, meta["lesions"])
    return Volume(pid, np.load(d / "label.npy"), np.load(d / "z.npy"), np.load(d / "alpha.npy"),
                  meta["voxel_mm"], meta["alpha_power"], np.load(d / "surface.npy"),
                  np.load(d / "lesion_proj.npy"), meta, np.load(d / "c.npy"), np.load(d / "lesion_id.npy"))


def load_meshes(pid):
    """{name: (verts (N,3) a,u,v mm, faces (M,3))} from the cache."""
    with np.load(cache_dir(pid) / "meshes.npz") as z:
        names = sorted({k.rsplit("_", 1)[0] for k in z.files})
        return {n: (z[f"{n}_verts"], z[f"{n}_faces"]) for n in names}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="List phantoms / build preview caches")
    ap.add_argument("ids", nargs="*", help="phantom ids to cache (default: list)")
    ap.add_argument("--all", action="store_true", help="cache every phantom that is not cached yet")
    args = ap.parse_args()
    ids = [p.id for p in list_phantoms() if not is_cached(p.id)] if args.all else args.ids
    if not ids:
        for p in list_phantoms():
            print(f"{p.id:24s} {'cached' if is_cached(p.id) else '      '}  {p.description}")
    for pid in ids:
        ensure_cache(pid, log=lambda m: print(m, flush=True))
