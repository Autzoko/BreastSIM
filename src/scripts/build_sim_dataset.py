#!/usr/bin/env python
"""Simulated B-mode segmentation dataset: frames + lesion masks in sharded HDF5.

    # 1) plan: which phantom / view / lesion each frame shows (deterministic, one JSON)
    python src/scripts/build_sim_dataset.py plan --out $BIG/simdata/v1 --n 10000 --pos-frac 0.8
    # 2) render shard k of K (resumable: frames already in the shard are skipped)
    python src/scripts/build_sim_dataset.py render --out $BIG/simdata/v1 --shard 0 --shards 8
    # 3) contact sheet with mask contours, for checking alignment by eye
    python src/scripts/build_sim_dataset.py sheet --out $BIG/simdata/v1 --shard 0 --n 24

Imaging is the unchanged preset (default "breast"). Phantoms are used as cached (phantom.load).
Lesion frames: the probe is placed around the chosen lesion with random rotation and small
tilt / rock (probe.ProbePose angles, +-12 deg; +-35 deg only when no such pose reaches the lesion,
e.g. USCT B / C, whose lesions lie 39-44 mm below the skin straight above them); among many candidate poses the lesion centre must project
inside the field of view and onto the slice (elevation offset < 0.6 x lesion radius), and
shallower placements are preferred (weight exp(-depth / depth_scale_mm)): the phantoms are pendant
(prone) breasts, so lesions straight below the highest skin point sit deeper than in supine
clinical scans. After rendering the frame is kept only if the rendered tissue labels contain
lesion pixels (area >= --min-area mm2, after dropping components < 1/4 of the largest: detached
sub-resolution mass fragments); normal frames must contain none. Frames with more than 2 %
background (probe off the breast) are redrawn.

Shard file <out>/shard_<k>.h5: img (N, H, W) uint8, mask (N, H, W) uint8 (lesion = 1),
label (N, H, W) uint8 tissue codes (tissue.py), meta (N,) JSON strings.
"""
import argparse
import json
import time
import zlib
from pathlib import Path

import h5py
import numpy as np

from breastsim import phantom, tissue
from breastsim.probe import PROBES, ProbePose, skin_offsets
from breastsim.realtime.renderer import ImagingParams, PreviewRenderer

SOURCES = {   # source -> share of the lesion frames (split evenly over its phantoms), renormalised per plan
    "victre": 0.40, "oat64": 0.45, "usct": 0.15}


def lesion_phantoms(ids):
    out = {}
    for pid in ids:
        meta = json.loads((phantom.cache_dir(pid) / "meta.json").read_text())
        les = [l["id"] for l in meta["lesions"] if l["diameter_mm"] >= 3.0]
        if les:
            out[pid] = les
    return out


def cmd_plan(a):
    rng = np.random.default_rng(a.seed)
    ids = [l.strip() for l in open(a.phantoms) if l.strip() and not l.startswith("#")]
    cached = [p for p in ids if phantom.is_cached(p)]
    if len(cached) < len(ids):
        raise SystemExit(f"not cached: {sorted(set(ids) - set(cached))}")
    les = lesion_phantoms(ids)
    n_pos = int(round(a.n * a.pos_frac))
    frames = []
    present = {s: w for s, w in SOURCES.items() if any(p.split("/")[0] == s for p in les)}
    for src, share in present.items():
        share /= sum(present.values())      # a plan may cover only some sources (dataset built in parts)
        pids = [p for p in les if p.split("/")[0] == src]
        per = np.full(len(pids), int(n_pos * share) // len(pids))
        per[:int(n_pos * share) - per.sum()] += 1
        for pid, k in zip(pids, per):
            frames += [dict(phantom=pid, view="lesion", lesion=int(rng.choice(les[pid]))) for _ in range(k)]
    while len(frames) < n_pos:      # rounding / missing sources: top up from all lesion phantoms
        pid = list(les)[rng.integers(len(les))]
        frames.append(dict(phantom=pid, view="lesion", lesion=int(rng.choice(les[pid]))))
    for i in range(a.n - len(frames)):  # normal frames spread over every phantom
        frames.append(dict(phantom=ids[i % len(ids)], view="normal", lesion=-1))
    frames.sort(key=lambda f: f["phantom"])            # shards load each phantom once
    for i, f in enumerate(frames):
        f["idx"] = i
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plan = dict(preset=a.preset, seed=a.seed, min_area_mm2=a.min_area, depth_scale_mm=a.depth_scale,
                n=a.n, pos_frac=a.pos_frac, frames=frames)
    (out / "plan.json").write_text(json.dumps(plan))
    by = {}
    for f in frames:
        by.setdefault(f["phantom"].split("/")[0] + ":" + f["view"], 0)
        by[f["phantom"].split("/")[0] + ":" + f["view"]] += 1
    print(f"{len(frames)} frames, {len(les)} lesion phantoms of {len(ids)}: {by}")


def project(vol, pose, spec, p, point):
    """Lateral x, elevation e and displayed depth z (mm) of a phantom point in the frame of pose."""
    origin, axes = pose.frame(vol.surface, vol.voxel_mm, spec)
    rel = np.asarray(point) - origin
    x, z, e = rel @ axes[0], rel @ axes[1], rel @ axes[2]
    off = skin_offsets(vol.surface, vol.voxel_mm, origin, axes, [x], conform=p.conform)[0]
    return x, e, z - off


def lesion_pose(vol, les, spec, p, rng, depth_scale, n_cand=64, max_angle=12.0):
    c = np.array(les["center_mm"])
    r = les["diameter_mm"] / 2
    half_w = spec.aperture * 1e3 / 2
    cands, w = [], []
    for _ in range(n_cand):
        rot = rng.uniform(-180, 180)
        tilt, rock = rng.uniform(-max_angle, max_angle, 2)
        x0 = rng.uniform(-0.6, 0.6) * half_w                  # lesion lateral position in the image
        e0 = rng.uniform(-0.6, 0.6) * r                        # slice offset from the lesion centre
        th = np.deg2rad(rot)
        lat, elev = np.array([np.cos(th), np.sin(th)]), np.array([-np.sin(th), np.cos(th)])
        u, v = c[1:] - x0 * lat - e0 * elev
        pose = ProbePose(u=float(u), v=float(v), rotation=float(rot), tilt=float(tilt), rock=float(rock))
        x, e, z = project(vol, pose, spec, p, c)
        if abs(x) < half_w - r and abs(e) < 0.6 * r and r < z < p.depth_mm - r - 1:
            cands.append(pose)
            w.append(np.exp(-z / depth_scale))
    if not cands:   # lesion deeper than the field of view below the skin: lean the probe further
        return lesion_pose(vol, les, spec, p, rng, depth_scale, n_cand, 35.0) if max_angle < 35 else None
    w = np.array(w)
    return cands[rng.choice(len(cands), p=w / w.sum())]


def normal_pose(vol, rng):
    iu, iv = np.nonzero(np.isfinite(vol.surface))
    j = rng.integers(len(iu))
    return ProbePose(u=float(iu[j] * vol.voxel_mm), v=float(iv[j] * vol.voxel_mm),
                     rotation=float(rng.uniform(-180, 180)), tilt=float(rng.uniform(-12, 12)),
                     rock=float(rng.uniform(-12, 12)))


def clean_mask(les):
    """Segmentation target: the largest lesion component and any other of >= 1/4 of its area (the rule of
    breastsim.realism). Drops sub-resolution mass fragments (breastMass leaves detached voxels around some
    masses; invisible in the image, they would be label noise). The raw tissue labels are stored as well."""
    from scipy import ndimage
    lab, n = ndimage.label(les)
    if n <= 1:
        return les
    area = np.bincount(lab.ravel())[1:]
    keep = np.nonzero(area >= area.max() / 4)[0] + 1
    return np.isin(lab, keep)


def frame_stats(img, les, px):
    """Lesion size / shape / depth and gray levels (for dataset statistics)."""
    out = dict(lesion_area_mm2=round(float(les.sum()) * px * px, 2), gray_mean=round(float(img.mean()), 1))
    if les.any():
        rr, cc = np.nonzero(les)
        out.update(lesion_w_mm=round(float(cc.max() - cc.min() + 1) * px, 2),
                   lesion_h_mm=round(float(rr.max() - rr.min() + 1) * px, 2),
                   lesion_top_mm=round(float(rr.min()) * px, 2), lesion_center_mm=round(float(rr.mean()) * px, 2),
                   gray_lesion=round(float(img[les].mean()), 1), gray_tissue=round(float(img[~les].mean()), 1))
    return out


def cmd_render(a):
    out = Path(a.out)
    plan = json.loads((out / "plan.json").read_text())
    frames = [f for f in plan["frames"] if f["idx"] % a.shards == a.shard] if a.interleave else \
        np.array_split(np.array(plan["frames"], dtype=object), a.shards)[a.shard].tolist()
    p = ImagingParams.from_preset(plan["preset"], overlay="none")
    spec = PROBES[p.probe]
    r = PreviewRenderer()
    f5 = h5py.File(out / f"shard_{a.shard:03d}.h5", "a")
    if "img" not in f5:
        h, w = int(round(p.depth_mm / p.px_z_mm)), int(round(spec.aperture * 1e3 / p.px_x_mm))
        kw = dict(maxshape=(None, h, w), chunks=(1, h, w), compression="gzip", compression_opts=4)
        for k in ("img", "mask", "label"):
            f5.create_dataset(k, (0, h, w), np.uint8, **kw)
        f5.create_dataset("meta", (0,), h5py.string_dtype(), maxshape=(None,), chunks=(256,))
        f5.attrs["plan"] = json.dumps({k: v for k, v in plan.items() if k != "frames"})
        f5.attrs["imaging"] = json.dumps({k: v for k, v in vars(p).items()}, default=str)
    done = {json.loads(m)["idx"] for m in f5["meta"][:]}
    todo = [f for f in frames if f["idx"] not in done]
    print(f"shard {a.shard}/{a.shards}: {len(frames)} frames, {len(todo)} to render", flush=True)
    vol_id, t0, n_new, tries = None, time.time(), 0, 0
    for f in todo:
        if f["phantom"] != vol_id:
            vol = phantom.load(f["phantom"], log=lambda *m: None)
            r.set_volume(vol)
            vol_id = f["phantom"]
            lesions = {l["id"]: l for l in vol.meta["lesions"]}
        rng = np.random.default_rng(zlib.crc32(f"{plan['seed']}#{f['idx']}".encode()))
        res = None
        for attempt in range(a.max_tries):
            tries += 1
            pose = lesion_pose(vol, lesions[f["lesion"]], spec, p, rng, plan["depth_scale_mm"]) \
                if f["view"] == "lesion" else normal_pose(vol, rng)
            if pose is None:
                continue
            img, meta, lab = r.render(pose, p)
            img, les = img[..., 0], clean_mask(np.isin(lab, tissue.LESIONS))
            area = les.sum() * meta["px_x_mm"] * meta["px_z_mm"]
            ok = (lab == tissue.BACKGROUND).mean() <= 0.02 and (
                area >= plan["min_area_mm2"] if f["view"] == "lesion" else area == 0)
            if ok:
                res = (img, les, lab, meta, pose, attempt)
                break
        if res is None:
            print(f"  frame {f['idx']} ({f['phantom']} {f['view']} {f['lesion']}): no valid pose", flush=True)
            continue
        img, les, lab, meta, pose, attempt = res
        m = dict(f, pose=pose.to_dict(), attempts=attempt + 1, px_mm=meta["px_z_mm"],
                 lesion_types=meta["lesion_types"], skin_offset_mm=meta["skin_offset_mm"],
                 **frame_stats(img, les, meta["px_z_mm"]))
        n = f5["img"].shape[0]
        for k in ("img", "mask", "label", "meta"):
            f5[k].resize(n + 1, axis=0)
        f5["img"][n], f5["mask"][n], f5["label"][n] = img, les.astype(np.uint8), lab
        f5["meta"][n] = json.dumps(m)
        n_new += 1
        if n_new % 100 == 0:
            f5.flush()
            dt = time.time() - t0
            print(f"  {n_new}/{len(todo)} frames, {dt / n_new:.2f} s/frame, {tries / n_new:.2f} renders/frame", flush=True)
    f5.close()
    print(f"shard {a.shard}: +{n_new} frames in {time.time() - t0:.0f} s", flush=True)


def cmd_sheet(a):
    from PIL import Image
    from scipy import ndimage
    out = Path(a.out)
    shards = sorted(out.glob("shard_*.h5")) if a.shard < 0 else [out / f"shard_{a.shard:03d}.h5"]
    tiles = []
    rng = np.random.default_rng(0)
    for s in shards:
        with h5py.File(s, "r") as f5:
            n = f5["img"].shape[0]
            for i in np.sort(rng.choice(n, min(a.n // len(shards) or 1, n), replace=False)):
                g = f5["img"][i]
                m = f5["mask"][i].astype(bool)
                rgb = np.repeat(g[..., None], 3, -1)
                edge = m & ~ndimage.binary_erosion(m)
                rgb[ndimage.binary_dilation(edge)] = (60, 255, 60)
                tiles.append(np.concatenate([np.repeat(g[..., None], 3, -1), rgb], 1))
    cols = 4
    h, w = tiles[0].shape[:2]
    sheet = np.zeros((((len(tiles) + cols - 1) // cols) * h, cols * w, 3), np.uint8)
    for k, t in enumerate(tiles):
        sheet[k // cols * h:(k // cols + 1) * h, k % cols * w:(k % cols + 1) * w] = t
    Image.fromarray(sheet).resize((cols * w // 2, sheet.shape[0] // 2)).save(a.png or out / "sheet.png")
    print(f"{len(tiles)} frames -> {a.png or out / 'sheet.png'}")


def cmd_stats(a):
    """Dataset statistics over all shards of the given directories (counts, lesion size / shape / depth, gray)."""
    import collections
    M = []
    for d in a.out:
        for sh in sorted(Path(d).glob("shard_*.h5")):
            with h5py.File(sh, "r") as f5:
                M += [json.loads(m) for m in f5["meta"][:]]
    pct = lambda v: " / ".join(f"{x:.1f}" for x in np.percentile(v, [5, 25, 50, 75, 95]))
    src = lambda m: m["phantom"].split("/")[0]
    print(f"{len(M)} frames; by source/view: {dict(collections.Counter(src(m) + ':' + m['view'] for m in M))}")
    print(f"phantoms: {len({m['phantom'] for m in M})}, lesion phantoms: {len({m['phantom'] for m in M if m['view'] == 'lesion'})}")
    for name, sel in [("all", lambda m: True)] + [(k, lambda m, k=k: src(m) == k) for k in ("oat64", "usct", "victre")]:
        L = [m for m in M if m["view"] == "lesion" and sel(m)]
        if not L:
            continue
        area = np.array([m["lesion_area_mm2"] for m in L])
        w, h = np.array([m["lesion_w_mm"] for m in L]), np.array([m["lesion_h_mm"] for m in L])
        print(f"[{name}] {len(L)} lesion frames (5/25/50/75/95 %)\n  area mm2 {pct(area)}\n"
              f"  eq. diameter mm {pct(2 * np.sqrt(area / np.pi))}\n  width/height {pct(w / h)}, wider-than-tall "
              f"{(w > h).mean():.2f}\n  top depth mm {pct([m['lesion_top_mm'] for m in L])}\n"
              f"  centre depth mm {pct([m['lesion_center_mm'] for m in L])}\n"
              f"  gray lesion {pct([m['gray_lesion'] for m in L])}, tissue {pct([m['gray_tissue'] for m in L])}\n"
              f"  lesion types {dict(collections.Counter(t for m in L for t in m['lesion_types']))}")
    print(f"image mean gray {pct([m['gray_mean'] for m in M])}; renders/frame mean "
          f"{np.mean([m['attempts'] for m in M]):.2f}; |tilt| or |rock| > 12 deg: "
          f"{sum(max(abs(m['pose']['tilt']), abs(m['pose']['rock'])) > 12 for m in M)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("plan")
    pl.add_argument("--out", required=True)
    pl.add_argument("--phantoms", required=True, help="text file, one phantom id per line")
    pl.add_argument("--n", type=int, default=10000)
    pl.add_argument("--pos-frac", type=float, default=0.8)
    pl.add_argument("--preset", default="breast")
    pl.add_argument("--min-area", type=float, default=4.0, help="mm2 of lesion in a lesion frame")
    pl.add_argument("--depth-scale", type=float, default=12.0, help="mm, preference for shallow lesion placements")
    pl.add_argument("--seed", type=int, default=0)
    rd = sub.add_parser("render")
    rd.add_argument("--out", required=True)
    rd.add_argument("--shard", type=int, default=0)
    rd.add_argument("--shards", type=int, default=1)
    rd.add_argument("--interleave", action="store_true", help="every k-th frame (small test batches)")
    rd.add_argument("--max-tries", type=int, default=20)
    sh = sub.add_parser("sheet")
    sh.add_argument("--out", required=True)
    sh.add_argument("--shard", type=int, default=-1)
    sh.add_argument("--n", type=int, default=24)
    sh.add_argument("--png")
    st = sub.add_parser("stats")
    st.add_argument("--out", nargs="+", required=True, help="dataset directories")
    a = ap.parse_args()
    {"plan": cmd_plan, "render": cmd_render, "sheet": cmd_sheet, "stats": cmd_stats}[a.cmd](a)


if __name__ == "__main__":
    main()
