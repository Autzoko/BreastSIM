#!/usr/bin/env python
"""Anteroposterior (supine, ABUS / probe-like) compression of a VICTRE phantom with FEBio.

    python src/scripts/compress_ap.py --src $BREASTSIM_BIG/victre/hetero_sup4 --dst $BREASTSIM_BIG/victre/hetero_sup4_c25 \
        --thickness 25

The VICTRE label volume (p_<seed> and, with typed lesions, pcl_<seed>) is never modified; a new phantom
folder is written. Model (VICTRE breastCompress's own materials and contact, see _ai notes):
- voxel (hex8) mesh of the breast + muscle at --mesh-mm (block-majority material; a marching cubes + TetGen
  mesh of these flat-sided shapes gave degenerate slivers and FEBio "negative jacobian");
- neo-Hookean, nu 0.49: fat / skin 5 kPa, glandular (with TDLU / duct) 15 kPa, mass (+ calcifications)
  20 kPa (breastCompress.cxx); muscle elements fixed (chest wall);
- a rigid plate (ABUS membrane / probe face) above the nipple, frictionless facet-to-facet sliding contact,
  moved toward the chest wall until the nipple-to-muscle distance is --thickness mm; FEBio (src/docker/febio).
- the nodal displacement field (on the hex lattice) is inverted by fixed-point iteration X = x - u(X) on that
  lattice, X is interpolated to the voxels, and every
  label volume is resampled (nearest) at X: all tissues, ligaments and lesions move with the same field, so
  lesion masks stay exact; lesion centres in insertion.json are mapped forward.
"""
import argparse
import gzip
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

FAT, SKIN, GLAND, TDLU, DUCT, MUSCLE, MASS, CALC = 1, 2, 29, 95, 125, 40, 200, 250
E_KPA = {"fat": 5.0, "gland": 15.0, "mass": 20.0}          # breastCompress.cxx


def read_mhd(p):
    return dict(l.split(" = ", 1) for l in p.read_text().strip().splitlines())


def read_raw(raw, hdr):
    nx, ny, nz = map(int, hdr["DimSize"].split())
    return np.frombuffer(gzip.open(raw).read(), np.uint8).reshape(nz, ny, nx)


def ap_axis(lab):
    """Chest-wall axis and side: the end slice with the largest tissue cross-section (as phantom._orient)."""
    best = None
    for ax in range(3):
        occ = (lab > 0).mean(axis=tuple(a for a in range(3) if a != ax))
        idx = np.nonzero(occ)[0]
        lo, hi = occ[idx[0]], occ[idx[-1]]
        if best is None or max(lo, hi) > best[0]:
            best = (max(lo, hi), ax, hi > lo)
    return best[1], best[2]                       # axis, True if the chest wall is at the high end


def hex_mesh(lab, vox, h):
    """Voxel (hex8) mesh of the tissue at h mm: one element per h-block that is mostly tissue, material from
    the block's majority label class. Returns nodes (mm, work frame), elements (n, 8), class per element
    (fat / gland / mass / muscle) and the top boundary faces (facing +AP, quad4) for the plate contact."""
    k = max(1, int(round(h / vox)))
    s = [n // k * k for n in lab.shape]
    blk = lab[:s[0], :s[1], :s[2]].reshape(s[0] // k, k, s[1] // k, k, s[2] // k, k).transpose(0, 2, 4, 1, 3, 5)
    blk = blk.reshape(*blk.shape[:3], -1)
    occ = (blk > 0).mean(-1) > 0.5
    cls = np.zeros(occ.shape, np.uint8)                                   # 1 fat, 2 gland, 3 mass, 4 muscle
    for c, labs in ((2, (GLAND, TDLU, DUCT)), (3, (MASS, CALC)), (4, (MUSCLE, 45))):
        cls[(np.isin(blk, labs).mean(-1) > 0.5) & occ] = c
    cls[occ & (cls == 0)] = 1
    idx = np.argwhere(occ)
    nid = -np.ones(np.array(occ.shape) + 1, np.int64)
    corners = [(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1), (1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]
    for c in corners:
        q = idx + c
        nid[q[:, 0], q[:, 1], q[:, 2]] = 0
    used = np.argwhere(nid == 0)
    nid[used[:, 0], used[:, 1], used[:, 2]] = np.arange(len(used))
    nodes = used * (k * vox)
    ele = np.stack([nid[idx[:, 0] + a, idx[:, 1] + b, idx[:, 2] + c] for a, b, c in corners], 1)
    top = idx[np.r_[~occ[1:], np.ones((1,) + occ.shape[1:], bool)][idx[:, 0], idx[:, 1], idx[:, 2]]]
    faces = np.stack([nid[top[:, 0] + 1, top[:, 1] + b, top[:, 2] + c] for b, c in ((0, 0), (1, 0), (1, 1), (0, 1))], 1)
    return nodes, ele, cls[occ], faces


def write_feb(path, nodes, ele, mats, face, fixed, plate, disp, steps=20):
    """FEBio input (spec 2.0, the layout of VICTRE breastCompress)."""
    pn, ph = plate
    L = ['<?xml version="1.0" encoding="ISO-8859-1"?>', '<febio_spec version="2.0">', '\t<Module type="solid"/>',
         "\t<Control>", f"\t\t<time_steps>{steps}</time_steps>", f"\t\t<step_size>{1 / steps}</step_size>",
         "\t\t<max_refs>15</max_refs>", "\t\t<max_ups>10</max_ups>", "\t\t<dtol>0.001</dtol>", "\t\t<etol>0.01</etol>",
         "\t\t<rtol>0</rtol>", "\t\t<lstol>0.9</lstol>", "\t\t<time_stepper>", f"\t\t\t<dtmin>{0.02 / steps}</dtmin>",
         f"\t\t\t<dtmax>{1 / steps}</dtmax>", "\t\t\t<max_retries>10</max_retries>", "\t\t\t<opt_iter>10</opt_iter>",
         "\t\t</time_stepper>", '\t\t<analysis type="static"/>', "\t</Control>", "\t<Material>"]
    for i, (name, e) in enumerate((("fat", E_KPA["fat"]), ("gland", E_KPA["gland"]), ("mass", E_KPA["mass"])), 1):
        L += [f'\t\t<material id="{i}" name="{name}" type="neo-Hookean">', "\t\t\t<density>1e-06</density>",
              f"\t\t\t<E>{e}</E>", "\t\t\t<v>0.49</v>", "\t\t</material>"]
    L += ['\t\t<material id="4" name="plate" type="rigid body">', "\t\t\t<density>1e-05</density>", "\t\t</material>",
          "\t</Material>", "\t<Geometry>", "\t\t<Nodes>"]
    L += [f'\t\t\t<node id="{i + 1}"> {p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}</node>' for i, p in enumerate(pn)]
    off = len(pn)
    L += [f'\t\t\t<node id="{off + i + 1}"> {p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}</node>' for i, p in enumerate(nodes)]
    L += ["\t\t</Nodes>", '\t\t<Elements type="hex8" mat="4" elset="plate">']
    L += [f'\t\t\t<elem id="{i + 1}"> ' + ", ".join(str(n + 1) for n in h) + "</elem>" for i, h in enumerate(ph)]
    L += ["\t\t</Elements>"]
    eid = len(ph)
    for mid, name in ((1, "fat"), (2, "gland"), (3, "mass")):
        sel = np.nonzero(mats == mid)[0]
        if not len(sel):
            continue
        L += [f'\t\t<Elements type="hex8" mat="{mid}" elset="{name}">']
        L += [f'\t\t\t<elem id="{eid + j + 1}"> ' + ", ".join(str(off + n + 1) for n in ele[i]) + "</elem>"
              for j, i in enumerate(sel)]
        L += ["\t\t</Elements>"]
        eid += len(sel)
    L += ["\t</Geometry>", "\t<Boundary>", '\t\t<fix bc="xyz">']
    L += [f'\t\t\t<node id="{off + n + 1}"/>' for n in fixed]
    L += ["\t\t</fix>", "\t</Boundary>", "\t<Contact>", '\t\t<contact type="facet-to-facet sliding">',
          "\t\t\t<laugon>0</laugon>", "\t\t\t<tolerance>1.0</tolerance>", "\t\t\t<penalty>25</penalty>",
          "\t\t\t<two_pass>0</two_pass>", "\t\t\t<auto_penalty>1</auto_penalty>", "\t\t\t<fric_coeff>0</fric_coeff>",
          "\t\t\t<search_tol>0.02</search_tol>", '\t\t\t<surface type="master">']
    # plate face toward the breast: hex8 nodes 0-3 (bottom layer)
    L += [f'\t\t\t\t<quad4 id="{i + 1}"> ' + ", ".join(str(n + 1) for n in h[[0, 3, 2, 1]]) + "</quad4>"
          for i, h in enumerate(ph)]
    L += ["\t\t\t</surface>", '\t\t\t<surface type="slave">']
    L += [f'\t\t\t\t<quad4 id="{i + 1}"> ' + ", ".join(str(off + n + 1) for n in t) + "</quad4>" for i, t in enumerate(face)]
    L += ["\t\t\t</surface>", "\t\t</contact>", "\t</Contact>", "\t<Constraints>", '\t\t<rigid_body mat="4">',
          '\t\t\t<fixed bc="y"/>', '\t\t\t<fixed bc="z"/>', '\t\t\t<fixed bc="Rx"/>', '\t\t\t<fixed bc="Ry"/>',
          '\t\t\t<fixed bc="Rz"/>', '\t\t\t<prescribed bc="x" lc="1">1</prescribed>', "\t\t</rigid_body>",
          "\t</Constraints>", "\t<LoadData>", '\t\t<loadcurve id="1" type="linear">', "\t\t\t<point>0,0</point>",
          f"\t\t\t<point>1,{disp:.4f}</point>", "\t\t</loadcurve>", "\t</LoadData>", "\t<Output>",
          '\t\t<logfile>', '\t\t\t<node_data data="ux;uy;uz" file="disp.txt"/>', "\t\t</logfile>",
          '\t\t<plotfile type="febio">', '\t\t\t<var type="displacement"/>', "\t\t</plotfile>", "\t</Output>",
          "</febio_spec>"]
    path.write_text("\n".join(L) + "\n")


def plate_mesh(lo, hi, x0, step=4.0, t=1.0):
    """Rigid plate: one layer of hex8 over the (b, c) rectangle lo-hi at height x0 (axis 0 = AP)."""
    b = np.arange(lo[0], hi[0] + step, step)
    c = np.arange(lo[1], hi[1] + step, step)
    B, C = np.meshgrid(b, c, indexing="ij")
    nb, nc = B.shape
    pts = np.concatenate([np.stack([np.full(B.size, x0 + dz), B.ravel(), C.ravel()], 1) for dz in (0.0, t)])
    idx = lambda i, j, k: k * nb * nc + i * nc + j
    hexes = [[idx(i, j, 0), idx(i + 1, j, 0), idx(i + 1, j + 1, 0), idx(i, j + 1, 0),
              idx(i, j, 1), idx(i + 1, j, 1), idx(i + 1, j + 1, 1), idx(i, j + 1, 1)]
             for i in range(nb - 1) for j in range(nc - 1)]
    return pts, np.array(hexes)


def read_disp(path, n_plate, n_nodes):
    """Last record of the FEBio node_data log: rows 'id ux uy uz' for every node."""
    txt = Path(path).read_text()
    last = txt[txt.rfind("*Data"):].split("\n", 1)[1]                    # block of the final step
    a = np.loadtxt(last.splitlines()[:n_plate + n_nodes])
    return a[n_plate:, 1:4]


def inverse_resample(labs, vox, nodes, disp, grid_mm, iters=20, full=False, device="cpu", solve_mm=0.5):
    """Labels at X with X + u(X) = x for every output voxel x (same array frame, axis 0 = AP).

    The fixed point X = x - u(X) is solved on a solve_mm grid and X is interpolated (trilinear) to the voxels
    (the inverse map is as smooth as u); full=True iterates at every voxel instead (slow reference)."""
    import torch
    import torch.nn.functional as F
    shape = labs[0].shape
    ext = np.array(shape) * vox
    # the hex mesh nodes lie on a regular grid_mm lattice: the displacement is that lattice (trilinear inside
    # the elements, as the hex8 shape functions), extended outside the tissue by its nearest node
    gs = [np.arange(0, e + grid_mm, grid_mm) for e in ext]
    lat = np.round(nodes / grid_mm).astype(int)
    U = np.full((*[len(g) for g in gs], 3), np.nan, np.float32)
    ok = (lat >= 0).all(1) & (lat < [len(g) for g in gs]).all(1)
    U[lat[ok, 0], lat[ok, 1], lat[ok, 2]] = disp[ok]
    miss = np.isnan(U[..., 0])
    _, ind = ndimage.distance_transform_edt(miss, return_indices=True)
    U = U[ind[0], ind[1], ind[2]]
    gsol = [np.arange(0, e + solve_mm, solve_mm) for e in ext]
    G = np.stack(np.meshgrid(*gsol, indexing="ij"), -1)
    smax = torch.tensor([g[-1] for g in gsol], dtype=torch.float32, device=device)
    ug = torch.from_numpy(U.transpose(3, 0, 1, 2).copy()).float()[None]
    dev = torch.device(device)
    ug = ug.to(dev)
    outs = [np.zeros(shape, np.uint8) for _ in labs]
    srcs = [torch.from_numpy(l).to(dev)[None, None] for l in labs]                    # uint8 on the device
    gmax = torch.tensor([g[-1] for g in gs], dtype=torch.float32, device=dev)
    lmax = torch.tensor([(n - 1) * vox for n in shape], dtype=torch.float32, device=dev)

    def solve(x):                                                                     # fixed point at points x
        X = x.clone()
        for _ in range(iters):
            g = (X / gmax * 2 - 1).flip(-1)[None]                                        # grid_sample (w, h, d)
            uX = F.grid_sample(ug, g, mode="bilinear", padding_mode="border", align_corners=True)[0]
            X = x - uX.permute(1, 2, 3, 0)
        return X

    if not full:
        Gt = torch.from_numpy(G).float().to(dev)
        Xg = solve(Gt).permute(3, 0, 1, 2)[None]                                          # (1, 3, *grid)
    b = torch.arange(shape[1], device=dev, dtype=torch.float32) * vox
    c = torch.arange(shape[2], device=dev, dtype=torch.float32) * vox
    step = 16 if full else 32
    for a0 in range(0, shape[0], step):
        a = torch.arange(a0, min(a0 + step, shape[0]), device=dev, dtype=torch.float32) * vox
        x = torch.stack(torch.meshgrid(a, b, c, indexing="ij"), -1)                       # (na, nb, nc, 3)
        if full:
            X = solve(x)
        else:
            g = (x / smax * 2 - 1).flip(-1)[None]
            X = F.grid_sample(Xg, g, mode="bilinear", padding_mode="border", align_corners=True)[0].permute(1, 2, 3, 0)
        inside = ((X >= 0) & (X <= lmax)).all(-1)
        idx = torch.round(X / vox).long()
        for d in range(3):
            idx[..., d].clamp_(0, shape[d] - 1)
        for src, out in zip(srcs, outs):
            v = src[0, 0][idx[..., 0], idx[..., 1], idx[..., 2]]
            out[a0:a0 + len(a)] = torch.where(inside, v, torch.zeros_like(v)).cpu().numpy()
    def forward(p):                                                                   # u at points p (mm)
        g = torch.from_numpy((np.asarray(p, np.float64) / np.array([g[-1] for g in gs]) * 2 - 1)[:, ::-1]
                             .astype(np.float32).copy())
        return F.grid_sample(ug.cpu(), g.reshape(1, -1, 1, 1, 3), mode="bilinear", padding_mode="border",
                             align_corners=True)[0, :, :, 0, 0].T.numpy()
    return outs, forward


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="VICTRE variant folder (<src>/<seed>/p_<seed>.raw.gz ...)")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--thickness", type=float, default=25.0, help="target nipple-to-muscle distance, mm")
    ap.add_argument("--mesh-mm", type=float, default=2.0, help="hex8 (voxel) element size, mm")
    ap.add_argument("--lateral-pad-mm", type=float, default=15.0, help="room for the lateral expansion")
    ap.add_argument("--image", default="breastsim/febio")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cpu", help="torch device for the resampling (cpu / cuda)")
    ap.add_argument("--full-inverse", action="store_true", help="fixed point at every voxel (slow reference)")
    a = ap.parse_args()
    t0 = time.time()
    src, dst = Path(a.src) / str(a.seed), Path(a.dst) / str(a.seed)
    dst.mkdir(parents=True, exist_ok=True)
    work = dst / "febio"
    work.mkdir(exist_ok=True)
    names = [n for n in (f"p_{a.seed}", f"pcl_{a.seed}") if (src / f"{n}.raw.gz").exists()]
    hdr = read_mhd(src / f"{names[-1]}.mhd")
    vox = float(hdr["ElementSpacing"].split()[0])
    labs = [read_raw(src / f"{n}.raw.gz", hdr) for n in names]
    lab = labs[-1]
    axis, high = ap_axis(lab)
    # canonical work frame: axis 0 = AP with the chest wall at index 0, lateral padding on axes 1, 2
    perm = [axis] + [i for i in range(3) if i != axis]
    pad = int(round(a.lateral_pad_mm / vox))
    to_work = lambda v: np.pad((v.transpose(perm)[::-1] if high else v.transpose(perm)), ((0, 0), (pad, pad), (pad, pad)))
    W = [to_work(l) for l in labs]
    lw = W[-1]
    tis = lw > 0
    musc = lw == MUSCLE
    top = np.where(tis.any(0), lw.shape[0] - 1 - np.argmax(tis[::-1], 0), -1)            # skin height per column
    mtop = np.where(musc.any(0), lw.shape[0] - 1 - np.argmax(musc[::-1], 0), -1)
    apex = top.max() * vox
    muscle_top = np.median(mtop[mtop >= 0]) * vox
    disp = -(apex - (muscle_top + a.thickness))
    print(f"[compress] {names}, voxel {vox}, apex {apex:.1f} mm, muscle top {muscle_top:.1f} mm, "
          f"plate displacement {disp:.1f} mm", flush=True)
    nodes, ele, ecls, face = hex_mesh(lw, vox, a.mesh_mm)
    mats = np.where(ecls == 4, 1, ecls)                                   # muscle elements: fat material, fixed
    fixed = np.unique(ele[ecls == 4])
    fixed = np.union1d(fixed, np.nonzero(nodes[:, 0] < 1.0)[0])                         # back plane
    print(f"[compress] hex mesh {a.mesh_mm} mm: {len(nodes)} nodes, {len(ele)} elements "
          f"(fat {int((ecls == 1).sum())}, gland {int((ecls == 2).sum())}, mass {int((ecls == 3).sum())}, "
          f"muscle {int((ecls == 4).sum())}), {len(face)} contact faces", flush=True)
    lo, hi = nodes[:, 1:].min(0) - 10, nodes[:, 1:].max(0) + 10
    pn, ph = plate_mesh(lo, hi, nodes[:, 0].max() + 0.5)
    write_feb(work / "model.feb", nodes, ele, mats, face, fixed, (pn, ph), disp)
    t1 = time.time()
    done = lambda: (work / "model.log").exists() and "N O R M A L   T E R M I N A T I O N" in (work / "model.log").read_text()
    if not done():                                         # (a finished run in this folder is reused)
        r = subprocess.run(["docker", "run", "--rm", "-u", f"{os.getuid()}:{os.getgid()}", "-e",
                            f"OMP_NUM_THREADS={a.threads}", "--cpus", str(a.threads), "-v", f"{work.resolve()}:/work",
                            "-w", "/work", a.image, "febio4", "-i", "model.feb", "-silent"], capture_output=True, text=True)
        (work / "febio_stdout.txt").write_text(r.stdout + r.stderr)
    if not done():
        raise SystemExit(f"FEBio did not terminate normally, see {work}/model.log")
    u = read_disp(work / "disp.txt", len(pn), len(nodes))
    t2 = time.time()
    t3 = time.time()
    outs, forward = inverse_resample(W, vox, nodes, u, grid_mm=max(1, int(round(a.mesh_mm / vox))) * vox,
                                     full=a.full_inverse, device=a.device)
    print(f"[compress] resampling {time.time() - t3:.0f} s ({'full' if a.full_inverse else 'grid'} inverse, {a.device})",
          flush=True)
    # back to the source orientation and size (the lateral padding stays: the breast widens)
    back = lambda v: (v[::-1] if high else v).transpose(np.argsort(perm))
    shp = None
    for n, o in zip(names, outs):
        o = np.ascontiguousarray(back(o))
        shp = o.shape
        with gzip.open(dst / f"{n}.raw.gz", "wb", compresslevel=1) as fh:
            fh.write(o.tobytes())
        h = dict(hdr)
        h["DimSize"] = f"{o.shape[2]} {o.shape[1]} {o.shape[0]}"
        h["ElementDataFile"] = f"{n}.raw"
        (dst / f"{n}.mhd").write_text("".join(f"{k} = {v}\n" for k, v in h.items()))
    gen = json.loads((src / "generation.json").read_text())
    gen["compression"] = {"source": str(src), "thickness_mm": a.thickness, "plate_displacement_mm": round(disp, 2),
                          "mesh_mm": a.mesh_mm, "nodes": len(nodes), "tets": len(ele), "E_kPa": E_KPA, "nu": 0.49,
                          "lateral_pad_mm": a.lateral_pad_mm, "febio_seconds": round(t2 - t1, 1),
                          "total_seconds": round(time.time() - t0, 1)}
    (dst / "generation.json").write_text(json.dumps(gen, indent=1))
    if (src / "insertion.json").exists():                      # lesion centres follow the same field
        ins = json.loads((src / "insertion.json").read_text())
        for rec in ins.get("lesions", []):
            c = np.asarray(rec["center_mm_zyx"]) / vox
            cw = np.array([c[i] for i in perm])
            if high:
                cw[0] = lab.shape[axis] - 1 - cw[0]
            cw = cw * vox + [0, pad * vox, pad * vox]
            cw2 = cw + forward(cw[None])[0]
            cw2 = cw2 / vox
            if high:
                cw2[0] = lab.shape[axis] - 1 - cw2[0]
            out = np.empty(3)
            out[perm] = cw2
            rec["center_mm_zyx"] = [round(float(x * vox), 2) for x in out]
        (dst / "insertion.json").write_text(json.dumps(ins, indent=1))
    print(json.dumps(gen["compression"]))


if __name__ == "__main__":
    main()
