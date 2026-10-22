#!/usr/bin/env python
"""Build the anatomical female torso asset for the BreastSIM 3D viewer.

Source data: MakeHuman *assets* (base mesh + morph targets), which are released
under CC0 1.0 (see LICENSE.ASSETS.md / LICENSE.md in the MakeHuman repository).
No MakeHuman *code* (AGPL) is used: the OBJ and .target files are parsed here as
plain data.

Pipeline
--------
1. Download the needed CC0 files, pinned to a MakeHuman commit SHA, into a cache
   directory (downloaded files are only ever parsed as text data).
2. Parse ``base.obj`` (hm08 basemesh, units = decimetres); keep only the ``body``
   skin group (drops helper-*, joint-* and eye/teeth/tongue/hair proxies).  The
   ``joint-*`` groups are only used as landmark points for cropping.
3. Apply targets: ``caucasian-female-young`` (w=1; full female, age 25),
   ``universal-female-young-averagemuscle-averageweight`` (w=1).  The left
   breast keeps MakeHuman's default (average cup / average firmness = no
   breast target).  On the phantom side the ``mincup-averagefirmness`` breast
   target is applied (restricted to that side), then the remaining breast
   mound is removed: a thin-plate spline height field (over the plane fitted to
   a 3-ring band of skin around the breast region) is fitted to that band and
   the region vertices are moved along the plane normal onto it.  That yields a
   smooth chest-wall surface continuous with the surrounding skin.
4. One level of Catmull-Clark subdivision (linear operator, so it is applied to
   the natural and flattened variants alike, keeping vertex correspondence).
5. Crop to a torso: plane cuts below the neck (head removed), at the hips and
   across each upper arm (short arm stubs kept).  Vertices of kept faces that
   lie beyond a cut plane are snapped onto it to give clean cut edges.
6. Triangulate, recompute normals, compute anchors, convert to millimetres and
   write ``torso.json`` + ``torso_LICENSE.md``.

Frame (MakeHuman, verified from the mesh): +y up (toward the head), +z anterior
(the face points +z), +x toward the subject's LEFT.  Origin = MakeHuman origin
(about at the hips), scaled to mm.

Usage::

    python src/scripts/build_torso_asset.py [--side right|left] [--cache DIR]
        [--out-dir src/breastsim/realtime/static/assets] [--preview DIR]
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

MH_REPO = "makehumancommunity/makehuman"
MH_SHA = "a8bc2d54ff0ac92e78ff71431b1023eda42bf482"
RAW = f"https://raw.githubusercontent.com/{MH_REPO}/{MH_SHA}/"

FILES = {
    "base.obj": "makehuman/data/3dobjs/base.obj",
    "caucasian-female-young.target": "makehuman/data/targets/macrodetails/caucasian-female-young.target",
    "universal-female-young-averagemuscle-averageweight.target":
        "makehuman/data/targets/macrodetails/universal-female-young-averagemuscle-averageweight.target",
    "female-young-averagemuscle-averageweight-mincup-averagefirmness.target":
        "makehuman/data/targets/breast/female-young-averagemuscle-averageweight-mincup-averagefirmness.target",
    "female-young-averagemuscle-averageweight-maxcup-averagefirmness.target":
        "makehuman/data/targets/breast/female-young-averagemuscle-averageweight-maxcup-averagefirmness.target",
    "nipple-point-incr.target": "makehuman/data/targets/breast/nipple-point-incr.target",
    "LICENSE.md": "LICENSE.md",
    "LICENSE.ASSETS.md": "LICENSE.ASSETS.md",
}

# Body-shape targets applied to the whole mesh: (file, weight)
BODY_TARGETS = [
    ("caucasian-female-young.target", 1.0),
    ("universal-female-young-averagemuscle-averageweight.target", 1.0),
]
# Applied only on the phantom side, before the masked smoothing.
FLAT_SIDE_TARGETS = [
    ("female-young-averagemuscle-averageweight-mincup-averagefirmness.target", 1.0),
]
DM_TO_MM = 100.0

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "src/breastsim/realtime/static/assets"
DEFAULT_CACHE = Path(os.environ.get("MH_CACHE", Path.home() / ".cache/breastsim/makehuman"))


# --------------------------------------------------------------------------- io
def fetch(cache: Path) -> dict[str, Path]:
    cache.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, rel in FILES.items():
        p = cache / name
        if not p.exists() or p.stat().st_size == 0:
            url = RAW + rel
            print(f"download {url}")
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
            p.write_bytes(data)
        out[name] = p
    return out


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_obj(path: Path):
    """Return verts (N,3), {group: faces (M,4) int}, all 0-based."""
    V, groups, g = [], {}, None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("v "):
                V.append([float(t) for t in line.split()[1:4]])
            elif line.startswith("g "):
                g = line.split()[1]
            elif line.startswith("f "):
                idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
                groups.setdefault(g, []).append(idx)
    return np.asarray(V, float), {k: np.asarray(v, int) for k, v in groups.items()}


def load_target(path: Path, nverts: int) -> np.ndarray:
    d = np.zeros((nverts, 3))
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            p = s.split()
            d[int(p[0])] = [float(x) for x in p[1:4]]
    return d


# ---------------------------------------------------------------- mesh helpers
def quad_edges(Q):
    e = np.concatenate([Q[:, [0, 1]], Q[:, [1, 2]], Q[:, [2, 3]], Q[:, [3, 0]]])
    return np.sort(e, axis=1)


def adjacency(n, Q):
    e = np.unique(quad_edges(Q), axis=0)
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    A = (A + A.T).tocsr()
    A.data[:] = 1.0
    return A


def catmull_clark_operator(n, Q):
    """Sparse S with V' = S @ V for one Catmull-Clark step on a quad mesh.

    Returns (S, new_quads).  New vertex order: [original verts | face points |
    edge points].  Boundary edges/vertices use the standard crease rules.
    """
    F = len(Q)
    E_all = quad_edges(Q)  # (4F,2), edge k of face f at row k*F+f
    E, inv, cnt = np.unique(E_all, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    nE = len(E)
    boundary_e = cnt == 1
    rows, cols, vals = [], [], []

    def add(r, c, v):
        rows.append(np.asarray(r)); cols.append(np.asarray(c)); vals.append(np.asarray(v, float))

    # face points
    fp = n + np.arange(F)
    for k in range(4):
        add(fp, Q[:, k], np.full(F, 0.25))
    # edge points
    ep0 = n + F
    # endpoints
    w_end = np.where(boundary_e, 0.5, 0.25)
    add(ep0 + np.arange(nE), E[:, 0], w_end)
    add(ep0 + np.arange(nE), E[:, 1], w_end)
    # adjacent face centroids (interior edges only): 0.25 * facepoint = 1/16 each vertex
    face_of = np.tile(np.arange(F), 4)
    inter = ~boundary_e[inv]
    for k in range(4):
        add(ep0 + inv[inter], Q[face_of[inter], k], np.full(inter.sum(), 0.25 * 0.25))

    # original vertices
    # valence & boundary status
    deg = np.bincount(E.ravel(), minlength=n).astype(float)
    bE = E[boundary_e]
    on_b = np.zeros(n, bool)
    on_b[bE.ravel()] = True
    interior_v = ~on_b & (deg > 0)
    # interior: (F_avg + 2 R_avg + (k-3) P) / k ; F_avg = mean of adjacent face points,
    # R_avg = mean of edge midpoints.
    vi = np.where(interior_v)[0]
    k = deg
    add(vi, vi, (k[vi] - 3) / k[vi])
    # 2*R/k: R = mean_e (P + Pj)/2 -> sum over edges (P+Pj)/(2k) ; times 2/k
    ie = ~boundary_e
    for a, b in [(0, 1), (1, 0)]:
        va, vb = E[ie, a], E[ie, b]
        m = interior_v[va]
        add(va[m], va[m], 1.0 / k[va[m]] ** 2)
        add(va[m], vb[m], 1.0 / k[va[m]] ** 2)
    # F/k: F = mean of adjacent face points (k faces for interior vertex)
    for c in range(4):
        v = Q[:, c]
        m = interior_v[v]
        for j in range(4):
            add(v[m], Q[m, j], 0.25 / k[v[m]] ** 2)
    # boundary vertices: 3/4 P + 1/8 (two boundary neighbours)
    bv = np.where(on_b)[0]
    add(bv, bv, np.full(len(bv), 0.75))
    for a, b in [(0, 1), (1, 0)]:
        add(bE[:, a], bE[:, b], np.full(len(bE), 0.125))
    # isolated (unused) verts: keep
    iso = np.where(deg == 0)[0]
    add(iso, iso, np.ones(len(iso)))

    N = n + F + nE
    S = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(N, n)).tocsr()
    e_id = inv.reshape(4, F).T  # (F,4): edge (k,k+1)
    f_id = n + np.arange(F)
    newQ = []
    for c in range(4):
        e_next = ep0 + e_id[:, c]           # edge c -> c+1
        e_prev = ep0 + e_id[:, (c - 1) % 4]  # edge c-1 -> c
        newQ.append(np.stack([Q[:, c], e_next, f_id, e_prev], 1))
    return S, np.concatenate(newQ)


def tri_normals(V, T):
    return np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])


def vertex_normals(V, T):
    fn = tri_normals(V, T)  # area weighted
    N = np.zeros_like(V)
    for k in range(3):
        np.add.at(N, T[:, k], fn)
    return N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)


def thin_plate_flatten(V, A, region, band_rings=3, smoothing=1e-4):
    """Replace the breast mound by a smooth chest-wall surface.

    The region and a band of ``band_rings`` vertex rings around it are expressed in a
    local frame (u, v, h) whose h axis is the normal of the plane fitted to the band.
    A thin-plate-spline h(u, v) is fitted to the band (the undisturbed skin around the
    breast) and evaluated at the region vertices, whose (u, v) are kept.  The result is
    independent of the (very non-uniform) vertex density around the nipple.
    """
    from scipy.interpolate import RBFInterpolator

    band = region.copy()
    for _ in range(band_rings):
        band = band | ((A @ band.astype(float)) > 0)
    band &= ~region
    B = V[band]
    c = B.mean(0)
    _, _, vt = np.linalg.svd(B - c, full_matrices=False)
    n0 = vt[2] if vt[2][2] > 0 else -vt[2]  # plane normal, oriented anterior
    e1 = vt[0] - (vt[0] @ n0) * n0
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n0, e1)
    def to_local(X):
        d = X - c
        return np.stack([d @ e1, d @ e2], 1), d @ n0
    uvB, hB = to_local(B)
    rbf = RBFInterpolator(uvB, hB, kernel="thin_plate_spline", smoothing=smoothing)
    uvR, hR = to_local(V[region])
    out = V.copy()
    out[region] = V[region] + (rbf(uvR) - hR)[:, None] * n0
    return out


def closest_point_on_tris(p, A, B, C):
    """Vectorised closest point from p to triangles (A,B,C each (M,3))."""
    ab, ac, ap = B - A, C - A, p - A
    d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
    bp = p - B
    d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
    cp = p - C
    d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    denom = np.where(np.abs(va + vb + vc) < 1e-30, 1e-30, va + vb + vc)
    v = vb / denom
    w = vc / denom
    res = A + ab * v[:, None] + ac * w[:, None]
    # region tests (Ericson)
    def setm(m, val):
        res[m] = val[m]
    m = (d1 <= 0) & (d2 <= 0); setm(m, A)
    m2 = (d3 >= 0) & (d4 <= d3); setm(m2 & ~m, B)
    m3 = (d6 >= 0) & (d5 <= d6); setm(m3 & ~m & ~m2, C)
    done = m | m2 | m3
    t = d1 / np.where(d1 - d3 == 0, 1e-30, d1 - d3)
    mm = (vc <= 0) & (d1 >= 0) & (d3 <= 0) & ~done
    res[mm] = (A + ab * t[:, None])[mm]; done |= mm
    t = d2 / np.where(d2 - d6 == 0, 1e-30, d2 - d6)
    mm = (vb <= 0) & (d2 >= 0) & (d6 <= 0) & ~done
    res[mm] = (A + ac * t[:, None])[mm]; done |= mm
    t = (d4 - d3) / np.where((d4 - d3) + (d5 - d6) == 0, 1e-30, (d4 - d3) + (d5 - d6))
    mm = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0) & ~done
    res[mm] = (B + (C - B) * t[:, None])[mm]
    d = np.linalg.norm(res - p, axis=1)
    i = int(np.argmin(d))
    return res[i], i, d[i]


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


# ----------------------------------------------------------------------- main
def build(args):
    paths = fetch(Path(args.cache))
    V0, groups = load_obj(paths["base.obj"])
    n = len(V0)
    Q = groups["body"]
    assert Q.shape[1] == 4

    # body shape
    V = V0.copy()
    for name, w in BODY_TARGETS:
        V += w * load_target(paths[name], n)
    joints = {g[6:]: V[np.unique(f)].mean(0) for g, f in groups.items() if g.startswith("joint-")}
    used = np.unique(Q)
    height_dm = V[used, 1].max() - V[used, 1].min()
    print(f"full-body height: {height_dm * DM_TO_MM:.0f} mm")

    # frame verification
    head_z = joints["head"][2]
    assert joints["l-shoulder"][0] > 0, "expected +x = subject's left"
    nose_like = V[used][np.argmax(V[used][:, 2] * (V[used][:, 1] > joints["neck"][1]))]
    assert nose_like[2] > head_z, "expected +z = anterior (face forward)"

    side_sign = {"left": 1.0, "right": -1.0}  # sign of x on each side of subject
    flat_side = args.side
    s_flat = side_sign[flat_side]

    # breast region (where cup-size targets act), per side
    cup_min = load_target(paths[FLAT_SIDE_TARGETS[0][0]], n)
    cup_max = load_target(paths["female-young-averagemuscle-averageweight-maxcup-averagefirmness.target"], n)
    span = np.linalg.norm(cup_max - cup_min, axis=1)
    nip_t = load_target(paths["nipple-point-incr.target"], n)
    nip_ring = np.where(np.linalg.norm(nip_t, axis=1) > 0)[0]

    A = adjacency(n, Q)
    body_mask = np.zeros(n, bool); body_mask[used] = True

    def breast_region(sign):
        side = (V[:, 0] * sign > 0) & body_mask
        m = (span > args.region_frac * span.max()) & side
        # the cup targets barely move the nipple/areola: add a disc around the nipple ring
        ring = nip_ring[V[nip_ring, 0] * sign > 0]
        c = V[ring].mean(0)
        m |= side & (np.linalg.norm(V[:, :2] - c[:2], axis=1) < args.nipple_disk) & (V[:, 2] > c[2] - 0.5)
        # dilate for a softer rim
        for _ in range(args.region_dilate):
            m = m | ((A @ m.astype(float)) > 0) & side & (np.abs(V[:, 0]) > 0.02)
        # fill holes: small components of the complement (within this side) join the region
        rest = side & ~m
        idx = np.where(rest)[0]
        ncomp, lab = sp.csgraph.connected_components(A[idx][:, idx], directed=False)
        sizes = np.bincount(lab)
        m[idx[sizes[lab] < 0.5 * sizes.max()]] = True
        return m

    # natural (pre-flattening) mesh = V; flattened variants per side
    flat = {}
    regions = {}
    for side, sgn in side_sign.items():
        Vs = V.copy()
        sidew = (Vs[:, 0] * sgn > 0).astype(float)[:, None]
        for name, w in FLAT_SIDE_TARGETS:
            Vs += w * sidew * load_target(paths[name], n)
        reg = breast_region(sgn)
        regions[side] = reg
        flat[side] = thin_plate_flatten(Vs, A, reg)
    Vout = V.copy()
    Vout[regions[flat_side]] = flat[flat_side][regions[flat_side]]
    # (vertices touched by the side-restricted min-cup target but outside the region)
    tm = (np.linalg.norm(cup_min, axis=1) > 0) & (V[:, 0] * s_flat > 0) & ~regions[flat_side]
    Vout[tm] = V[tm] + load_target(paths[FLAT_SIDE_TARGETS[0][0]], n)[tm]

    # nipple (pre-flattening): centre of the nipple ring, pushed to the skin surface
    nipples = {}
    for side, sgn in side_sign.items():
        ring = nip_ring[V[nip_ring, 0] * sgn > 0]
        c = V[ring].mean(0)
        rr = np.linalg.norm(V[ring, :2] - c[:2], axis=1).max()
        inner = np.where(body_mask & (np.linalg.norm(V[:, :2] - c[:2], axis=1) < rr) & (V[:, 2] > c[2] - 0.3))[0]
        tip = inner[np.argmax(V[inner, 2])] if len(inner) else None
        nipples[side] = V[tip] if tip is not None else c

    # ---- subdivision (linear: same operator for every variant)
    # compact to body verts first
    remap = -np.ones(n, int); remap[used] = np.arange(len(used))
    Qc = remap[Q]
    S, Q2 = catmull_clark_operator(len(used), Qc)
    variants = {"out": Vout, "nat": V, "flat_left": flat["left"], "flat_right": flat["right"]}
    W = {k: S @ v[used] for k, v in variants.items()}
    for _ in range(args.subdiv - 1):
        S2, Q2n = catmull_clark_operator(len(W["out"]), Q2)
        W = {k: S2 @ v for k, v in W.items()}
        Q2 = Q2n

    # ---- crop
    P = W["out"]
    neck, pelvis = joints["neck"], joints["pelvis"]
    # each cut: (point, normal, restrict) -> remove where dot(x - p, n) > 0 and restrict(x)
    planes = []
    y_top = neck[1] + args.neck_offset
    planes.append((np.array([0, y_top, 0.0]), np.array([0, 1.0, 0]), None))
    y_bot = pelvis[1] + args.hip_offset
    planes.append((np.array([0, y_bot, 0.0]), np.array([0, -1.0, 0]), None))

    def seg_dist(X, a, b):
        ab = b - a
        t = np.clip(((X - a) @ ab) / (ab @ ab), 0, 1)
        return np.linalg.norm(X - (a + t[:, None] * ab), axis=1)

    for lr in ("l", "r"):
        sh, el, ha = (joints[f"{lr}-{j}"] for j in ("shoulder", "elbow", "hand"))
        stub = sh + args.arm_stub * (el - sh)
        r = args.arm_radius

        def in_arm(X, stub=stub, el=el, ha=ha, r=r):
            return np.minimum(seg_dist(X, stub, el), seg_dist(X, el, ha)) < r

        planes.append((stub, unit(el - sh), in_arm))

    def beyond(X, cut):
        p0, nrm, restrict = cut
        m = ((X - p0) @ nrm) > 0
        if restrict is not None:
            m &= restrict(X)
        return m

    def _all_beyond(X, Qs, cut):  # face removed only if all 4 corners are beyond the cut
        return beyond(X, cut)[Qs].all(1)

    keep = np.ones(len(Q2), bool)
    for cut in planes:
        keep &= ~_all_beyond(P, Q2, cut)
    Qk = Q2[keep]
    # drop small disconnected pieces (e.g. hand/forearm fragments): keep largest component
    nv = len(P)
    Ak = adjacency(nv, Qk)
    ncomp, lab = sp.csgraph.connected_components(Ak, directed=False)
    vused = np.unique(Qk)
    big = np.bincount(lab[vused]).argmax()
    Qk = Qk[lab[Qk[:, 0]] == big]
    # snap kept verts lying beyond a cut plane onto that plane (masks from the output mesh)
    vk = np.unique(Qk)
    snaps = []
    for p0, nrm, restrict in planes:
        dd = (P[vk] - p0) @ nrm
        m = dd > 0
        if restrict is not None:
            # vertices of boundary faces may sit just outside the capsule: test a point pulled
            # back onto the plane side
            m &= restrict(P[vk] - np.maximum(dd, 0)[:, None] * nrm)
        snaps.append((p0, nrm, m))
    for k in W:
        X = W[k]
        for p0, nrm, m in snaps:
            dd = (X[vk[m]] - p0) @ nrm
            X[vk[m]] -= np.maximum(dd, 0)[:, None] * nrm
    rem = -np.ones(nv, int); rem[vk] = np.arange(len(vk))
    Qk = rem[Qk]
    W = {k: v[vk] * DM_TO_MM for k, v in W.items()}
    T = np.concatenate([Qk[:, [0, 1, 2]], Qk[:, [0, 2, 3]]])
    Vf = W["out"]
    Nf = vertex_normals(Vf, T)

    # ---- anchors
    body_up = np.array([0, 1.0, 0]); body_ant = np.array([0, 0, 1.0])
    anchors = {}
    for side, sgn in side_sign.items():
        nip = nipples[side] * DM_TO_MM
        Xf = W[f"flat_{side}"]
        Xn = W["nat"]
        cen = Xf[T].mean(1)
        near = np.where(np.linalg.norm(cen - nip, axis=1) < 250)[0]
        cp, ti, dist = closest_point_on_tris(nip, Xf[T[near, 0]], Xf[T[near, 1]], Xf[T[near, 2]])
        Nside = vertex_normals(Xf, T)
        nb = np.linalg.norm(Xf - cp, axis=1) < args.normal_radius_mm
        nrm = unit(Nside[nb].mean(0))
        sup = unit(body_up - (body_up @ nrm) * nrm)
        lat = unit(np.cross(sup, nrm))
        if lat[0] * sgn < 0:
            lat = -lat
        # footprint: vertices displaced by flattening by > thr
        disp = np.linalg.norm(Xn - Xf, axis=1)
        fp = disp > args.footprint_mm
        tri_fp = fp[T].all(1)
        area = 0.5 * np.linalg.norm(tri_normals(Xf, T[tri_fp]), axis=1).sum()
        r_area = float(np.sqrt(area / np.pi))
        anchors[side] = {
            "nipple": [round(float(x), 2) for x in nip],
            "chest_wall_point": [round(float(x), 2) for x in cp],
            "normal": [round(float(x), 5) for x in nrm],
            "superior": [round(float(x), 5) for x in sup],
            "lateral": [round(float(x), 5) for x in lat],
            "base_radius_mm": round(r_area, 1),
            "projection_mm": round(float(dist), 1),
            "flattened_in_mesh": side == flat_side,
        }
        print(side, json.dumps(anchors[side]))

    bbox = [Vf.min(0).round(1).tolist(), Vf.max(0).round(1).tolist()]
    asset = {
        "units": "mm",
        "frame": ("MakeHuman frame scaled to mm: +x = subject's LEFT, +y = superior (toward head), "
                  "+z = anterior (out of the chest). Right-handed. Origin = MakeHuman origin (near the hips)."),
        "source": f"MakeHuman CC0 assets @ {MH_SHA}",
        "phantom_side": flat_side,
        "anchors_doc": ("Per side (subject's right/left), mm in this frame: nipple = original (pre-flattening) "
                        "nipple tip; chest_wall_point = closest point to that nipple on the flattened chest-wall "
                        "surface (for the side that is NOT flattened in the mesh it lies under the natural breast); "
                        "normal = outward chest-wall normal there (25 mm average); superior = body_up projected "
                        "onto the tangent plane; lateral = superior x normal, sign chosen away from the midline "
                        "(normal/superior/lateral are orthonormal); base_radius_mm = equal-area radius of the skin "
                        "patch moved > 2 mm by the flattening; projection_mm = nipple to chest-wall distance."),
        "n_verts": int(len(Vf)),
        "n_tris": int(len(T)),
        "bbox_mm": bbox,
        "full_body_height_mm": round(float(height_dm * DM_TO_MM), 1),
        "verts": base64.b64encode(Vf.astype("<f4").tobytes()).decode(),
        "faces": base64.b64encode(T.astype("<u4").tobytes()).decode(),
        "normals": base64.b64encode(Nf.astype("<f4").tobytes()).decode(),
        "anchors": {
            "body_up": body_up.tolist(),
            "body_anterior": body_ant.tolist(),
            **anchors,
        },
    }
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    jp = out_dir / "torso.json"
    jp.write_text(json.dumps(asset, separators=(",", ":")))
    print(f"wrote {jp} ({jp.stat().st_size / 1e6:.2f} MB), {len(Vf)} verts, {len(T)} tris, bbox {bbox}")
    write_license(out_dir / "torso_LICENSE.md", paths, args, asset)
    if args.preview:
        render_previews(Path(args.preview), Vf, T, asset["anchors"])
    return asset


def write_license(path, paths, args, asset):
    lines = [
        "# torso.json - attribution and provenance",
        "",
        "The torso mesh in `torso.json` is derived from **MakeHuman** assets, which the MakeHuman",
        "team released under **CC0 1.0 Universal** (public domain dedication) in September 2020",
        "(see `LICENSE.md` section C and `LICENSE.ASSETS.md` in the MakeHuman repository).",
        "Only assets (mesh/target data files) were used; no MakeHuman program code (AGPL) was",
        "used, copied or linked. Attribution is not legally required under CC0 but is given here.",
        "",
        f"- Source: https://github.com/{MH_REPO}",
        f"- Commit: `{MH_SHA}`",
        "- Original copyright (released to CC0): Manuel Bastioni, Data Collection AB, Joel Palmius,",
        "  Jonas Hauquier and the MakeHuman team.",
        "",
        "## Files used (sha256)",
        "",
    ]
    for name, rel in FILES.items():
        lines.append(f"- `{rel}` - `{sha256(paths[name])}`")
    lines += ["", "## Targets applied", ""]
    for name, w in BODY_TARGETS:
        lines.append(f"- `{name}` weight {w} (whole body)")
    for name, w in FLAT_SIDE_TARGETS:
        lines.append(f"- `{name}` weight {w} (subject's {args.side} side only)")
    lines += [
        "- Opposite breast: MakeHuman default (average cup, average firmness; no breast target).",
        "- `maxcup-averagefirmness` and `nipple-point-incr` were read only to locate the breast",
        "  region and the nipple (not applied).",
        "",
        "## Processing (by `src/scripts/build_torso_asset.py`)",
        "",
        "1. Kept only the `body` face group of `base.obj` (removed helper-* proxies: eyes, eyelashes,",
        "   teeth, tongue, hair, genital, skirt, tights; and all joint-* helper cubes, which were used",
        "   only as landmarks for cropping).",
        f"2. Phantom-side ({args.side}) breast flattened: vertices of the breast region (where the",
        f"   cup-size targets differ by > {args.region_frac:.0%} of their max, plus a {args.nipple_disk * 100:.0f} mm",
        f"   disc around the nipple, dilated {args.region_dilate} ring, holes filled) were moved along the",
        "   local chest normal onto a thin-plate-spline height field fitted to a 3-ring band of",
        "   undisturbed skin around the region (vertex tangential positions kept).",
        f"3. {args.subdiv} level(s) of Catmull-Clark subdivision.",
        "4. Cropped to a torso (plane cuts below the neck, at the hips and across the upper arms),",
        "   boundary vertices snapped to the cut planes, quads triangulated, normals recomputed.",
        "5. Units converted from MakeHuman decimetres to millimetres (x100).",
        "",
        f"Result: {asset['n_verts']} vertices, {asset['n_tris']} triangles, bbox (mm) {asset['bbox_mm']}.",
        "",
        "The derived `torso.json` is likewise offered under CC0 1.0.",
        "",
    ]
    path.write_text("\n".join(lines))
    print(f"wrote {path}")


def render_previews(out: Path, V, T, anchors):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    out.mkdir(parents=True, exist_ok=True)
    fn = tri_normals(V, T)
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    # plot in (x, z, y) so that y is up in matplotlib
    def M(X):  # world (x, y up, z anterior) -> right-handed plot coords (x, -z, y)
        X = np.asarray(X, float)
        return np.stack([X[..., 0], -X[..., 2], X[..., 1]], -1)

    P = M(V)
    views = {"front": (5, -90), "oblique_right": (15, -135), "side_right": (5, 180), "oblique_left": (15, -45),
             "top": (75, -90), "front_xray": (5, -90)}
    for name, (elev, azim) in views.items():
        fig = plt.figure(figsize=(8, 9), dpi=110)
        ax = fig.add_subplot(111, projection="3d")
        e, a = np.radians(elev), np.radians(azim)
        # camera direction in plot coords (x, z, y) -> world
        cam_plot = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        cam = np.array([cam_plot[0], cam_plot[2], -cam_plot[1]])
        light = unit(cam + np.array([0.3, 0.5, 0.0]))
        shade = np.clip(fn @ light, 0, 1) * 0.75 + 0.2
        col = np.stack([shade * 0.93, shade * 0.78, shade * 0.70, np.ones_like(shade)], 1)
        # painter's ordering
        order = np.argsort(V[T].mean(1) @ cam)
        if name.endswith("xray"):
            col[:, 3] = 0.12
        pc = Poly3DCollection(P[T[order]], facecolors=col[order], edgecolors="none", linewidths=0, antialiased=False)
        ax.add_collection3d(pc)
        for side, c in [("right", "tab:blue"), ("left", "tab:green")]:
            an = anchors[side]
            nip = M(an["nipple"])
            cw = M(an["chest_wall_point"])
            ax.scatter(*nip, color="red", s=30, depthshade=False)
            ax.scatter(*cw, color=c, s=30, depthshade=False)
            for key, cc in [("normal", "red"), ("superior", "blue"), ("lateral", "green")]:
                d = M(an[key]) * 60
                ax.plot(*np.stack([cw, cw + d]).T, color=cc, lw=2)
            # base circle
            nrm, sup, lat = (np.array(an[k]) for k in ("normal", "superior", "lateral"))
            th = np.linspace(0, 2 * np.pi, 60)
            circ = np.array(an["chest_wall_point"]) + an["base_radius_mm"] * (np.cos(th)[:, None] * sup + np.sin(th)[:, None] * lat)
            ax.plot(*M(circ).T, color=c, lw=1.2)
        lo, hi = P.min(0), P.max(0)
        c0 = (lo + hi) / 2; r = (hi - lo).max() / 2
        ax.set_xlim(c0[0] - r, c0[0] + r); ax.set_ylim(c0[1] - r, c0[1] + r); ax.set_zlim(c0[2] - r, c0[2] + r)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_title(f"torso {name}")
        fig.tight_layout()
        p = out / f"torso_{name}.png"
        fig.savefig(p)
        plt.close(fig)
        print(f"preview {p}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side", choices=["right", "left"], default="right",
                    help="subject side whose breast is replaced by the phantom (flattened)")
    ap.add_argument("--cache", default=str(DEFAULT_CACHE), help="download cache dir for MakeHuman files")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--preview", default=None, help="directory for preview PNGs (optional)")
    ap.add_argument("--subdiv", type=int, default=2)
    ap.add_argument("--region-frac", type=float, default=0.10)
    ap.add_argument("--region-dilate", type=int, default=1)
    ap.add_argument("--neck-offset", type=float, default=-0.15, help="dm, relative to neck joint")
    ap.add_argument("--hip-offset", type=float, default=0.9, help="dm, relative to pelvis joint")
    ap.add_argument("--arm-stub", type=float, default=0.35, help="fraction of shoulder->elbow kept")
    ap.add_argument("--arm-radius", type=float, default=0.8, help="dm, capsule radius selecting arm geometry")
    ap.add_argument("--nipple-disk", type=float, default=0.45, help="dm, disc around nipple added to region")
    ap.add_argument("--normal-radius-mm", type=float, default=25.0)
    ap.add_argument("--footprint-mm", type=float, default=2.0)
    args = ap.parse_args(argv)
    build(args)


if __name__ == "__main__":
    sys.exit(main())
