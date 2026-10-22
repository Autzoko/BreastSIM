"""2D full-wave B-mode simulation (k-Wave) of a probe pose in a numerical breast phantom.

Content-consistent with the real-time preview:
- the image plane comes from the same ProbePose / probe spec, located in the preview
  cache's world frame and mapped back to the phantom's ORIGINAL full-resolution voxels
  (sound speed, density, attenuation and labels are used exactly as published);
- sub-resolution scattering uses the same deterministic scatterer lattice as the preview
  (breastsim.scatter, same seed): each lattice scatterer in a thin slab around the plane
  becomes a small density/sound-speed perturbation, scaled by the tissue echogenicity
  (k-Wave and Fullwave B-mode practice; the phantoms themselves carry no sub-wavelength
  texture). The macro acoustic values are not changed.

Imaging: flat linear array at the probe face (water/gel above the skin where the skin
curves away), plane-wave transmits at several angles, element signals recorded by k-Wave,
delay-and-sum beamforming with c0 = 1540 m/s, coherent compounding, envelope, TGC, log.
"""
import time
import zlib

import h5py
import numpy as np
from scipy import ndimage, signal

from .. import phantom, scatter, tissue
from ..probe import PROBES

C0 = 1540.0
WATER = (1500.0, 1000.0, 0.0022)       # c m/s, rho kg/m^3, alpha dB/(MHz^y cm)


# ---------------------------------------------------------------- geometry
def world_to_source(meta, src_shape, src_voxel, pts):
    """Preview-cache world coordinates (a, u, v) mm -> original full-resolution voxel indices.

    Inverts phantom.canonicalize (axis order, flip, crop, padding) and the preview
    downsampling factor."""
    ori = meta["orientation"]
    order, k = ori["axis_order"], int(round(meta["voxel_mm"] / src_voxel))
    t = np.asarray(pts, np.float64) / meta["voxel_mm"] - np.array([0.0, 4.0, 4.0])   # canonical padding
    t = t + np.array(ori["crop_lo"], np.float64)
    if ori["flip"]:
        n0 = src_shape[order[0]] // k
        t[..., 0] = (n0 - 1) - t[..., 0]
    ds = np.empty_like(t)
    for i, ax in enumerate(order):
        ds[..., ax] = t[..., i]
    return ds * k


def _sample(dset, idx, order):
    """Sample an h5 dataset at float indices (..., 3), reading only the bounding box."""
    lo = np.maximum(np.floor(idx.reshape(-1, 3).min(0)).astype(int) - 2, 0)
    hi = np.minimum(np.ceil(idx.reshape(-1, 3).max(0)).astype(int) + 3, dset.shape)
    block = dset[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    rel = (idx - lo).reshape(-1, 3).T
    out = ndimage.map_coordinates(block, rel, order=order, mode="nearest")
    return out.reshape(idx.shape[:-1])


def plane_maps(pid, vol, pose, spec, depth_mm, dx_mm, lateral_margin_mm=1.0):
    """Acoustic maps (rows = depth from the probe face, cols = lateral) on a dx grid."""
    info = phantom.get_info(pid)
    if info.source != "oat64":
        raise NotImplementedError("full-resolution plane extraction is implemented for IDB-8164905 (oat64)")
    origin, axes = pose.frame(vol.surface, vol.voxel_mm, spec)
    width = spec.aperture * 1e3 + 2 * lateral_margin_mm
    nx = int(round(width / dx_mm))
    nz = int(round(depth_mm / dx_mm))
    x = (np.arange(nx) - (nx - 1) / 2) * dx_mm
    z = np.arange(nz) * dx_mm
    pts = origin + x[None, :, None] * axes[0] + z[:, None, None] * axes[1]
    with h5py.File(info.files["label"], "r") as hl, h5py.File(info.files["acou"], "r") as ha:
        src_voxel = float(np.array(hl["voxel_size"]).ravel()[0])
        idx = world_to_source(vol.meta, hl["label"].shape, src_voxel, pts)
        label = _sample(hl["label"], idx, 0).astype(np.uint8)
        c = _sample(ha["sound_speed"], idx, 1) * 1e3            # mm/us -> m/s
        rho = _sample(ha["density"], idx, 1) * 1e6              # g/mm^3 -> kg/m^3
        alpha = _sample(ha["alpha_coeff"], idx, 1) * 10.0       # dB/(MHz^y mm) -> dB/(MHz^y cm)
        power = float(np.array(ha["alpha_power"]).ravel()[0])
    water = label == tissue.BACKGROUND
    c[water], rho[water], alpha[water] = WATER
    return dict(label=label, c=c.astype(np.float32), rho=rho.astype(np.float32), alpha=alpha.astype(np.float32),
                alpha_power=power, x=x, z=z, origin=origin, axes=axes, dx=dx_mm)


def add_scatterers(maps, seed, strength=0.02, slab_mm=0.15):
    """Lattice scatterers within +-slab of the plane -> relative density / sound-speed perturbations."""
    dx, x, z, (e_lat, e_ax, e_el), o = maps["dx"], maps["x"], maps["z"], maps["axes"], maps["origin"]
    cell = scatter.SCATTER_CELL_MM
    step = cell / np.sqrt(2)
    xs = np.arange(x[0], x[-1], step)
    zs = np.arange(0, z[-1], step)
    ys = np.arange(-slab_mm, slab_mm + 1e-9, step)
    X, Z, Y = np.meshgrid(xs, zs, ys, indexing="ij")
    pts = o + X[..., None] * e_lat + Z[..., None] * e_ax + Y[..., None] * e_el
    import torch
    idx = scatter.unique_cells(torch.from_numpy(pts.reshape(-1, 3)).float())
    amp, pos = scatter.from_index(idx, seed=seed)
    rel = pos.numpy().astype(np.float64) - o
    px, pz, py = rel @ e_lat, rel @ e_ax, rel @ e_el
    col = np.round((px - x[0]) / dx).astype(int)
    row = np.round(pz / dx).astype(int)
    ok = (np.abs(py) <= slab_mm) & (col >= 0) & (col < len(x)) & (row >= 0) & (row < len(z))
    row, col, amp = row[ok], col[ok], amp.numpy()[ok]
    echo = tissue.lut(tissue.ECHOGENICITY)[maps["label"][row, col]]
    delta = np.zeros(maps["c"].shape, np.float32)
    np.add.at(delta, (row, col), strength * echo * amp)
    maps["c"] = np.clip(maps["c"] * (1 + delta), 1300, 1800).astype(np.float32)
    maps["rho"] = (maps["rho"] * (1 + delta)).astype(np.float32)
    maps["n_scatterers"] = int(ok.sum())
    return maps


# ------------------------------------------------------------- simulation
def _nice(n, pml_min=16, pml_max=40):
    """PML size so that n + 2 pml has only small prime factors (fast FFTs)."""
    def small(m):
        for p in (2, 3, 5, 7):
            while m % p == 0:
                m //= p
        return m == 1
    for p in range(pml_min, pml_max + 1):
        if small(n + 2 * p):
            return p
    return pml_min


def simulate(maps, spec, angles_deg=(-8, -4, 0, 4, 8), cfl=0.3, gpu=True, log=print):
    """Run one k-Wave simulation per plane-wave angle; returns per-element RF (angles, elements, Nt), dt."""
    from kwave.kgrid import kWaveGrid
    from kwave.kmedium import kWaveMedium
    from kwave.ksensor import kSensor
    from kwave.ksource import kSource
    from kwave.kspaceFirstOrder2D import kspaceFirstOrder2D
    from kwave.options.simulation_execution_options import SimulationExecutionOptions
    from kwave.options.simulation_options import SimulationOptions

    dx = maps["dx"] * 1e-3
    nz, nx = maps["c"].shape
    depth = nz * dx

    def grid_and_medium():
        # k-Wave pads the grid and medium arrays with the PML in place, so each run gets fresh ones
        kg = kWaveGrid([nz, nx], [dx, dx])
        kg.makeTime(float(maps["c"].max()), cfl, t_end)
        med = kWaveMedium(sound_speed=maps["c"].copy(), density=maps["rho"].copy(),
                          alpha_coeff=maps["alpha"].copy(), alpha_power=maps["alpha_power"])
        return kg, med

    fc, bw = spec.fc, spec.bandwidth
    sig_t = 1 / (2 * np.pi * (bw * fc / 2.355))                 # Gaussian envelope of the pulse
    t_c = 4 * sig_t
    pitch, n_el = spec.pitch, spec.n_elements
    x = maps["x"] * 1e-3
    x_el = (np.arange(n_el) - (n_el - 1) / 2) * pitch
    el_of_col = np.full(nx, -1)
    for e, xe in enumerate(x_el):
        el_of_col[np.abs(x - xe) <= 0.45 * pitch] = e
    cols = np.nonzero(el_of_col >= 0)[0]
    mask = np.zeros((nz, nx), bool)
    mask[0, cols] = True
    max_delay = (x_el[-1] - x_el[0]) * np.sin(np.deg2rad(max(abs(a) for a in angles_deg))) / C0
    t_end = 2 * depth / 1400 + 2 * t_c + max_delay
    kgrid, _ = grid_and_medium()
    t = kgrid.t_array.ravel()
    pml = (_nice(nz), _nice(nx))
    rf = []
    for ang in angles_deg:
        th = np.deg2rad(ang)
        xr = x_el[0] if th >= 0 else x_el[-1]
        tau = (x[cols] - xr) * np.sin(th) / C0                   # per source point (column order)
        tt = t[None, :] - tau[:, None] - t_c
        src = kSource()
        src.p_mask = mask
        src.p = (np.exp(-0.5 * (tt / sig_t) ** 2) * np.sin(2 * np.pi * fc * tt)).astype(np.float32)
        sensor = kSensor(mask=mask, record=["p"])
        t0 = time.time()
        kgrid, medium = grid_and_medium()
        out = kspaceFirstOrder2D(kgrid, src, sensor, medium,
                                 SimulationOptions(pml_inside=False, pml_size=list(pml), data_cast="single",
                                                   save_to_disk=True),
                                 SimulationExecutionOptions(is_gpu_simulation=gpu, show_sim_log=False))
        p = np.asarray(out["p"] if isinstance(out, dict) else out)
        if p.shape[0] == len(t):                                    # (Nt, points) -> (points, Nt)
            p = p.T
        el = np.zeros((n_el, len(t)), np.float32)                   # average the points of each element
        np.add.at(el, el_of_col[cols], p)
        el /= np.maximum(np.bincount(el_of_col[cols], minlength=n_el), 1)[:, None]
        el[:, t < 2 * t_c + max_delay] = 0.0          # transmit pulse picked up by the receiving elements
        rf.append(el)
        log(f"    angle {ang:+.0f}: {nz}x{nx} grid, Nt={len(t)}, {time.time() - t0:.1f} s")
    return np.stack(rf), float(kgrid.dt), x_el, t_c


def beamform(rf, dt, x_el, t_c, angles_deg, depth_mm, width_mm, px_mm=0.1, f_number=1.5, device=None):
    """Plane-wave delay-and-sum with dynamic receive aperture; coherent compounding."""
    import torch
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ana = torch.from_numpy(signal.hilbert(rf, axis=-1).astype(np.complex64)).to(dev)   # (A, E, Nt)
    xs = torch.arange(-width_mm / 2, width_mm / 2, px_mm, device=dev) * 1e-3
    zs = torch.arange(px_mm / 2, depth_mm, px_mm, device=dev) * 1e-3
    Z, X = torch.meshgrid(zs, xs, indexing="ij")
    xe = torch.from_numpy(x_el).to(dev, torch.float32)
    img = torch.zeros(Z.shape, dtype=torch.complex64, device=dev)
    nt = ana.shape[-1]
    for a, ang in enumerate(angles_deg):
        th = np.deg2rad(ang)
        xr = float(x_el[0] if th >= 0 else x_el[-1])
        t_tx = (Z * np.cos(th) + (X - xr) * np.sin(th)) / C0 + t_c
        for e in range(len(x_el)):
            d = X - xe[e]
            ap = (d.abs() <= Z / (2 * f_number)).float() * (0.5 + 0.5 * torch.cos(np.pi * d / (Z / (2 * f_number)).clamp(min=1e-9)))
            ti = (t_tx + torch.sqrt(Z ** 2 + d ** 2) / C0) / dt
            i0 = ti.floor().long().clamp(0, nt - 2)
            fr = (ti - i0).clamp(0, 1)
            s = ana[a, e]
            img += ap * (s[i0] * (1 - fr) + s[i0 + 1] * fr)
    return img.abs().cpu().numpy()


def to_bmode(env, depth_mm, fc_mhz, tgc_db_cm_mhz=0.5, dr_db=55.0):
    z_cm = (np.arange(env.shape[0]) + 0.5) / env.shape[0] * depth_mm / 10
    db = 20 * np.log10(env + 1e-12) + (2 * tgc_db_cm_mhz * fc_mhz * z_cm)[:, None]
    db -= np.percentile(db, 99.5)
    return (np.clip(db / dr_db + 1, 0, 1) * 255).astype(np.uint8)


def render_pose(pid, vol, pose, probe="L12-hf", depth_mm=40.0, dx_mm=0.024, angles_deg=(-8, -4, 0, 4, 8),
                scatter_strength=0.02, log=print):
    spec = PROBES[probe]
    t0 = time.time()
    maps = plane_maps(pid, vol, pose, spec, depth_mm, dx_mm)
    maps = add_scatterers(maps, zlib.crc32(pid.encode()) % 100_000, scatter_strength)
    log(f"  medium {maps['c'].shape} ready ({maps['n_scatterers']} scatterers, {time.time() - t0:.1f} s)")
    rf, dt, x_el, t_c = simulate(maps, spec, angles_deg, log=log)
    env = beamform(rf, dt, x_el, t_c, angles_deg, depth_mm, spec.aperture * 1e3)
    img = to_bmode(env, depth_mm, spec.fc / 1e6)
    lab = maps["label"][:, np.abs(maps["x"]) <= spec.aperture * 1e3 / 2]    # aperture only, like the image
    lab_img = ndimage.zoom(lab, (env.shape[0] / lab.shape[0], env.shape[1] / lab.shape[1]), order=0)
    return img, lab_img, {"seconds": round(time.time() - t0, 1), "grid": list(maps["c"].shape),
                          "scatterers": maps["n_scatterers"]}
