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
import itertools
import os
import tempfile
import time
import zlib

# Parallel k-Wave runs each use their own files; HDF5's advisory locking only makes concurrent
# opens fail with EAGAIN. Must be set before the HDF5 library is loaded.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import numpy as np
from scipy import ndimage, signal, stats

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


def boundary_jitter(pts, seed, amp_mm, corr_mm=0.2):
    """Smooth random displacement field (mm), a function of world position only.

    Value noise on a corr_mm lattice, each component within +-amp_mm. Used to sample the
    voxelised phantom with sub-voxel uncertainty: a published tissue boundary is only known to
    within half a voxel, and an exactly planar staircase acts as a mirror at 10+ MHz, whereas
    real interfaces are rough on the wavelength scale (Rayleigh criterion: rms > lambda/8)."""
    import torch
    p = torch.from_numpy(np.ascontiguousarray(pts.reshape(-1, 3)) / corr_mm)
    i0 = torch.floor(p).to(torch.int64)
    f = p - i0
    f = f * f * (3 - 2 * f)
    out = torch.zeros_like(p)
    for corner in itertools.product((0, 1), repeat=3):
        o = torch.tensor(corner)
        w = torch.prod(torch.where(o.bool(), f, 1 - f), -1)
        c = i0 + o
        for d in range(3):
            out[:, d] += w * (2 * scatter._uniform(scatter._hash(c[:, 0], c[:, 1], c[:, 2], seed, 101 + d)) - 1)
    return (out.numpy() * amp_mm).reshape(pts.shape)


def plane_maps(pid, vol, pose, spec, depth_mm, cols_per_element=3, lateral_margin_mm=1.0, seed=0, roughness=0.5,
               conform=True, gel_mm=0.5):
    """Acoustic maps (rows = depth from the probe face, cols = lateral) on a grid aligned to the array.

    dx = pitch / cols_per_element, and every element covers exactly cols_per_element columns
    (a pitch that is not a multiple of dx gives elements of 4 or 5 columns, whose periodic
    pattern shows up as a comb in the near field).
    roughness: maximum sub-voxel displacement of the sampling positions, in source voxels
    (0 samples the voxel grid exactly).
    conform: the probe is pressed onto the skin, as in a real exam and in the preview
    (conform=True there): each column is shifted along the beam so that tissue starts gel_mm
    below the face, instead of a flat face over a curved breast with water wedges at the sides."""
    dx_mm = spec.pitch * 1e3 / cols_per_element
    info = phantom.get_info(pid)
    if info.source != "oat64":
        raise NotImplementedError("full-resolution plane extraction is implemented for IDB-8164905 (oat64)")
    origin, axes = pose.frame(vol.surface, vol.voxel_mm, spec)
    n_ap = spec.n_elements * cols_per_element
    m = int(round(lateral_margin_mm / dx_mm))
    x = (np.arange(-m, n_ap + m) - (n_ap - 1) / 2) * dx_mm
    el_of_col = np.full(len(x), -1)
    el_of_col[m:m + n_ap] = np.arange(n_ap) // cols_per_element
    nz = int(round(depth_mm / dx_mm))
    z = np.arange(nz) * dx_mm
    shift = np.zeros(len(x))
    with h5py.File(info.files["label"], "r") as hl, h5py.File(info.files["acou"], "r") as ha:
        src_voxel = float(np.array(hl["voxel_size"]).ravel()[0])
        if conform:
            zt = np.arange(0, 20.0, dx_mm)                      # look up to 20 mm below the face
            pt = origin + x[None, :, None] * axes[0] + zt[:, None, None] * axes[1]
            lt = _sample(hl["label"], world_to_source(vol.meta, hl["label"].shape, src_voxel, pt), 0)
            has = (lt != tissue.BACKGROUND).any(0)
            first = np.where(has, np.argmax(lt != tissue.BACKGROUND, 0), 0) * dx_mm
            first = np.where(has, first, np.interp(x, x[has], first[has]) if has.any() else 0.0)
            shift = np.clip(ndimage.median_filter(first, 9) - gel_mm, 0, None)
        pts = origin + x[None, :, None] * axes[0] + (z[:, None, None] + shift[None, :, None]) * axes[1]
        if roughness > 0:
            pts = pts + boundary_jitter(pts, seed, roughness * src_voxel)
        idx = world_to_source(vol.meta, hl["label"].shape, src_voxel, pts)
        label = _sample(hl["label"], idx, 0).astype(np.uint8)
        c = _sample(ha["sound_speed"], idx, 1) * 1e3            # mm/us -> m/s
        rho = _sample(ha["density"], idx, 1) * 1e6              # g/mm^3 -> kg/m^3
        alpha = _sample(ha["alpha_coeff"], idx, 1) * 10.0       # dB/(MHz^y mm) -> dB/(MHz^y cm)
        power = float(np.array(ha["alpha_power"]).ravel()[0])
    water = label == tissue.BACKGROUND
    c[water], rho[water], alpha[water] = WATER
    return dict(label=label, c=c.astype(np.float32), rho=rho.astype(np.float32), alpha=alpha.astype(np.float32),
                alpha_power=power, x=x, z=z, origin=origin, axes=axes, dx=dx_mm, el_of_col=el_of_col, shift=shift)


def add_scatterers(maps, seed, strength=0.02, slab_mm=0.15, per_cell=3):
    """Scatterers within +-slab of the plane -> relative density / sound-speed perturbations.

    Each 0.1 mm lattice cell holds per_cell independent scatterers (the first one is the
    preview's), giving ~15-20 per resolution cell at 11 MHz (fully developed speckle needs
    >10). Amplitudes are scaled by 1/sqrt(per_cell) so `strength` keeps its meaning (rms
    per cell). Each scatterer is spread bilinearly over its 4 grid neighbours: rounding to the
    nearest grid point beats the 0.1 mm lattice against the grid pitch (moire)."""
    dx, x, z, (e_lat, e_ax, e_el), o = maps["dx"], maps["x"], maps["z"], maps["axes"], maps["origin"]
    cell = scatter.SCATTER_CELL_MM
    step = cell / np.sqrt(2)
    xs = np.arange(x[0], x[-1], step)
    shift = maps.get("shift", np.zeros(len(x)))
    zs = np.arange(0, z[-1] + shift.max(), step)
    ys = np.arange(-slab_mm, slab_mm + 1e-9, step)
    X, Z, Y = np.meshgrid(xs, zs, ys, indexing="ij")
    pts = o + X[..., None] * e_lat + Z[..., None] * e_ax + Y[..., None] * e_el
    import torch
    idx = scatter.unique_cells(torch.from_numpy(pts.reshape(-1, 3)).float())
    delta = np.zeros(maps["c"].shape, np.float32)
    echo_lut = tissue.lut(tissue.ECHOGENICITY)
    n = 0
    for k in range(per_cell):
        amp, pos = scatter.from_index(idx, seed=seed + 7919 * k)
        rel = pos.numpy().astype(np.float64) - o
        px, pz, py = rel @ e_lat, rel @ e_ax, rel @ e_el
        fc_ = (px - x[0]) / dx
        fr_ = (pz - np.interp(px, x, shift)) / dx               # conformed columns start deeper
        c0_, r0_ = np.floor(fc_).astype(int), np.floor(fr_).astype(int)
        ok = (np.abs(py) <= slab_mm) & (c0_ >= 0) & (c0_ < len(x) - 1) & (r0_ >= 0) & (r0_ < len(z) - 1)
        c0_, r0_, wc, wr = c0_[ok], r0_[ok], (fc_ - np.floor(fc_))[ok], (fr_ - np.floor(fr_))[ok]
        a = amp.numpy()[ok] * strength / np.sqrt(per_cell)
        a = a * echo_lut[maps["label"][np.where(wr < 0.5, r0_, r0_ + 1), np.where(wc < 0.5, c0_, c0_ + 1)]]
        for dr, dc, w in ((0, 0, (1 - wr) * (1 - wc)), (0, 1, (1 - wr) * wc), (1, 0, wr * (1 - wc)), (1, 1, wr * wc)):
            np.add.at(delta, (r0_ + dr, c0_ + dc), (a * w).astype(np.float32))
        n += int(ok.sum())
    maps["c"] = np.clip(maps["c"] * (1 + delta), 1300, 1800).astype(np.float32)
    maps["rho"] = (maps["rho"] * (1 + delta)).astype(np.float32)
    maps["n_scatterers"] = n
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


def _pulse(spec):
    """Gaussian-windowed tone burst: (sigma_t, t_c = time of the envelope peak)."""
    sig_t = 1 / (2 * np.pi * (spec.bandwidth * spec.fc / 2.355))
    return sig_t, 4 * sig_t


def _kwave_run(c, rho, alpha, power, dx, cols, delays, apod, spec, t_end, cfl=0.3, gpu=True):
    """One k-Wave run: the face row emits the pulse from `cols` with per-column delays / weights and
    records pressure there. Returns (p (len(cols), Nt), t)."""
    from kwave.kgrid import kWaveGrid
    from kwave.kmedium import kWaveMedium
    from kwave.ksensor import kSensor
    from kwave.ksource import kSource
    from kwave.kspaceFirstOrder2D import kspaceFirstOrder2D
    from kwave.options.simulation_execution_options import SimulationExecutionOptions
    from kwave.options.simulation_options import SimulationOptions

    nz, nx = c.shape
    kgrid = kWaveGrid([nz, nx], [dx, dx])
    kgrid.makeTime(float(c.max()), cfl, t_end)
    t = kgrid.t_array.ravel()
    medium = kWaveMedium(sound_speed=c.copy(), density=rho.copy(), alpha_coeff=alpha.copy(), alpha_power=power)
    sig_t, t_c = _pulse(spec)
    mask = np.zeros((nz, nx), bool)
    mask[0, cols] = True                                     # one row: k-Wave orders points by column
    tt = t[None, :] - delays[:, None] - t_c
    src = kSource()
    src.p_mask = mask
    src.p = (apod[:, None] * np.exp(-0.5 * (tt / sig_t) ** 2) * np.sin(2 * np.pi * spec.fc * tt)).astype(np.float32)
    sensor = kSensor(mask=mask, record=["p"])
    with tempfile.TemporaryDirectory(prefix="breastsim_kwave_") as tmp:   # k-Wave leaves its h5 files behind
        out = kspaceFirstOrder2D(kgrid, src, sensor, medium,
                                 SimulationOptions(pml_inside=False, pml_size=[_nice(nz), _nice(nx)],
                                                   data_cast="single", save_to_disk=True, data_path=tmp),
                                 SimulationExecutionOptions(is_gpu_simulation=gpu, show_sim_log=False))
    p = np.asarray(out["p"] if isinstance(out, dict) else out)
    if p.shape[0] == len(t):
        p = p.T
    return p, t


def simulate_focused(maps, spec, depth_mm, focus_mm, n_active=128, line_step=2, margin_mm=1.5, f_number=1.5,
                     px_mm=0.05, tx_apod="tukey", gpu=True, device=None, workers=4, x_windows=None, log=print):
    """Conventional focused line-by-line B-mode, as in k-Wave's example_us_bmode_linear_transducer:
    for every scan line the active sub-aperture (n_active elements centred on the line) fires a
    beam focused at focus_mm and receives; only the medium under the sub-aperture (+ margin) is
    simulated. Each line is beamformed with dynamic receive focusing (F-number, Hann apodisation).
    Two-way focusing gives the low side-lobe / clutter level of clinical scanners, which a few
    unfocused plane waves do not (shadows and cysts get filled by clutter).
    x_windows: optional [(x0_mm, half_width_mm), ...]; only lines inside them are simulated
    (point-spread checks), the others stay zero.
    Returns the envelope image (z, lateral) on a px_mm grid over the aperture."""
    import torch
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dx = maps["dx"] * 1e-3
    x = maps["x"] * 1e-3
    el_of_col = maps["el_of_col"]
    n_el, pitch = spec.n_elements, spec.pitch
    x_el = (np.arange(n_el) - (n_el - 1) / 2) * pitch
    sig_t, t_c = _pulse(spec)
    zf = focus_mm * 1e-3
    zs = np.arange(px_mm / 2, depth_mm, px_mm) * 1e-3
    lines = np.arange(0, n_el, line_step)
    x_lines = x_el[lines] + (pitch / 2 if line_step % 2 == 0 else 0.0)
    if x_windows:
        keep = np.zeros(len(lines), bool)
        for x0, hw in x_windows:
            keep |= np.abs(x_lines * 1e3 - x0) <= hw
        lines, x_lines = lines[keep], x_lines[keep]
    img = np.zeros((len(zs), len(lines)), np.complex64)
    mcols = int(round(margin_mm * 1e-3 / dx))
    zt = torch.from_numpy(zs).to(dev, torch.float64)
    def run_line(li):
        l, xl = lines[li], x_lines[li]
        act = np.arange(max(l - n_active // 2, 0), min(l + n_active // 2, n_el))
        acols = np.nonzero(np.isin(el_of_col, act))[0]
        lo, hi = max(acols.min() - mcols, 0), min(acols.max() + 1 + mcols, len(x))
        sub = slice(lo, hi)
        r_f = np.sqrt(zf ** 2 + (x_el[act] - xl) ** 2)
        tau = (r_f.max() - r_f) / C0                                  # per element, outer ones first
        w = np.ones(len(act))
        if tx_apod == "tukey" and len(act) > 2:
            w = signal.windows.tukey(len(act), 0.5)
        e_idx = el_of_col[acols] - act[0]
        t_end = 2 * depth_mm * 1e-3 / 1400 + 2 * t_c + tau.max()
        for attempt in range(3):
            try:
                p, t = _kwave_run(maps["c"][:, sub], maps["rho"][:, sub], maps["alpha"][:, sub], maps["alpha_power"],
                                  dx, acols - lo, tau[e_idx], w[e_idx], spec, t_end, gpu=gpu)
                break
            except (BlockingIOError, OSError) as err:
                if attempt == 2:
                    raise
                log(f"    line {li}: {err!r}, retrying")
                time.sleep(2)
        rf = np.zeros((len(act), len(t)), np.float32)
        np.add.at(rf, e_idx, p)
        rf /= np.maximum(np.bincount(e_idx, minlength=len(act)), 1)[:, None]
        rf[t[None, :] < (tau + 2 * t_c)[:, None]] = 0.0             # own transmit pulse
        return li, act, r_f, rf, t, hi - lo

    def beamform_line(li, act, r_f, rf, t):
        xl = x_lines[li]
        ana = torch.from_numpy(signal.hilbert(rf, axis=-1).astype(np.complex64)).to(dev)
        dt = float(t[1] - t[0])
        # transmit arrival on the line axis: all elements meet at the focus at max(r_f)/c0
        t_tx = (r_f.max() - zf + zt) / C0 + t_c
        d = torch.from_numpy(x_el[act] - xl).to(dev, torch.float64)
        half = zt[:, None] / (2 * f_number)
        ap = (d.abs()[None] <= half).double() * (0.5 + 0.5 * torch.cos(np.pi * d[None] / half.clamp(min=1e-9)))
        r = torch.sqrt(zt[:, None] ** 2 + d[None] ** 2)
        ti = (t_tx[:, None] + r / C0) / dt
        i0 = ti.floor().long().clamp(0, ana.shape[-1] - 2)
        fr = (ti - i0).clamp(0, 1).float()
        e_ix = torch.arange(len(act), device=dev)[None].expand_as(i0)
        s0, s1 = ana[e_ix, i0], ana[e_ix, i0 + 1]
        wsum = (ap / torch.sqrt(r / 1e-3)).sum(1)
        col = ((ap.float() * (s0 * (1 - fr) + s1 * fr)).sum(1) / wsum.clamp(min=1e-6).float()) * (wsum > 0)
        img[:, li] = col.cpu().numpy()

    # Each line is a small grid that leaves most of the GPU idle, so several k-Wave runs go in
    # parallel (each in its own process via the k-Wave binary; the threads only wait on them).
    from concurrent.futures import ThreadPoolExecutor, as_completed
    t_line = time.time()
    with ThreadPoolExecutor(workers) as pool:
        futs = [pool.submit(run_line, li) for li in range(len(lines))]
        for k_done, f in enumerate(as_completed(futs)):
            li, act, r_f, rf, t, width = f.result()
            beamform_line(li, act, r_f, rf, t)
            if k_done % 32 == 0:
                log(f"    {k_done + 1}/{len(lines)} lines: grid {maps['c'].shape[0]}x{width}, Nt={len(t)}, "
                    f"{(time.time() - t_line) / (k_done + 1):.2f} s/line ({workers} parallel)")
    # lines -> px_mm lateral grid over the aperture
    xs = np.arange(-spec.aperture / 2, spec.aperture / 2, px_mm * 1e-3)
    re = np.stack([np.interp(xs, x_lines, row.real) for row in img])
    im = np.stack([np.interp(xs, x_lines, row.imag) for row in img])
    return np.abs(re + 1j * im)


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
    if "el_of_col" in maps:
        el_of_col = maps["el_of_col"]
    else:
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
        with tempfile.TemporaryDirectory(prefix="breastsim_kwave_") as tmp:
            out = kspaceFirstOrder2D(kgrid, src, sensor, medium,
                                     SimulationOptions(pml_inside=False, pml_size=list(pml), data_cast="single",
                                                       save_to_disk=True, data_path=tmp),
                                     SimulationExecutionOptions(is_gpu_simulation=gpu, show_sim_log=False))
        p = np.asarray(out["p"] if isinstance(out, dict) else out)
        if p.shape[0] == len(t):                                    # (Nt, points) -> (points, Nt)
            p = p.T
        el = np.zeros((n_el, len(t)), np.float32)                   # average the points of each element
        np.add.at(el, el_of_col[cols], p)
        el /= np.maximum(np.bincount(el_of_col[cols], minlength=n_el), 1)[:, None]
        # the transmit pulse is recorded by the element that emits it: mute each element until its
        # own emission has ended (emission centre tau + t_c, Gaussian tail t_c later)
        tau_el = (x_el - xr) * np.sin(th) / C0
        el[t[None, :] < (tau_el + 2 * t_c)[:, None]] = 0.0
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
    wsum = torch.zeros(Z.shape, device=dev)
    nt = ana.shape[-1]
    for a, ang in enumerate(angles_deg):
        th = np.deg2rad(ang)
        xr = float(x_el[0] if th >= 0 else x_el[-1])
        t_tx = (Z * np.cos(th) + (X - xr) * np.sin(th)) / C0 + t_c
        for e in range(len(x_el)):
            d = X - xe[e]
            ap = (d.abs() <= Z / (2 * f_number)).float() * (0.5 + 0.5 * torch.cos(np.pi * d / (Z / (2 * f_number)).clamp(min=1e-9)))
            r = torch.sqrt(Z ** 2 + d ** 2)
            ti = (t_tx + r / C0) / dt
            i0 = ti.floor().long().clamp(0, nt - 2)
            fr = (ti - i0).clamp(0, 1)
            s = ana[a, e]
            img += ap * (s[i0] * (1 - fr) + s[i0 + 1] * fr)
            wsum += ap / torch.sqrt(r / 1e-3)
    # The dynamic aperture grows with depth (F-number), so the raw sum brightens with depth even
    # in a uniform medium. In 2D each element's echo amplitude falls as 1/sqrt(r) (cylindrical
    # spreading), hence the beamformed speckle / point amplitude scales as sum(w / sqrt(r)):
    # dividing by it leaves a uniform, non-attenuating medium flat with depth, as a scanner's
    # system gain calibration does. TGC then only compensates attenuation.
    return (img.abs() / wsum.clamp(min=1e-6) * (wsum > 0)).cpu().numpy()


def to_bmode(env, depth_mm, spec, alpha_power, tgc_db_cm_mhz=0.5, dr_db=55.0, auto_tgc=True):
    """Envelope -> 8-bit B-mode. TGC compensates a nominal tissue attenuation (dB/cm/MHz^y)
    for the whole pulse spectrum, not just fc: attenuation shifts the echo spectrum down, so
    the received energy falls more slowly than alpha*fc^y predicts and a centre-frequency TGC
    over-amplifies the deep image."""
    from ..realtime.renderer import spectral_attenuation_lut
    z_cm = (np.arange(env.shape[0]) + 0.5) / env.shape[0] * depth_mm / 10
    A, E, _, _ = spectral_attenuation_lut(spec, alpha_power, a_max=float(tgc_db_cm_mhz / 8.686 * z_cm[-1] * 1.05))
    loss_db = -10 * np.log10(np.interp(tgc_db_cm_mhz / 8.686 * z_cm, A, E))
    db = 20 * np.log10(env + 1e-12) + loss_db[:, None]
    if auto_tgc:
        # Operator-style TGC: the nominal slope rarely matches the tissue (fat ~0.4, gland ~0.8
        # dB/cm/MHz^y here), and sonographers set the sliders until the background is even.
        # As PyMUST's tgc(): robust linear fit of the per-depth level over the central 80% of
        # the depth, then remove that linear trend. Row medians ignore bright interfaces and
        # local shadows, so lesion contrast and posterior features are kept.
        ok = env > 0
        med = np.array([np.median(r[m]) if m.sum() > 10 else np.nan for r, m in zip(db, ok)])
        n = len(med)
        sel = np.arange(n // 10, 9 * n // 10)
        sel = sel[np.isfinite(med[sel])]
        if len(sel) > 20:
            slope = stats.theilslopes(med[sel], z_cm[sel])[0]
            db -= (slope * (z_cm - z_cm[sel].mean()))[:, None]
    db -= np.percentile(db, 99.5)
    return (np.clip(db / dr_db + 1, 0, 1) * 255).astype(np.uint8)


def render_pose(pid, vol, pose, probe="L12-hf", depth_mm=40.0, cols_per_element=3, tx="focused", focus_mm=None,
                angles_deg=(-8, -4, 0, 4, 8), n_active=128, line_step=2, scatter_strength=0.02, roughness=0.5,
                conform=True, px_mm=0.05, log=print):
    spec = PROBES[probe]
    t0 = time.time()
    seed = zlib.crc32(pid.encode()) % 100_000
    maps = plane_maps(pid, vol, pose, spec, depth_mm, cols_per_element, seed=seed, roughness=roughness,
                      conform=conform)
    maps = add_scatterers(maps, seed, scatter_strength)
    log(f"  medium {maps['c'].shape} ready ({maps['n_scatterers']} scatterers, {time.time() - t0:.1f} s)")
    if tx == "focused":
        env = simulate_focused(maps, spec, depth_mm, focus_mm or depth_mm / 2, n_active=n_active,
                               line_step=line_step, px_mm=px_mm, log=log)
    else:
        rf, dt, x_el, t_c = simulate(maps, spec, angles_deg, log=log)
        env = beamform(rf, dt, x_el, t_c, angles_deg, depth_mm, spec.aperture * 1e3, px_mm=px_mm)
    img = to_bmode(env, depth_mm, spec, maps["alpha_power"])
    lab = maps["label"][:, np.abs(maps["x"]) <= spec.aperture * 1e3 / 2]    # aperture only, like the image
    lab_img = ndimage.zoom(lab, (env.shape[0] / lab.shape[0], env.shape[1] / lab.shape[1]), order=0)
    return img, lab_img, {"seconds": round(time.time() - t0, 1), "grid": list(maps["c"].shape),
                          "scatterers": maps["n_scatterers"], "tx": tx, "focus_mm": focus_mm or depth_mm / 2}
