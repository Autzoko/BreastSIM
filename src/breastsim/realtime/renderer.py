"""Fast B-mode preview on the GPU.

Approximate but content-consistent with the high-fidelity generator:
- anatomy, attenuation and speed of sound come from the same canonical phantom;
- the image plane comes from probe.ProbePose.frame / probe.skin_offsets
  (optionally pressing the skin flat under the probe);
- speckle: every scatterer of the shared deterministic lattice
  (breastsim.scatter) inside the elevation beam is counted once, weighted by
  the beam profile at its true elevation, phased by its true depth and
  splatted linearly onto the depth grid;
- specular echoes: reflection coefficient of the per-tissue mean impedance
  at label boundaries, placed at the boundary, scaled by an angle-dependent
  (specular + diffuse) factor from the boundary normal;
- imaging: separable Gaussian PSF (complex envelope). Attenuation is
  integrated over the pulse-echo spectrum, so deep echoes lose energy, shift
  to lower frequency and lose resolution (axial and lateral widths follow the
  local mean frequency and bandwidth);
- display depth follows the scanner's c0 = 1540 m/s assumption, so tissue
  slower than c0 appears deeper, as in a real scanner or a full-wave run;
- output is resampled per axis to the preset's pixel spacing (area average
  when shrinking), with optional vendor-style smoothing.
"""
import time
import zlib
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .. import scatter, tissue
from ..presets import DEFAULT_PRESET, PRESETS
from ..probe import PROBES, ProbePose, skin_offsets

NP_PER_DB = 1 / 8.686
C0 = 1540.0                       # speed of sound assumed by the beamformer, m/s


@dataclass
class ImagingParams:
    probe: str = "L12-hf"
    depth_mm: float = 42.0
    px_x_mm: float = 0.078            # output pixel spacing along the array
    px_z_mm: float = 0.078            # output pixel spacing in depth
    gain_db: float = -13.0
    dynamic_range_db: float = 52.0
    tgc_db_cm_mhz: float = 0.5        # depth gain compensation (one-way slope at fc)
    specular: float = 30.0            # perpendicular boundary echo = specular * R * (speckle rms of echogenicity 1)
    lateral_f_number: float = 2.0     # minimum receive F-number
    speckle_reduction: float = 1.0    # Gaussian sigma (output px) on the log image
    conform: bool = True              # probe presses the skin flat
    shadow_blur_mm: float = 1.0       # lateral beam extent averaging the attenuation (partial shadows)
    overlay: str = "contour"          # none | contour | labels
    elevation_samples: int = 5        # label planes across the elevation beam (specular term)
    preset: str = DEFAULT_PRESET

    @classmethod
    def from_preset(cls, name, **overrides):
        p = PRESETS[name]
        fields = {k: getattr(p, k) for k in ("probe", "depth_mm", "px_x_mm", "px_z_mm", "gain_db",
                                             "dynamic_range_db", "speckle_reduction",
                                             "lateral_f_number", "conform")}
        return cls(**{**fields, "preset": name, **overrides})


# ------------------------------------------------------------------ helpers
def _gauss_kernel(sigma_px, device):
    r = max(1, int(np.ceil(3 * sigma_px)))
    t = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    k = torch.exp(-0.5 * (t / max(sigma_px, 1e-3)) ** 2)
    return k / k.sum()


def _conv_along(x, k, dim):
    """Real 1D convolution of a tensor along dim, replicate padding."""
    x = x.movedim(dim, -1)
    shape = x.shape
    y = F.conv1d(F.pad(x.reshape(-1, 1, shape[-1]), (len(k) // 2,) * 2, mode="replicate"), k[None, None])
    return y.reshape(shape).movedim(-1, dim)


def _banded_conv(x, sigma_rows, dim, n_bands=5):
    """Gaussian convolution of (nz, nx) x along dim with a sigma (px) that varies per row.

    Convolves with a few band sigmas and interpolates per row between the
    nearest two (sigma varies slowly with depth).
    """
    lo, hi = float(sigma_rows.min()), float(sigma_rows.max())
    if hi - lo < 1e-3:
        return _conv_along(x, _gauss_kernel(lo, x.device), dim)
    bands = torch.linspace(lo, hi, n_bands, device=x.device)
    outs = torch.stack([_conv_along(x, _gauss_kernel(float(s), x.device), dim) for s in bands])
    pos = torch.clamp((sigma_rows - lo) / (bands[1] - bands[0]), 0, n_bands - 1 - 1e-4)
    i0 = pos.long()
    frac = (pos - i0)[:, None]
    rr = torch.arange(x.shape[0], device=x.device)
    return outs[i0, rr] * (1 - frac) + outs[i0 + 1, rr] * frac


def _resize_axis(x, n, dim, mode):
    """Resize one axis: "linear" interpolation (scan conversion), "area" average, or "nearest"."""
    if x.shape[dim] == n:
        return x
    y = x.movedim(dim, -1)
    shape = y.shape
    y = y.reshape(-1, 1, shape[-1])
    if mode == "nearest":
        y = F.interpolate(y, size=n, mode="nearest")
    elif mode == "area" and n < shape[-1]:
        y = F.adaptive_avg_pool1d(y, n)
    else:
        y = F.interpolate(y, size=n, mode="linear", align_corners=False)
    return y.reshape(*shape[:-1], n).movedim(-1, dim)


def spectral_attenuation_lut(spec, power, n=512, a_max=3.0):
    """Pulse-echo spectrum after power-law attenuation, as a function of the
    one-way path integral A = int alpha0 ds (Np / MHz^y).

    Amplitude spectrum S(f) = exp(-(f-fc)^2 / (2 sf^2)), sf = B fc / 2.355
    (B = -6 dB pulse-echo fractional bandwidth); received S(f) exp(-2 A f^y).
    Returns A grid and, per A: intensity factor E(A) relative to A = 0,
    centroid frequency (MHz) and amplitude-spectrum sigma (MHz).
    """
    fc = spec.fc / 1e6
    sf = spec.bandwidth * fc / 2.355
    f = np.linspace(max(1e-3, fc - 6 * sf), fc + 6 * sf, 2048)
    s2 = np.exp(-((f - fc) ** 2) / sf ** 2)                       # |S|^2
    A = np.linspace(0, a_max, n)
    P = s2[None] * np.exp(-4 * A[:, None] * f[None] ** power)
    tot = P.sum(1)
    E = tot / s2.sum()
    fbar = (P * f).sum(1) / tot
    var = (P * (f - fbar[:, None]) ** 2).sum(1) / tot
    return A, E, fbar, np.sqrt(2 * var)                           # intensity sigma -> amplitude sigma


# ----------------------------------------------------------------- renderer
class PreviewRenderer:
    def __init__(self, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.vol = None
        self._labels = self._props = self._zsmooth = None
        self.echo_lut = torch.tensor(tissue.lut(tissue.ECHOGENICITY), device=self.device)
        self.color_lut = torch.tensor(tissue.color_lut(), device=self.device)
        self.lesion_lut = torch.zeros(256, dtype=torch.bool, device=self.device)
        self.lesion_lut[list(tissue.LESIONS)] = True
        self._att_luts = {}

    # ------------------------------------------------------------- volume
    def set_volume(self, vol):
        self.vol = None
        self._labels = self._props = self._zsmooth = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(self.device, torch.float32)
        lab = vol.label
        # per-tissue mean impedance (texture-free) for specular boundaries
        counts = np.bincount(lab.ravel(), minlength=256)
        zsum = np.bincount(lab.ravel(), weights=vol.z.ravel().astype(np.float64), minlength=256)
        zlut = np.where(counts > 0, zsum / np.maximum(counts, 1), 1.5).astype(np.float32)
        zlut[tissue.BACKGROUND] = 1.5
        self.z_lut = torch.from_numpy(zlut).to(self.device)
        c = vol.c if vol.c is not None else np.full(lab.shape, C0, np.float32)
        self._labels = t(lab)[None, None]
        self._props = torch.stack([t(vol.alpha), t(c)])[None]                 # (1, 2, A, U, V)
        # smoothed tissue-impedance field: boundary normals for the angle dependence
        zs = self.z_lut[self._labels[0, 0].long()]
        k = _gauss_kernel(0.3 / vol.voxel_mm, self.device)
        for d in range(3):
            zs = _conv_along(zs, k, d)
        self._zsmooth = zs[None, None]
        self._norm = torch.tensor([(n - 1) * vol.voxel_mm / 2 for n in lab.shape],
                                  device=self.device, dtype=torch.float32)
        self.seed = zlib.crc32(vol.id.encode()) % 100_000  # stable across processes
        self.vol = vol

    def _sample(self, volume, pts, mode):
        """pts (..., 3) mm in (a, u, v) -> (C, ...) values (border-clamped)."""
        shape = pts.shape[:-1]
        g = (pts.reshape(1, 1, 1, -1, 3) / self._norm - 1).flip(-1)  # grid_sample order (v, u, a)
        out = F.grid_sample(volume, g, mode=mode, padding_mode="border", align_corners=True)
        return out.reshape(volume.shape[1], *shape)

    def _att_lut(self, spec, power):
        key = (spec.name, round(power, 4))
        if key not in self._att_luts:
            A, E, fbar, sf = spectral_attenuation_lut(spec, power)
            self._att_luts[key] = [torch.tensor(a, device=self.device, dtype=torch.float32) for a in (A, E, fbar, sf)]
        return self._att_luts[key]

    @staticmethod
    def _interp(lut_x, lut_y, x):
        """Linear interpolation on a uniform grid (clamped)."""
        n = len(lut_x)
        pos = torch.clamp(x / float(lut_x[-1]) * (n - 1), 0, n - 1 - 1e-4)
        i0 = pos.long()
        f = pos - i0
        return lut_y[i0] * (1 - f) + lut_y[i0 + 1] * f

    @staticmethod
    def _interp_vals(xp, fp, x):
        """np.interp for 1D tensors (xp sorted)."""
        if len(xp) == 1:
            return fp.expand_as(x)
        i = torch.clamp(torch.searchsorted(xp, x), 1, len(xp) - 1)
        x0, x1, f0, f1 = xp[i - 1], xp[i], fp[i - 1], fp[i]
        t = ((x - x0) / (x1 - x0).clamp(min=1e-9)).clamp(0, 1)
        return f0 + (f1 - f0) * t

    # ------------------------------------------------------------ speckle
    def _speckle(self, origin, axes, offs, nx, nz, dx, dz, width, sigma_y, k2, max_points=8_000_000):
        """Sum of every lattice scatterer inside the elevation beam, splatted to pixels.

        Each scatterer is counted once, weighted by the elevation beam profile at
        its true elevation, phased by its true depth along its line and split
        linearly between the two nearest rows, so a small probe move changes the
        image smoothly. Returns complex (nz, nx), unit variance for echogenicity 1.
        """
        dev, cell = self.device, scatter.SCATTER_CELL_MM
        e_lat, e_ax, e_el = axes
        step = cell / np.sqrt(2)                       # dense enough to hit (nearly) every cell
        depth = nz * dz
        ny_max = 2 * float(sigma_y(torch.linspace(0, depth, 64, device=dev)).max()) + cell
        slab = 2 * ny_max * (width + 2 * cell)         # mm^2 per mm of depth
        chunk = max(1.0, max_points * step ** 3 / slab)
        xs = torch.arange(-width / 2 - cell, width / 2 + cell, step, device=dev)
        x_cols = torch.linspace(-width / 2, width / 2, nx, device=dev)
        xs_off = torch.from_numpy(np.interp(xs.cpu().numpy(), x_cols.cpu().numpy(), offs.cpu().numpy())).to(dev, torch.float32)
        re = torch.zeros((nz + 1) * nx, device=dev)
        im = torch.zeros((nz + 1) * nx, device=dev)
        z0 = 0.0
        while z0 < depth:
            z1 = min(depth, z0 + chunk)
            zs = torch.arange(max(-cell, z0 - cell), z1 + cell, step, device=dev)
            ys = torch.arange(-ny_max, ny_max + step, step, device=dev)
            Z, X, Y = torch.meshgrid(zs, torch.arange(len(xs), device=dev, dtype=zs.dtype), ys, indexing="ij")
            keep = Y.abs() <= 2 * sigma_y(Z.clamp(min=0)) + cell
            Z, X, Y = Z[keep], X[keep].long(), Y[keep]
            pts = origin + xs[X, None] * e_lat + (Z + xs_off[X])[:, None] * e_ax + Y[:, None] * e_el
            idx = scatter.unique_cells(pts, cell)
            del pts, Z, X, Y, keep
            amp, pos = scatter.from_index(idx, cell, self.seed)
            rel = pos - origin
            col = torch.round((rel @ e_lat + width / 2) / dx).long()
            ok = (col >= 0) & (col < nx)
            col, rel, amp, pos = col[ok], rel[ok], amp[ok], pos[ok]
            zd = rel @ e_ax - offs[col]                                     # depth along its own line
            ok = (zd >= z0) & (zd < z1) & (zd >= 0) & (zd < depth)          # each scatterer in one chunk
            col, zd, amp, pos, yv = col[ok], zd[ok], amp[ok], pos[ok], (rel @ e_el)[ok]
            w = torch.exp(-0.5 * (yv / sigma_y(zd)) ** 2)
            echo = self.echo_lut[self._sample(self._labels, pos, "nearest")[0].long()]
            val = w * echo * amp
            rf = zd / dz - 0.5                                              # fractional row (centres at (i+0.5) dz)
            r0 = torch.floor(rf).clamp(min=-1).long()
            fr = (rf - r0).clamp(0, 1)
            cr, ci = val * torch.cos(k2 * zd), val * torch.sin(k2 * zd)
            for rr, wt in ((r0, 1 - fr), (r0 + 1, fr)):                     # linear splat onto two rows
                okr = (rr >= 0) & (rr < nz)
                flat = rr[okr] * nx + col[okr]
                re.index_add_(0, flat, (cr * wt)[okr])
                im.index_add_(0, flat, (ci * wt)[okr])
            z0 = z1
        zc = (torch.arange(nz, device=dev, dtype=torch.float32) + 0.5) * dz
        # expected scatterers per pixel x integral of w^2 over elevation (sigma sqrt(pi))
        norm = torch.sqrt(dx * dz / cell ** 3 * sigma_y(zc) * np.sqrt(np.pi))
        return torch.complex(re[:nz * nx], im[:nz * nx]).reshape(nz, nx) / norm[:, None]

    # ------------------------------------------------------------ specular
    def _specular(self, plane, sig_y, e_ax, e_el, dz, k2, f_number, n_planes):
        """Boundary echoes: R of the mean tissue impedances, at the boundary depth,
        weighted by the elevation beam and an angle factor
        (1-s) exp(-theta^2 / (2 theta0^2)) + s cos(theta), theta0 = 0.25 / F#, s = 0.25."""
        dev = self.device
        nz, nx = plane.shape[:2]
        out_re = torch.zeros(nz + 1, nx, device=dev)
        out_im = torch.zeros(nz + 1, nx, device=dev)
        ts = torch.linspace(-1.5, 1.5, max(1, n_planes), device=dev) if n_planes > 1 else torch.zeros(1, device=dev)
        ws = torch.exp(-0.5 * ts ** 2)
        ws = ws / ws.sum()
        theta0, s_diff, h = 0.25 / f_number, 0.25, 0.2
        for t, w in zip(ts, ws):
            pts = plane + (t * sig_y)[..., None] * e_el
            zl = self.z_lut[self._sample(self._labels, pts, "nearest")[0].long()]
            r = (zl[1:] - zl[:-1]) / (zl[1:] + zl[:-1])                       # between rows i and i+1
            hit = r.abs() > 1e-4
            if not hit.any():
                continue
            iz, ix = torch.nonzero(hit, as_tuple=True)
            pb = 0.5 * (pts[iz, ix] + pts[iz + 1, ix])                         # boundary point
            grad = torch.stack([self._sample(self._zsmooth, pb + h * e, "bilinear")[0]
                                - self._sample(self._zsmooth, pb - h * e, "bilinear")[0]
                                for e in torch.eye(3, device=dev)], -1)
            nrm = grad / grad.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            cos = (nrm @ e_ax).abs().clamp(max=1.0)
            theta = torch.acos(cos)
            factor = (1 - s_diff) * torch.exp(-0.5 * (theta / theta0) ** 2) + s_diff * cos
            zb = (iz + 1).float() * dz                                         # boundary depth
            amp = 0.5 * w * r[iz, ix] * factor          # boundary sits between rows iz and iz+1
            for o, comp in ((out_re, amp * torch.cos(k2 * zb)), (out_im, amp * torch.sin(k2 * zb))):
                o.index_put_((iz, ix), comp, accumulate=True)
                o.index_put_((iz + 1, ix), comp, accumulate=True)
        return torch.complex(out_re[:nz], out_im[:nz])

    # ------------------------------------------------------------- render
    @torch.no_grad()
    def render(self, pose: ProbePose, p: ImagingParams):
        """Returns (rgb uint8 (H, W, 3), meta dict, tissue labels uint8 (H, W)) on the output grid."""
        t0 = time.perf_counter()
        vol, dev = self.vol, self.device
        spec = PROBES[p.probe]
        lam0 = spec.wavelength * 1e3                     # mm at fc
        width, h_el, zf = spec.aperture * 1e3, spec.height * 1e3, spec.elev_focus * 1e3
        fc = spec.fc / 1e6
        sig_f0 = spec.bandwidth * fc / 2.355             # MHz, amplitude pulse-echo spectrum
        sig_z0 = spec.c * 1e-3 / (2 * np.pi * sig_f0) / 2  # mm, two-way envelope sigma at fc
        dz = float(np.clip(sig_z0 / 1.5, 0.02, 0.05))     # internal axial sampling
        dx = float(np.clip(p.px_x_mm, 0.1, 0.2))          # internal lateral sampling
        nx = int(round(width / dx)) + 1
        s_max = p.depth_mm * 1.12                         # true path covering the displayed (c0) depth
        nz = int(round(s_max / dz))
        x = torch.linspace(-width / 2, width / 2, nx, device=dev)
        z = (torch.arange(nz, device=dev, dtype=torch.float32) + 0.5) * dz   # true path length along a line

        origin_np, axes_np = pose.frame(vol.surface, vol.voxel_mm, spec)
        offs_np = skin_offsets(vol.surface, vol.voxel_mm, origin_np, axes_np, x.cpu().numpy(),
                               conform=p.conform, standoff=pose.standoff)
        offs = torch.from_numpy(offs_np).to(dev)
        origin = torch.tensor(origin_np, device=dev, dtype=torch.float32)
        e_lat, e_ax, e_el = torch.tensor(axes_np, device=dev, dtype=torch.float32)
        plane = origin + x[None, :, None] * e_lat + (z[:, None] + offs[None])[..., None] * e_ax   # (nz, nx, 3)

        # center plane: labels, attenuation, speed of sound
        lab_c = self._sample(self._labels, plane, "nearest")[0].long()
        alpha_c, c_c = self._sample(self._props, plane, "bilinear")
        A = torch.cumsum(alpha_c * dz * 1e-3, 0)                                 # one-way Np / MHz^y
        if p.shadow_blur_mm > 0:  # a finite-width beam fills shadows in from the sides
            A = _conv_along(A, _gauss_kernel(p.shadow_blur_mm / dx, dev), 1)
        A_lut, E_lut, fbar_lut, sf_lut = self._att_lut(spec, vol.alpha_power)
        att_gain = self._interp(A_lut, E_lut, A)                                  # two-way intensity factor
        A_row = A.mean(1)
        lam_row = spec.c * 1e-3 / self._interp(A_lut, fbar_lut, A_row)           # mm, local mean wavelength
        sig_z_row = spec.c * 1e-3 / (2 * np.pi * self._interp(A_lut, sf_lut, A_row)) / 2

        def lam_at(depth):
            return lam_row[torch.clamp((depth / dz).long(), 0, nz - 1)]

        # two-way elevation beam: diffraction at focus (0.886 lam zf / h), geometric off focus
        def sigma_y(depth):
            lam = lam_at(depth)
            return torch.sqrt((0.886 * lam * zf / h_el) ** 2 + (h_el * (1 - depth / zf)) ** 2 / 2) / 2.355

        k2 = 4 * np.pi / lam0                                                     # 2k at fc, rad/mm
        speckle = self._speckle(origin, (e_lat, e_ax, e_el), offs, nx, nz, dx, dz, width, sigma_y, k2)
        spec_echo = self._specular(plane, sigma_y(z)[:, None], e_ax, e_el, dz, k2, p.lateral_f_number,
                                   p.elevation_samples)

        # PSF: axial and lateral widths follow the local spectrum
        kz0 = _gauss_kernel(sig_z0 / dz, dev)
        sig_x_row = lam_row * torch.clamp(z / width, min=p.lateral_f_number) / 2.355 / dx
        ref = float((kz0 ** 2).sum() * (_gauss_kernel(float(sig_x_row[0]), dev) ** 2).sum())
        # a perpendicular boundary with reflection R peaks at specular * R speckle-rms units
        rf = speckle + (p.specular * np.sqrt(ref) / float(kz0.max())) * spec_echo
        parts = []
        for comp in (rf.real, rf.imag):
            comp = _banded_conv(comp, sig_z_row / dz, 0)                          # axial, sigma per depth
            parts.append(_banded_conv(comp, sig_x_row, 1))                        # lateral, sigma per row
        # sum-normalised kernels lower the speckle energy as the PSF widens with depth;
        # echo energy is conserved (its loss is E(A)), so restore sum(k^2) per row
        k2sum = lambda sig: torch.stack([(_gauss_kernel(float(v), dev) ** 2).sum() for v in sig])
        rows_z, rows_x = sig_z_row / dz, sig_x_row
        uz, ux = torch.unique(torch.round(rows_z * 20) / 20), torch.unique(torch.round(rows_x * 20) / 20)
        ez = self._interp_vals(uz, k2sum(uz), rows_z)
        ex = self._interp_vals(ux, k2sum(ux), rows_x)
        energy = float((kz0 ** 2).sum()) * float((_gauss_kernel(float(sig_x_row[0]), dev) ** 2).sum()) / (ez * ex)
        env2 = (parts[0] ** 2 + parts[1] ** 2) * energy[:, None] * att_gain      # intensity (nz, nx)

        # display depth: the scanner maps time to depth with c0
        z_app = torch.cumsum(C0 / c_c * dz, 0) - 0.5 * dz * C0 / c_c[:1]          # (nz, nx), monotonic
        n_app = int(round(p.depth_mm / dz))
        targets = ((torch.arange(n_app, device=dev, dtype=torch.float32) + 0.5) * dz)
        zt = z_app.t().contiguous()                                               # (nx, nz)
        idx = torch.searchsorted(zt, targets.expand(nx, -1).contiguous()).clamp(1, nz - 1)
        lo, hi = torch.gather(zt, 1, idx - 1), torch.gather(zt, 1, idx)
        frac = ((targets[None] - lo) / (hi - lo).clamp(min=1e-6)).clamp(0, 1)
        e2t = env2.t()
        env2 = (torch.gather(e2t, 1, idx - 1) * (1 - frac) + torch.gather(e2t, 1, idx) * frac).t()
        lab_c = torch.gather(lab_c.t(), 1, torch.where(frac < 0.5, idx - 1, idx)).t()
        # TGC is a function of receive time, i.e. of displayed depth
        env2 = env2 * torch.exp(4 * p.tgc_db_cm_mhz * 100 * NP_PER_DB * fc * targets[:, None] * 1e-3)

        # scan conversion to the preset pixel grid: interpolate the envelope amplitude
        # (area-averaging intensity would smooth the speckle and break its statistics)
        out_h = max(2, int(round(p.depth_mm / p.px_z_mm)))
        out_w = max(2, int(round(width / p.px_x_mm)))
        env2 = _resize_axis(_resize_axis(env2.sqrt(), out_h, 0, "linear"), out_w, 1, "linear") ** 2
        lab_o = _resize_axis(_resize_axis(lab_c.float(), out_h, 0, "nearest"), out_w, 1, "nearest").long()
        self.last_intensity = env2       # linear echo intensity before compression (validation)
        db = 10 * torch.log10(env2 / ref + 1e-12)
        if p.speckle_reduction > 0:
            ks = _gauss_kernel(p.speckle_reduction, dev)
            db = _conv_along(_conv_along(db, ks, 0), ks, 1)
        img = torch.clamp((db + p.gain_db) / p.dynamic_range_db + 1, 0, 1)
        rgb = (img[..., None] * 255).expand(-1, -1, 3).clone()

        mask = self.lesion_lut[lab_o]
        if p.overlay == "labels":
            rgb = 0.6 * rgb + 0.4 * self.color_lut[lab_o].float()
        elif p.overlay == "contour" and mask.any():
            m = mask.float()[None, None]
            edge = (F.max_pool2d(m, 3, 1, 1) + F.max_pool2d(-m, 3, 1, 1))[0, 0] > 0   # dilate - erode
            rgb[edge] = torch.tensor([60.0, 255.0, 60.0], device=dev)
        out = rgb.clamp(0, 255).to(torch.uint8).cpu().numpy()
        px_x, px_z = width / out_w, p.depth_mm / out_h
        lesion_types = sorted({tissue.NAMES[int(v)] for v in lab_o[mask].unique().tolist()}) if mask.any() else []
        if dev.type == "cuda":
            torch.cuda.synchronize()
        meta = {
            "origin_mm": origin_np.tolist(), "axes": axes_np.tolist(), "face_a_mm": float(origin_np[0]),
            "width_mm": width, "depth_mm": p.depth_mm, "px_x_mm": px_x, "px_z_mm": px_z,
            "skin_offset_mm": [float(offs_np.min()), float(offs_np.max())],
            # line start offsets at 33 positions across the array, for drawing the scan plane
            "plane_offsets_mm": offs_np[np.linspace(0, nx - 1, 33).round().astype(int)].tolist(),
            "lesion_area_mm2": round(float(mask.sum()) * px_x * px_z, 2), "lesion_types": lesion_types,
            "mean_frequency_mhz": [round(float(fc), 2), round(float(spec.c * 1e-3 / lam_row[-1]), 2)],
            "internal_dz_mm": dz,
            "render_ms": round((time.perf_counter() - t0) * 1e3, 1),
        }
        return out, meta, lab_o.to(torch.uint8).cpu().numpy()
