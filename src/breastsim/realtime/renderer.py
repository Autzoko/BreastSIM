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
from .beam import BeamModel, probe_spectrum, reverberation
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
    tgc_db_cm_mhz: float = 0.5        # TGC: compensates this one-way attenuation (dB/cm/MHz^y) over the pulse spectrum
    gray_gamma: float = 1.0           # display gray map: gray = (compressed level in [0, 1]) ** gamma
    specular: float = 30.0            # perpendicular boundary echo = specular * R * (speckle rms of echogenicity 1)
    lateral_f_number: float = 2.0     # receive F-number (dynamic aperture z / F#, limited by the array)
    tx_focus_mm: float = 0.0          # transmit focal depth; 0 = focused at every depth (multi-zone scanners)
    tx_f_number: float = 2.0          # transmit aperture = focus / F#
    aberration: bool = True           # speed-of-sound delays along the element paths (else ideal c0 beam)
    reverberation: bool = True        # second-order multiple reflections (probe face and interfaces)
    speckle_reduction: float = 1.0    # Gaussian sigma (output px) on the log image
    conform: bool = True              # probe presses the skin flat
    shadow_blur_mm: float = 0.0       # extra lateral fill-in of shadows (beam averaging is always applied)
    edge_shadow_db: float = 12.0      # max two-way loss per bulk boundary crossing (refraction / critical angle); 0 = off
    overlay: str = "contour"          # none | contour | labels
    elevation_samples: int = 5        # planes across the elevation beam (attenuation, boundaries)
    preset: str = DEFAULT_PRESET

    @classmethod
    def from_preset(cls, name, **overrides):
        p = PRESETS[name]
        fields = {k: getattr(p, k) for k in ("probe", "depth_mm", "px_x_mm", "px_z_mm", "gain_db",
                                             "dynamic_range_db", "tgc_db_cm_mhz", "gray_gamma",
                                             "speckle_reduction", "lateral_f_number", "conform")}
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

    Pulse-echo amplitude spectrum S(f) of MUST / PyMUST (beam.probe_spectrum: generalized
    normal probe response with the -6 dB fractional bandwidth, one-cycle excitation);
    received S(f) exp(-2 A f^y).
    Returns A grid and, per A: intensity factor E(A) relative to A = 0,
    centroid frequency (MHz) and amplitude-spectrum sigma (MHz).
    """
    fc = spec.fc / 1e6
    f = np.linspace(1e-3, 2 * fc, 2048)
    s2 = probe_spectrum(f, spec) ** 2                             # |S|^2
    A = np.linspace(0, a_max, n)
    P = s2[None] * np.exp(-4 * A[:, None] * f[None] ** power)
    tot = P.sum(1)
    E = tot / s2.sum()
    fbar = (P * f).sum(1) / tot
    var = (P * (f - fbar[:, None]) ** 2).sum(1) / tot
    return A, E, fbar, np.sqrt(2 * var)                           # intensity sigma -> amplitude sigma


def lesion_alpha(vol):
    """Attenuation map with each lesion set to its drawn slope (phantom.lesion_acoustics).

    Values are dB/cm/MHz (taken at the phantom's power law) converted to Np/m/MHz^y.
    Falls back to the phantom's own values when lesion ids or draws are missing.
    """
    lesions = vol.meta.get("lesions") or []
    if vol.lesion_id is None or not lesions or "attenuation_db_cm_mhz" not in lesions[0]:
        return vol.alpha
    lut = np.full(256, np.nan, np.float32)
    for les in lesions:
        lut[les["id"] + 1] = les["attenuation_db_cm_mhz"] * tissue.NP_PER_M_PER_DB_PER_CM
    over = lut[vol.lesion_id]
    return np.where(np.isfinite(over), over, vol.alpha.astype(np.float32))


def display_map(env2, ref, p):
    """Linear echo intensity (after TGC) -> display level in [0, 1].

    Vendor-style speckle smoothing on the amplitude (not on dB, where averaging a thin bright
    reflector with the black shadow behind it would erase the reflector), log compression to
    the dynamic range with the gain offset (as PyMUST bmode / k-Wave logCompression), then a
    power-law gray map.
    """
    if p.speckle_reduction > 0:
        ks = _gauss_kernel(p.speckle_reduction, env2.device)
        env2 = _conv_along(_conv_along(env2.sqrt(), ks, 0), ks, 1) ** 2
    db = 10 * torch.log10(env2 / ref + 1e-12)
    return torch.clamp((db + p.gain_db) / p.dynamic_range_db + 1, 0, 1) ** p.gray_gamma


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
        self.thin_lut = torch.zeros(256, dtype=torch.bool, device=self.device)
        self.thin_lut[list(tissue.THIN_STRUCTURES)] = True
        self.rough_lut = torch.tensor(tissue.lut(tissue.ROUGHNESS, default=tissue.DEFAULT_ROUGHNESS), device=self.device)
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
        csum = np.bincount(lab.ravel(), weights=c.ravel().astype(np.float64), minlength=256)
        clut = np.where(counts > 0, csum / np.maximum(counts, 1), C0).astype(np.float32)
        clut[tissue.BACKGROUND] = 1500.0
        self.c_lut = torch.from_numpy(clut).to(self.device)
        alpha = lesion_alpha(vol)
        self._labels = t(lab)[None, None]
        self._props = torch.stack([t(alpha), t(c)])[None]                     # (1, 2, A, U, V)
        # smoothed tissue-impedance field: boundary normals for the angle dependence
        zs = self.z_lut[self._labels[0, 0].long()]
        k = _gauss_kernel(0.3 / vol.voxel_mm, self.device)
        for d in range(3):
            zs = _conv_along(zs, k, d)
        self._zsmooth = zs[None, None]
        self._norm = torch.tensor([(n - 1) * vol.voxel_mm / 2 for n in lab.shape],
                                  device=self.device, dtype=torch.float32)
        self._origin = torch.tensor([vol.a0_mm, 0.0, 0.0], device=self.device, dtype=torch.float32)
        self.seed = zlib.crc32(vol.id.encode()) % 100_000  # stable across processes
        self.vol = vol

    def _sample(self, volume, pts, mode):
        """pts (..., 3) mm in (a, u, v) -> (C, ...) values (border-clamped)."""
        shape = pts.shape[:-1]
        g = ((pts.reshape(1, 1, 1, -1, 3) - self._origin) / self._norm - 1).flip(-1)  # grid order (v, u, a)
        out = F.grid_sample(volume, g, mode=mode, padding_mode="border", align_corners=True)
        return out.reshape(volume.shape[1], *shape)

    def _att_lut(self, spec, power):
        key = (spec.name, round(power, 4))
        if key not in self._att_luts:
            A, E, fbar, sf = spectral_attenuation_lut(spec, power)
            self._att_luts[key] = [torch.tensor(a, device=self.device, dtype=torch.float32) for a in (A, E, fbar, sf)]
        return self._att_luts[key]

    def tgc_gain(self, spec, power, depth_mm, slope_db_cm_mhz):
        """Intensity gain of the TGC curve at displayed depths (tensor, mm).

        Compensates a medium of the given one-way power-law attenuation for this probe's
        pulse-echo spectrum, including the downshift of the mean frequency with depth:
        gain = 1 / E(A), A = slope * depth (k-Wave's B-mode example compensates the same
        slope at fc only, which overcorrects deep echoes by the downshift).
        """
        A_lut, E_lut, _, _ = self._att_lut(spec, power)
        A = slope_db_cm_mhz * 100 * NP_PER_DB * depth_mm * 1e-3                   # Np / MHz^y, one way
        return 1 / self._interp(A_lut, E_lut, A).clamp(min=1e-30)

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
        # depth chunks sized by the local beam thickness (thin near the elevation focus)
        zgrid = torch.linspace(0, depth, 256, device=dev)
        half = (2 * sigma_y(zgrid) + cell).cpu().numpy()       # half slab thickness per depth
        zg = zgrid.cpu().numpy()
        xs = torch.arange(-width / 2 - cell, width / 2 + cell, step, device=dev)
        x_cols = torch.linspace(-width / 2, width / 2, nx, device=dev)
        xs_off = torch.from_numpy(np.interp(xs.cpu().numpy(), x_cols.cpu().numpy(), offs.cpu().numpy())).to(dev, torch.float32)
        re = torch.zeros((nz + 1) * nx, device=dev)
        im = torch.zeros((nz + 1) * nx, device=dev)
        z0 = 0.0
        while z0 < depth:
            # grow the chunk until it holds ~max_points samples
            z1, ny = z0, 0.0
            while z1 < depth:
                ny = max(ny, float(np.interp(min(depth, z1 + 0.5), zg, half)), float(np.interp(z1, zg, half)))
                if (z1 + 0.5 - z0 + 2 * cell) * (width + 2 * cell) * 2 * ny / step ** 3 > max_points and z1 > z0:
                    break
                z1 = min(depth, z1 + 0.5)
            zs = torch.arange(max(-cell, z0 - cell), z1 + cell, step, device=dev)
            ys = torch.arange(-ny, ny + step, step, device=dev)
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

    # ---------------------------------------------------------- interfaces
    def _interfaces(self, plane, sig_y, axes, dz, k2, p, spec, power):
        """Per elevation plane across the beam: attenuation, boundary transmission and echoes.

        At each tissue boundary along a line (rows i -> i+1), with normal n from a
        smoothed impedance field, incidence theta1 = acos|n . e_ax|:
          - Snell: sin theta2 = c2/c1 sin theta1 (total reflection beyond the
            critical angle), Fresnel intensity R(theta), and the share of the
            refracted beam still inside the aperture half-angle,
            exp(-(theta1 - theta2)^2 / (2 (2 theta0)^2)); the two-way transmission T^2
            is accumulated below the boundary -> refraction / edge shadows;
          - echo amplitude r = (Z2-Z1)/(Z2+Z1) times the angle factor
            (1-s) exp(-theta^2 / (2 theta0^2)) + s cos theta, theta0 = 0.25/F#, with the
            diffuse fraction s from tissue.ROUGHNESS (0.25 soft tissue, 0.8 bone).
        Attenuation is integrated per plane (one-way A = int alpha ds) and turned into
        intensity with the spectral LUT; planes are averaged in intensity with the
        elevation beam weights. Returns (echo complex (nz, nx), intensity factor
        (nz, nx), weighted A (nz, nx), coherent specular reflection complex (nz, nx),
        attenuation-only intensity factor (nz, nx), boundary loss density (nz, nx): the two-way
        -ln(T^2) deposited at each crossing, for the aperture-averaged transmission (beam.BeamModel).
        The last one is r times the specular lobe only, without the diffuse share: it drives
        the multiple reflections (rough bone scatters its energy away and casts clean
        shadows, the smooth pleura reverberates).
        """
        dev = self.device
        e_lat, e_ax, e_el = axes
        nz, nx = plane.shape[:2]
        A_lut, E_lut, _, _ = self._att_lut(spec, power)
        out_re = torch.zeros(nz + 1, nx, device=dev)
        out_im = torch.zeros(nz + 1, nx, device=dev)
        coh_re = torch.zeros(nz + 1, nx, device=dev)
        coh_im = torch.zeros(nz + 1, nx, device=dev)
        inten = torch.zeros(nz, nx, device=dev)
        a_mean = torch.zeros(nz, nx, device=dev)
        att_only = torch.zeros(nz, nx, device=dev)
        bloss = torch.zeros(nz, nx, device=dev)
        n = max(1, p.elevation_samples)
        ts = torch.linspace(-1.5, 1.5, n, device=dev) if n > 1 else torch.zeros(1, device=dev)
        ws = torch.exp(-0.5 * ts ** 2)
        ws = ws / ws.sum()
        theta0, h = 0.25 / p.lateral_f_number, 0.2
        # per-crossing cap of the two-way loss (Np): bulk interfaces up to edge_shadow_db, thin sheets
        # (ligaments, ducts, vessel walls, skin) much less - they are narrower than the beam, which
        # mostly passes around them
        cap_bulk = p.edge_shadow_db / 4.343
        cap_thin = min(p.edge_shadow_db, 3.0) / 4.343
        for t, w in zip(ts, ws):
            pts = plane + (t * sig_y)[..., None] * e_el
            lab = self._sample(self._labels, pts, "nearest")[0].long()
            alpha = self._sample(self._props[:, :1], pts, "bilinear")[0]
            A = torch.cumsum(alpha * dz * 1e-3, 0)                                 # one-way Np / MHz^y
            loss = torch.zeros(nz, nx, device=dev)                                 # two-way -ln(T^2) at boundaries
            hit = lab[1:] != lab[:-1]
            if hit.any():
                iz, ix = torch.nonzero(hit, as_tuple=True)
                l1, l2 = lab[iz, ix], lab[iz + 1, ix]
                z1, z2, c1, c2 = self.z_lut[l1], self.z_lut[l2], self.c_lut[l1], self.c_lut[l2]
                pb = 0.5 * (pts[iz, ix] + pts[iz + 1, ix])                         # boundary point
                grad = torch.stack([self._sample(self._zsmooth, pb + h * e, "bilinear")[0]
                                    - self._sample(self._zsmooth, pb - h * e, "bilinear")[0]
                                    for e in torch.eye(3, device=dev)], -1)
                gn = grad.norm(dim=-1, keepdim=True)
                nrm = grad / gn.clamp(min=1e-9)
                cos1 = torch.where(gn[:, 0] > 1e-6, (nrm @ e_ax).abs(), torch.ones_like(gn[:, 0])).clamp(max=1.0)
                sin2 = (c2 / c1) * torch.sqrt((1 - cos1 ** 2).clamp(min=0))
                total = sin2 >= 1
                cos2 = torch.sqrt((1 - sin2 ** 2).clamp(min=0))
                rr = ((z2 * cos1 - z1 * cos2) / (z2 * cos1 + z1 * cos2).clamp(min=1e-9)) ** 2
                theta = torch.acos(cos1)
                # refraction steers the transmitted beam by |theta1 - theta2|; energy bent outside the
                # aperture half-angle (~1/(2F#), i.e. 2x the specular lobe, which deviates by 2 theta)
                # leaves the line of sight -> lateral edge shadows
                bend = (theta - torch.asin(sin2.clamp(max=1.0))).abs()
                T = torch.where(total, torch.zeros_like(rr), (1 - rr) * torch.exp(-0.5 * (bend / (2 * theta0)) ** 2))
                if p.edge_shadow_db > 0:
                    thin = self.thin_lut[l1] | self.thin_lut[l2]
                    two_way = -2 * torch.log(T.clamp(min=1e-6))
                    capped = torch.where(thin, torch.clamp(0.3 * two_way, max=cap_thin), torch.clamp(two_way, max=cap_bulk))
                    # bone and air interfaces (|r| > 0.3) block the beam outright: no soft-tissue cap
                    two_way = torch.where(((z2 - z1) / (z2 + z1)).abs() > 0.3, two_way.clamp(max=30.0), capped)
                    loss.index_put_((iz + 1, ix), two_way, accumulate=True)
                # echo
                r = (z2 - z1) / (z2 + z1)
                s_diff = torch.maximum(self.rough_lut[l1], self.rough_lut[l2])      # rougher side dominates
                factor = (1 - s_diff) * torch.exp(-0.5 * (theta / theta0) ** 2) + s_diff * cos1
                zb = (iz + 1).float() * dz                                         # boundary depth
                # the echo leaves before the transmission loss of this crossing: last row above it
                amp = w * r * factor
                coh = w * r * (1 - s_diff) * torch.exp(-0.5 * (theta / theta0) ** 2)
                cs, sn = torch.cos(k2 * zb), torch.sin(k2 * zb)
                for o, comp in ((out_re, amp * cs), (out_im, amp * sn), (coh_re, coh * cs), (coh_im, coh * sn)):
                    o.index_put_((iz, ix), comp, accumulate=True)
            E = self._interp(A_lut, E_lut, A)
            inten += w * E * torch.exp(-torch.cumsum(loss, 0))
            att_only += w * E
            bloss += w * loss
            a_mean += w * A
        return (torch.complex(out_re[:nz], out_im[:nz]), inten, a_mean, torch.complex(coh_re[:nz], coh_im[:nz]),
                att_only, bloss)

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
        sig_f0 = float(self._att_lut(spec, vol.alpha_power)[3][0])   # MHz, amplitude pulse-echo spectrum sigma
        sig_z0 = spec.c * 1e-3 / (2 * np.pi * sig_f0) / 2  # mm, two-way envelope sigma at the probe face
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

        # center plane: labels and speed of sound (display mapping, ground truth)
        lab_c = self._sample(self._labels, plane, "nearest")[0].long()
        # bone and air interiors are anechoic and sit behind total shadows: their extreme speeds
        # are clamped to the soft-tissue range for the delays rather than stretching the display
        c_c = self._sample(self._props[:, 1:], plane, "bilinear")[0].clamp(1300.0, 1800.0)

        def sigma_y_lam(depth, lam):
            """Two-way elevation beam: diffraction at focus (0.886 lam zf / h), geometric off focus."""
            return torch.sqrt((0.886 * lam * zf / h_el) ** 2 + (h_el * (1 - depth / zf)) ** 2 / 2) / 2.355

        # attenuation, boundary transmission and echoes across the elevation beam
        k2 = 4 * np.pi / lam0                                                     # 2k at fc, rad/mm
        spec_echo, att_int, A, r_coh, att_only, bloss = self._interfaces(plane, sigma_y_lam(z, lam0)[:, None], (e_lat, e_ax, e_el),
                                                 dz, k2, p, spec, vol.alpha_power)
        A_lut, _, fbar_lut, sf_lut = self._att_lut(spec, vol.alpha_power)
        # per-depth spectral state, weighted by the transmitted intensity: fully shadowed
        # columns (behind bone or air) contribute nothing visible and must not shift it
        A_row = (A * att_int).sum(1) / att_int.sum(1).clamp(min=1e-12)
        lam_row = spec.c * 1e-3 / self._interp(A_lut, fbar_lut, A_row)           # mm, local mean wavelength
        sig_z_row = spec.c * 1e-3 / (2 * np.pi * self._interp(A_lut, sf_lut, A_row)) / 2

        def sigma_y(depth):
            return sigma_y_lam(depth, lam_row[torch.clamp((depth / dz).long(), 0, nz - 1)])

        speckle = self._speckle(origin, (e_lat, e_ax, e_el), offs, nx, nz, dx, dz, width, sigma_y, k2)

        # PSF: axial width follows the local spectrum (Gaussian envelope); lateral PSF from the
        # array beam model (focusing, apodization, side lobes, aberration; beam.BeamModel)
        slow = (1 / (c_c * 1e-3) - 1 / (C0 * 1e-3)) if p.aberration else torch.zeros_like(c_c)   # us/mm
        bm = BeamModel(spec, slow, x, dz, A_row, vol.alpha_power, p, loss=bloss)
        # boundary transmission (refraction / edge shadows) averaged over the element paths of the
        # transmit and receive apertures: shadows of structures smaller than the aperture cone fill
        # in with depth, as behind real ligaments and lobule margins
        att_int = att_only * bm.transmission_field(z)
        kz0 = _gauss_kernel(sig_z0 / dz, dev)
        sig_x_row = lam_row * torch.clamp(z / width, min=p.lateral_f_number) / 2.355 / dx
        k2x = bm.per_row(bm.k2sum0, z)                                            # ideal lateral kernel energy per row
        ref = float((kz0 ** 2).sum() * k2x[0])
        # sum-normalised kernels lower the (incoherent) speckle energy as the PSF widens with
        # depth; echo energy is conserved (its loss is E(A)), so restore sum(k^2) per row with the
        # ideal in-focus kernel: the transmit focal gain, edge truncation and aberration remain.
        # Coherent boundary echoes are pre-divided so the restoration leaves them unchanged.
        k2sum = lambda sig: torch.stack([(_gauss_kernel(float(v), dev) ** 2).sum() for v in sig])
        rows_z = sig_z_row / dz
        uz = torch.unique(torch.round(rows_z * 20) / 20)
        energy = ref / (self._interp_vals(uz, k2sum(uz), rows_z) * k2x)
        # transmitted intensity averaged over the two-way lateral beam (partial shadows)
        att_int = _banded_conv(att_int, sig_x_row, 1)
        if p.shadow_blur_mm > 0:                                                  # optional extra fill-in
            att_int = _conv_along(att_int, _gauss_kernel(p.shadow_blur_mm / dx, dev), 1)
        # a perpendicular boundary with reflection R peaks at specular * R speckle-rms units;
        # every echo is attenuated along its own path before it is blurred by the PSF, so a
        # bright reflector in front of a sudden loss (bone, air) keeps its full peak
        att_amp = att_int.sqrt()
        diffuse = speckle * att_amp
        specular = (p.specular * np.sqrt(ref) / float(kz0.max())) * spec_echo / energy.sqrt()[:, None] * att_amp
        rf = diffuse + specular
        if p.reverberation:
            # multiple reflections of specular echoes stay coherent across the aperture (replicas,
            # A-lines); those of diffuse scattering are incoherent (Pinton et al. 2014) and the
            # delay-and-sum gain over speckle, G = 2M/3 for M receive elements (Long et al. 2020),
            # suppresses them relative to the primary speckle
            m_rx = torch.clamp(z / p.lateral_f_number, max=width) / (spec.pitch * 1e3)
            g_das = (2 * m_rx.clamp(min=1.0) / 3)[:, None]
            rf = (rf + reverberation(specular, r_coh, att_int, spec.face_reflection, dz)
                  + reverberation(diffuse, r_coh, att_int, spec.face_reflection, dz) / g_das.sqrt())
        rf = torch.complex(_banded_conv(rf.real, sig_z_row / dz, 0), _banded_conv(rf.imag, sig_z_row / dz, 0))
        env2 = bm.lateral(rf, z).abs() ** 2 * energy[:, None]                     # intensity (nz, nx)

        # display depth: the scanner maps time to depth with c0; echoes arrive late by the
        # aperture-averaged excess delay of the transmit and receive paths (beam.BeamModel)
        z_app = torch.cummax(z[:, None] + bm.depth_shift(z), 0).values             # (nz, nx), monotonic
        n_app = int(round(p.depth_mm / dz))
        targets = ((torch.arange(n_app, device=dev, dtype=torch.float32) + 0.5) * dz)
        zt = z_app.t().contiguous()                                               # (nx, nz)
        idx = torch.searchsorted(zt, targets.expand(nx, -1).contiguous()).clamp(1, nz - 1)
        lo, hi = torch.gather(zt, 1, idx - 1), torch.gather(zt, 1, idx)
        frac = ((targets[None] - lo) / (hi - lo).clamp(min=1e-6)).clamp(0, 1)
        e2t = env2.t()
        env2 = (torch.gather(e2t, 1, idx - 1) * (1 - frac) + torch.gather(e2t, 1, idx) * frac).t()
        lab_c = torch.gather(lab_c.t(), 1, torch.where(frac < 0.5, idx - 1, idx)).t()

        # scan conversion to the preset pixel grid: interpolate the envelope amplitude
        # (area-averaging intensity would smooth the speckle and break its statistics)
        out_h = max(2, int(round(p.depth_mm / p.px_z_mm)))
        out_w = max(2, int(round(width / p.px_x_mm)))
        amp = env2.sqrt()
        # anti-alias before decimating: thin bright reflectors (bone, pleura, fascia) are
        # narrower than an output pixel and would otherwise be skipped at random
        for dim, n_out in ((0, out_h), (1, out_w)):
            ratio = amp.shape[dim] / n_out
            if ratio > 2.0:
                amp = _conv_along(amp, _gauss_kernel(0.42 * ratio, dev), dim)
        env2 = _resize_axis(_resize_axis(amp, out_h, 0, "linear"), out_w, 1, "linear") ** 2
        lab_o = _resize_axis(_resize_axis(lab_c.float(), out_h, 0, "nearest"), out_w, 1, "nearest").long()
        # TGC is a function of receive time, i.e. of displayed depth
        z_out = (torch.arange(out_h, device=dev, dtype=torch.float32) + 0.5) * (p.depth_mm / out_h)
        env2 = env2 * self.tgc_gain(spec, vol.alpha_power, z_out, p.tgc_db_cm_mhz)[:, None]
        self.last_intensity = env2       # linear echo intensity before compression (validation)
        self.last_ref = ref              # intensity of a unit-echogenicity speckle at 0 dB
        img = display_map(env2, ref, p)
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
            "aberration_rms_ns": round(float(bm.ab_rms_ns.median()), 1),
            "render_ms": round((time.perf_counter() - t0) * 1e3, 1),
        }
        return out, meta, lab_o.to(torch.uint8).cpu().numpy()
