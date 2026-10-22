"""Fast B-mode preview on the GPU.

Approximate but content-consistent with the high-fidelity generator:
- anatomy, impedance and attenuation come from the same canonical phantom;
- the image plane comes from probe.ProbePose.frame / probe.skin_offsets
  (optionally pressing the skin flat under the probe);
- speckle comes from the shared deterministic scatterer lattice
  (breastsim.scatter), with phase from each scatterer's true depth, and is
  integrated over the elevation beam, so neighbouring frames decorrelate
  like a real sweep;
- imaging is a separable Gaussian PSF (axial from the pulse bandwidth,
  lateral widening with depth) instead of a field simulation;
- specular echoes come from axial impedance changes, shadowing /
  enhancement from the cumulative attenuation along each line;
- output is resampled to the preset's pixel spacing, with optional
  vendor-style speckle reduction.
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


@dataclass
class ImagingParams:
    probe: str = "L12-hf"
    depth_mm: float = 42.0
    px_x_mm: float = 0.078            # output pixel spacing along the array
    px_z_mm: float = 0.078            # output pixel spacing in depth
    gain_db: float = -13.0
    dynamic_range_db: float = 52.0
    tgc_db_cm_mhz: float = 0.5        # depth gain compensation
    specular: float = 12.0            # weight of impedance-boundary echoes
    lateral_f_number: float = 2.0     # minimum receive F-number
    speckle_reduction: float = 1.0    # Gaussian sigma (output px) on the log image
    conform: bool = True              # probe presses the skin flat
    shadow_blur_mm: float = 1.0       # lateral beam extent averaging the attenuation (partial shadows)
    overlay: str = "contour"          # none | contour | labels
    elevation_samples: int = 5        # impedance planes across the elevation beam (specular term)
    preset: str = DEFAULT_PRESET

    @classmethod
    def from_preset(cls, name, **overrides):
        p = PRESETS[name]
        fields = {k: getattr(p, k) for k in ("probe", "depth_mm", "px_x_mm", "px_z_mm", "gain_db",
                                             "dynamic_range_db", "speckle_reduction",
                                             "lateral_f_number", "conform")}
        return cls(**{**fields, "preset": name, **overrides})


def _gauss_kernel(sigma_px, device):
    r = max(1, int(np.ceil(3 * sigma_px)))
    t = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    k = torch.exp(-0.5 * (t / max(sigma_px, 1e-3)) ** 2)
    return k / k.sum()


def _conv_along(x, k, dim):
    """Real 1D convolution of a 2D tensor along dim, replicate padding."""
    x = x.movedim(dim, -1)
    shape = x.shape
    y = F.conv1d(F.pad(x.reshape(-1, 1, shape[-1]), (len(k) // 2,) * 2, mode="replicate"), k[None, None])
    return y.reshape(shape).movedim(-1, dim)


class PreviewRenderer:
    DZ_MM = 0.05    # internal axial sampling

    def __init__(self, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.vol = None
        self._labels = None
        self._props = None
        self.echo_lut = torch.tensor(tissue.lut(tissue.ECHOGENICITY), device=self.device)
        self.color_lut = torch.tensor(tissue.color_lut(), device=self.device)
        self.lesion_lut = torch.zeros(256, dtype=torch.bool, device=self.device)
        self.lesion_lut[list(tissue.LESIONS)] = True

    # ------------------------------------------------------------- volume
    def set_volume(self, vol):
        self.vol = None
        self._labels = self._props = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(self.device, torch.float32)
        self._labels = t(vol.label)[None, None]
        self._props = torch.stack([t(vol.z), t(vol.alpha)])[None]
        self._norm = torch.tensor([(n - 1) * vol.voxel_mm / 2 for n in vol.label.shape],
                                  device=self.device, dtype=torch.float32)
        self.seed = zlib.crc32(vol.id.encode()) % 100_000  # stable across processes
        self.vol = vol

    def _sample(self, volume, pts, mode):
        """pts (..., 3) mm in (a, u, v) -> (C, ...) values (border-clamped)."""
        shape = pts.shape[:-1]
        g = (pts.reshape(1, 1, 1, -1, 3) / self._norm - 1).flip(-1)  # grid_sample order (v, u, a)
        out = F.grid_sample(volume, g, mode=mode, padding_mode="border", align_corners=True)
        return out.reshape(volume.shape[1], *shape)

    # ------------------------------------------------------------ speckle
    def _speckle(self, origin, axes, offs, nx, nz, dx, width, sigma_y, k2, max_points=8_000_000):
        """Sum of every lattice scatterer inside the elevation beam, binned to pixels.

        Each scatterer is counted once, weighted by the elevation beam profile at
        its true elevation and phased by its true depth along its line, so a
        small probe move changes the image smoothly (correct frame-to-frame
        speckle correlation). Returns complex (nz, nx), unit variance for
        echogenicity 1.
        """
        dev, cell = self.device, scatter.SCATTER_CELL_MM
        e_lat, e_ax, e_el = axes
        step = cell / np.sqrt(2)                       # dense enough to hit (nearly) every cell
        depth = nz * self.DZ_MM
        ny_max = 2 * float(sigma_y(torch.linspace(0, depth, 64, device=dev)).max()) + cell
        slab = 2 * ny_max * (width + 2 * cell)         # mm^2 per mm of depth
        chunk = max(1.0, max_points * step ** 3 / slab)
        xs = torch.arange(-width / 2 - cell, width / 2 + cell, step, device=dev)
        x_cols = torch.linspace(-width / 2, width / 2, nx, device=dev)
        xs_off = torch.from_numpy(np.interp(xs.cpu().numpy(), x_cols.cpu().numpy(), offs.cpu().numpy())).to(dev, torch.float32)
        re = torch.zeros(nz * nx, device=dev)
        im = torch.zeros(nz * nx, device=dev)
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
            row = torch.floor(zd / self.DZ_MM).long()
            ok = (zd >= z0) & (zd < z1) & (row >= 0) & (row < nz)           # each scatterer in one chunk
            col, row, zd, amp, pos, yv = col[ok], row[ok], zd[ok], amp[ok], pos[ok], (rel @ e_el)[ok]
            w = torch.exp(-0.5 * (yv / sigma_y(zd)) ** 2)
            echo = self.echo_lut[self._sample(self._labels, pos, "nearest")[0].long()]
            val = w * echo * amp
            flat = row * nx + col
            re.index_add_(0, flat, val * torch.cos(k2 * zd))
            im.index_add_(0, flat, val * torch.sin(k2 * zd))
            z0 = z1
        zc = (torch.arange(nz, device=dev, dtype=torch.float32) + 0.5) * self.DZ_MM
        # expected scatterers per pixel x integral of w^2 over elevation (sigma sqrt(pi))
        norm = torch.sqrt(dx * self.DZ_MM / cell ** 3 * sigma_y(zc) * np.sqrt(np.pi))
        return torch.complex(re, im).reshape(nz, nx) / norm[:, None]

    # ------------------------------------------------------------- render
    @torch.no_grad()
    def render(self, pose: ProbePose, p: ImagingParams):
        """Returns (rgb uint8 (H, W, 3), meta dict, tissue labels uint8 (H, W)) on the output grid."""
        t0 = time.perf_counter()
        vol, dev = self.vol, self.device
        spec = PROBES[p.probe]
        lam = spec.wavelength * 1e3                     # mm
        width, h_el, zf = spec.aperture * 1e3, spec.height * 1e3, spec.elev_focus * 1e3
        dx = float(np.clip(p.px_x_mm, 0.1, 0.2))        # internal lateral sampling
        nx = int(round(width / dx)) + 1
        nz = int(round(p.depth_mm / self.DZ_MM))
        x = torch.linspace(-width / 2, width / 2, nx, device=dev)
        z = (torch.arange(nz, device=dev, dtype=torch.float32) + 0.5) * self.DZ_MM

        origin_np, axes_np = pose.frame(vol.surface, vol.voxel_mm, spec)
        offs_np = skin_offsets(vol.surface, vol.voxel_mm, origin_np, axes_np, x.cpu().numpy(),
                               conform=p.conform, standoff=pose.standoff)
        offs = torch.from_numpy(offs_np).to(dev)
        origin = torch.tensor(origin_np, device=dev, dtype=torch.float32)
        e_lat, e_ax, e_el = torch.tensor(axes_np, device=dev, dtype=torch.float32)
        zt = z[:, None] + offs[None]                                              # depth along each line
        plane = origin + x[None, :, None] * e_lat + zt[..., None] * e_ax          # (nz, nx, 3)

        # two-way elevation beam width (diffraction at focus, geometric off focus)
        def sigma_y(depth):
            return torch.sqrt((lam * zf / h_el) ** 2 + (h_el * (1 - depth / zf)) ** 2) / np.sqrt(2) / 2.355

        k2 = 4 * np.pi / lam                                                      # 2k, rad/mm
        speckle = self._speckle(origin, (e_lat, e_ax, e_el), offs, nx, nz, dx, width, sigma_y, k2)

        # specular echoes: axial impedance change, averaged over the elevation beam
        sig_y = sigma_y(z)[:, None]
        refl = torch.zeros(nz, nx, device=dev)
        ts = torch.linspace(-1.5, 1.5, max(1, p.elevation_samples), device=dev)
        ws = torch.exp(-0.5 * ts ** 2)
        d = 2  # samples, impedance difference half-span (0.1 mm)
        for t, w in zip(ts, ws):
            imp = self._sample(self._props[:, :1], plane + (t * sig_y)[..., None] * e_el, "bilinear")[0]
            zp = torch.cat([imp[d:], imp[-1:].expand(d, -1)])
            zm = torch.cat([imp[:1].expand(d, -1), imp[:-d]])
            refl += w * (zp - zm) / (zp + zm)
        refl /= ws.sum()
        rf = speckle + p.specular * refl * torch.polar(torch.ones_like(z), k2 * z)[:, None]

        # attenuation along each line (center plane) and TGC
        center = self._sample(torch.cat([self._labels, self._props[:, 1:]], 1), plane, "nearest")
        lab_c, alpha_c = center[0].long(), center[1]
        f_mhz = spec.fc / 1e6
        att_np = torch.cumsum(alpha_c * f_mhz ** vol.alpha_power * self.DZ_MM * 1e-3, 0)
        if p.shadow_blur_mm > 0:  # a finite-width beam fills shadows in from the sides
            att_np = _conv_along(att_np, _gauss_kernel(p.shadow_blur_mm / dx, dev), 1)
        tgc_np = p.tgc_db_cm_mhz * 100 * NP_PER_DB * f_mhz * z[:, None] * 1e-3
        gain = torch.exp(2 * (tgc_np - att_np))

        # separable PSF: axial from the bandwidth, lateral widening with depth
        sig_f = spec.bandwidth * spec.fc / 2.355                                 # Hz
        sig_z = spec.c / (2 * np.pi * sig_f) / 2 * 1e3                            # mm, two-way
        kz = _gauss_kernel(sig_z / self.DZ_MM, dev)
        rf = torch.complex(_conv_along(rf.real, kz, 0), _conv_along(rf.imag, kz, 0))
        fwhm_x = lam * torch.clamp(z / width, min=p.lateral_f_number)
        sig_x = fwhm_x / 2.355 / dx                                               # px, per row
        bands = torch.linspace(float(sig_x.min()), float(sig_x.max()), 5, device=dev)
        outs = []
        for s in bands:
            k = _gauss_kernel(float(s), dev)
            outs.append(torch.stack([_conv_along(rf.real, k, 1), _conv_along(rf.imag, k, 1)]))
        outs = torch.stack(outs)                                                  # (B, 2, nz, nx)
        step = float(bands[1] - bands[0])
        if step < 1e-6:
            lat = outs[0]
        else:  # per row, interpolate between the two nearest bands
            pos = torch.clamp((sig_x - bands[0]) / step, 0, len(bands) - 1 - 1e-4)
            i0 = pos.long()
            frac = (pos - i0)[:, None, None]
            rr = torch.arange(nz, device=dev)
            lat = (outs[i0, :, rr] * (1 - frac) + outs[i0 + 1, :, rr] * frac).permute(1, 0, 2)
        env2 = (lat ** 2).sum(0) * gain ** 2                                      # intensity (nz, nx)

        # resample to the preset pixel grid (area-average intensity when shrinking)
        out_h = max(2, int(round(nz * self.DZ_MM / p.px_z_mm)))
        out_w = max(2, int(round(width / p.px_x_mm)))
        mode = "area" if out_h <= nz and out_w <= nx else "bilinear"
        env2 = F.interpolate(env2[None, None], size=(out_h, out_w), mode=mode)[0, 0]
        self.last_intensity = env2       # linear echo intensity before compression (validation)
        ref = float((kz ** 2).sum() * (_gauss_kernel(float(sig_x.median()), dev) ** 2).sum())
        db = 10 * torch.log10(env2 / ref + 1e-12)
        if p.speckle_reduction > 0:
            ks = _gauss_kernel(p.speckle_reduction, dev)
            db = _conv_along(_conv_along(db, ks, 0), ks, 1)
        img = torch.clamp((db + p.gain_db) / p.dynamic_range_db + 1, 0, 1)
        rgb = (img[..., None] * 255).expand(-1, -1, 3).clone()

        lab_o = F.interpolate(lab_c[None, None].float(), size=(out_h, out_w), mode="nearest")[0, 0].long()
        mask = self.lesion_lut[lab_o]
        if p.overlay == "labels":
            rgb = 0.6 * rgb + 0.4 * self.color_lut[lab_o].float()
        elif p.overlay == "contour" and mask.any():
            m = mask.float()[None, None]
            edge = (F.max_pool2d(m, 3, 1, 1) + F.max_pool2d(-m, 3, 1, 1))[0, 0] > 0   # dilate - erode
            rgb[edge] = torch.tensor([60.0, 255.0, 60.0], device=dev)
        out = rgb.clamp(0, 255).to(torch.uint8).cpu().numpy()
        px_x, px_z = width / out_w, nz * self.DZ_MM / out_h
        lesion_types = sorted({tissue.NAMES[int(v)] for v in lab_o[mask].unique().tolist()}) if mask.any() else []
        if dev.type == "cuda":
            torch.cuda.synchronize()
        meta = {
            "origin_mm": origin_np.tolist(), "axes": axes_np.tolist(), "face_a_mm": float(origin_np[0]),
            "width_mm": width, "depth_mm": nz * self.DZ_MM, "px_x_mm": px_x, "px_z_mm": px_z,
            "skin_offset_mm": [float(offs_np.min()), float(offs_np.max())],
            # line start offsets at 33 positions across the array, for drawing the scan plane
            "plane_offsets_mm": offs_np[np.linspace(0, nx - 1, 33).round().astype(int)].tolist(),
            "lesion_area_mm2": round(float(mask.sum()) * px_x * px_z, 2), "lesion_types": lesion_types,
            "render_ms": round((time.perf_counter() - t0) * 1e3, 1),
        }
        return out, meta, lab_o.to(torch.uint8).cpu().numpy()
