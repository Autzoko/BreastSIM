"""Fast B-mode preview on the GPU.

Approximate but content-consistent with the high-fidelity generator:
- anatomy, impedance and attenuation come from the same canonical phantom;
- speckle comes from the shared deterministic scatterer lattice
  (breastsim.scatter), with phase from each scatterer's true depth, and is
  integrated over the elevation beam, so neighbouring frames decorrelate
  like a real sweep;
- imaging is a separable Gaussian PSF (axial from the pulse bandwidth,
  lateral widening with depth) instead of a field simulation;
- specular echoes come from axial impedance changes, shadowing /
  enhancement from the cumulative attenuation along each line.
"""
import time
import zlib
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .. import scatter, tissue
from ..probe import PROBES, ProbePose

NP_PER_DB = 1 / 8.686


@dataclass
class ImagingParams:
    probe: str = "L11-5v"
    depth_mm: float = 40.0
    gain_db: float = -8.0
    dynamic_range_db: float = 55.0
    tgc_db_cm_mhz: float = 0.5        # depth gain compensation
    specular: float = 12.0            # weight of impedance-boundary echoes
    lateral_f_number: float = 2.0     # minimum receive F-number
    overlay: str = "contour"          # none | contour | labels
    elevation_samples: int = 24
    seed: int = 0


def _gauss_kernel(sigma_px, device):
    r = max(1, int(np.ceil(3 * sigma_px)))
    t = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    k = torch.exp(-0.5 * (t / max(sigma_px, 1e-3)) ** 2)
    return k / k.sum()


class PreviewRenderer:
    DX_MM = 0.1     # lateral sampling
    DZ_MM = 0.05    # axial sampling (output is pooled to 0.1 mm)

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
        self.vol = vol
        self._labels = None
        self._props = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(self.device, torch.float32)
        self._labels = t(vol.label)[None, None]
        self._props = torch.stack([t(vol.z), t(vol.alpha)])[None]
        self._norm = torch.tensor([(n - 1) * vol.voxel_mm / 2 for n in vol.label.shape],
                                  device=self.device, dtype=torch.float32)
        self.seed = zlib.crc32(vol.id.encode()) % 100_000  # stable across processes

    def _sample(self, volume, pts, mode):
        """pts (..., 3) mm in (a, u, v) -> (C, ...) values."""
        shape = pts.shape[:-1]
        g = pts.reshape(1, 1, 1, -1, 3) / self._norm - 1
        g = g.flip(-1)  # grid_sample wants (x=v, y=u, z=a)
        out = F.grid_sample(volume, g, mode=mode, padding_mode="border", align_corners=True)
        return out.reshape(volume.shape[1], *shape)

    # ------------------------------------------------------------- render
    @torch.no_grad()
    def render(self, pose: ProbePose, p: ImagingParams):
        t0 = time.perf_counter()
        vol, dev = self.vol, self.device
        spec = PROBES[p.probe]
        lam = spec.wavelength * 1e3                     # mm
        width, h_el, zf = spec.aperture * 1e3, spec.height * 1e3, spec.elev_focus * 1e3
        nx = int(round(width / self.DX_MM)) + 1
        nz = int(round(p.depth_mm / self.DZ_MM)) // 2 * 2
        x = torch.linspace(-width / 2, width / 2, nx, device=dev)
        z = (torch.arange(nz, device=dev, dtype=torch.float32) + 0.5) * self.DZ_MM

        origin, axes = pose.frame(vol.surface, vol.voxel_mm, spec)
        origin = torch.tensor(origin, device=dev, dtype=torch.float32)
        e_lat, e_ax, e_el = torch.tensor(axes, device=dev, dtype=torch.float32)
        plane = origin + x[None, :, None] * e_lat + z[:, None, None] * e_ax     # (nz, nx, 3)

        # two-way elevation beam width (diffraction at focus, geometric off focus)
        fwhm_y = torch.sqrt((lam * zf / h_el) ** 2 + (h_el * (1 - z / zf)) ** 2) / np.sqrt(2)
        sig_y = (fwhm_y / 2.355)[:, None]                                         # (nz, 1)

        k2 = 4 * np.pi / lam                                                      # 2k, rad/mm
        speckle = torch.zeros(nz, nx, dtype=torch.complex64, device=dev)
        refl = torch.zeros(nz, nx, device=dev)
        ts = torch.linspace(-2.5, 2.5, p.elevation_samples, device=dev)
        ws = torch.exp(-0.5 * ts ** 2)
        d = 2  # samples, impedance difference half-span (0.1 mm)
        for t, w in zip(ts, ws):
            pts = plane + (t * sig_y)[..., None] * e_el
            lab = self._sample(self._labels, pts, "nearest")[0]
            imp = self._sample(self._props[:, :1], pts, "bilinear")[0]
            amp, pos = scatter.cells(pts, seed=self.seed)
            depth = (pos - origin) @ e_ax
            echo = self.echo_lut[lab.long()] * amp
            speckle += w * echo * torch.polar(torch.ones_like(depth), k2 * depth)
            zp = torch.cat([imp[d:], imp[-1:].expand(d, -1)])
            zm = torch.cat([imp[:1].expand(d, -1), imp[:-d]])
            refl += w * (zp - zm) / (zp + zm)
        speckle /= torch.sqrt((ws ** 2).sum())
        refl /= ws.sum()
        rf = speckle + p.specular * refl * torch.polar(torch.ones_like(z), k2 * z)[:, None]

        # attenuation along each line (center plane) and TGC
        center = self._sample(torch.cat([self._labels, self._props[:, 1:]], 1), plane, "nearest")
        lab_c, alpha_c = center[0].long(), center[1]
        f_mhz = spec.fc / 1e6
        att_np = torch.cumsum(alpha_c * f_mhz ** vol.alpha_power * self.DZ_MM * 1e-3, 0)
        tgc_np = p.tgc_db_cm_mhz * 100 * NP_PER_DB * f_mhz * z[:, None] * 1e-3
        gain = torch.exp(2 * (tgc_np - att_np))

        # separable PSF: axial from bandwidth, lateral widening with depth
        sig_f = spec.bandwidth * spec.fc / 2.355                                 # Hz
        sig_z = spec.c / (2 * np.pi * sig_f) / 2 * 1e3                            # mm
        kz = _gauss_kernel(sig_z / self.DZ_MM, dev)
        ri = torch.view_as_real(rf).permute(2, 1, 0).reshape(2 * nx, 1, nz)      # lines along z
        ri = F.conv1d(F.pad(ri, (len(kz) // 2,) * 2, mode="replicate"), kz[None, None])
        rf = torch.view_as_complex(ri.reshape(2, nx, nz).permute(2, 1, 0).contiguous())
        fwhm_x = lam * torch.clamp(z / width, min=p.lateral_f_number)
        sig_x = fwhm_x / 2.355 / self.DX_MM
        bands = torch.linspace(float(sig_x.min()), float(sig_x.max()), 5, device=dev)
        rows = torch.view_as_real(rf).permute(0, 2, 1).reshape(2 * nz, 1, nx)
        outs = []
        for s in bands:
            k = _gauss_kernel(float(s), dev)
            outs.append(F.conv1d(F.pad(rows, (len(k) // 2,) * 2, mode="replicate"), k[None, None]))
        outs = torch.stack(outs).reshape(len(bands), nz, 2, nx)
        pos = torch.clamp((sig_x - bands[0]) / max(float(bands[1] - bands[0]), 1e-6), 0, len(bands) - 1 - 1e-4)
        i0 = pos.long()
        frac = (pos - i0)[:, None, None]
        rr = torch.arange(nz, device=dev)
        lat = outs[i0, rr] * (1 - frac) + outs[i0 + 1, rr] * frac if len(bands) > 1 else outs[0]
        env = torch.sqrt((lat ** 2).sum(1)) * gain                                # (nz, nx)

        # reference: rms of unit-variance white noise through the PSF
        ref = float(torch.sqrt((kz ** 2).sum() * (_gauss_kernel(float(sig_x.median()), dev) ** 2).sum()))
        env = torch.sqrt(F.avg_pool2d((env ** 2)[None, None], (2, 1))[0, 0])      # 0.1 mm square pixels
        db = 20 * torch.log10(env / ref + 1e-6)
        img = torch.clamp((db + p.gain_db) / p.dynamic_range_db + 1, 0, 1)
        rgb = (img[..., None] * 255).expand(-1, -1, 3).clone()

        lab_c = lab_c[::2]
        mask = self.lesion_lut[lab_c]
        if p.overlay == "labels":
            rgb = 0.6 * rgb + 0.4 * self.color_lut[lab_c].float()
        elif p.overlay == "contour" and mask.any():
            m = mask.float()[None, None]
            edge = (F.max_pool2d(m, 3, 1, 1) - (-F.max_pool2d(-m, 3, 1, 1)))[0, 0] > 0
            rgb[edge] = torch.tensor([60.0, 255.0, 60.0], device=dev)
        out = rgb.clamp(0, 255).to(torch.uint8).cpu().numpy()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        return out, {
            "face_a_mm": float(origin[0]), "pixel_mm": 2 * self.DZ_MM, "width_mm": width,
            "depth_mm": nz * self.DZ_MM, "lesion_pixels": int(mask.sum()),
            "render_ms": round((time.perf_counter() - t0) * 1e3, 1),
        }
