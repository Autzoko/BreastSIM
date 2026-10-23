"""Array beam model of the preview: probe spectrum, aberration, focusing, side lobes, reverberation.

Replaces the ideal separable Gaussian lateral PSF with the beam of the actual array, computed
per lateral block and depth band from the speed of sound along each element's path:

- probe response (pulse-echo) as in MUST / PyMUST (pfield, Param.getProbeFunction): a
  generalized normal window with the -6 dB fractional bandwidth and the exponent
  p = ln(126) / ln(2 fc / B), times the spectrum of a one-cycle windowed sine excitation
  (Param.getPulseSpectrumFunction, TXnow = 1);
- delays: for every element u and focal point (x_b, z) the excess travel time
  tau_u = int (1/c - 1/c0) ds along the straight element-to-point path through the phantom's
  speed of sound (near-field phase-screen / time-delay model of aberration, as in
  Field II aberration studies and Hinkelman, Mast & Waag's breast measurements);
  the apodization-weighted mean over the transmit and receive apertures is the bulk delay,
  which moves the echo in depth (the scanner maps time with c0); the residual is the
  aberration, which defocuses the beam;
- lateral PSF: narrowband exact-distance sum over the active elements (as pfield /
  Field II in the continuous-wave limit, element directivity sinc(pi w sin(theta) / lambda)),
  transmit focused at a fixed depth with a fixed F-number, dynamic receive focus with an
  F-number-limited aperture, Hann apodization on both (tx_focus_mm = 0: transmit focused at every
  depth, as multi-zone / retrospective transmit focusing on current scanners); the two-way PSF is the product of
  transmit and receive patterns, summed coherently over the attenuated pulse-echo spectrum.
  Side lobes, the transmit focal zone, edge truncation and aberration all follow from it;
- reverberation: second-order multiple reflections along each line (layered-medium model):
  probe face - interface - target, and interface - interface - target. Echoes arrive later by
  the extra round trip and are attenuated along it; specular sources give replicas
  (A-lines behind the pleura), diffuse ones a speckle-like haze.

Units: mm, microseconds, MHz (c0 = 1.54 mm/us).
"""
import numpy as np
import torch
import torch.nn.functional as F

C0_MM_US = 1.54
ELEMENT_FILL = 0.9      # element width / pitch (MUST's L11-5v: 0.27 mm elements on a 0.30 mm pitch)
# smoothing of the slowness map for the path delays (finite-frequency travel times average the medium
# over the Fresnel zone, ~1 mm radius at 10 MHz over 20-30 mm). 2 mm matches k-Wave point targets in a
# real phantom slice (2026-10-23, 9 targets 12-32 mm): energy outside +-1 mm 0.16 (k-Wave 0.03; 0.5 mm
# gave 0.4), aperture delay rms 7-73 ns (breast: 33 +- 7 ns in vivo, 67 +- 13 ns ex vivo); the bulk
# depth shift matches k-Wave within ~0.2 mm for any value
FRESNEL_SIGMA_MM = 2.0


def probe_spectrum(f_mhz, spec):
    """Pulse-echo amplitude spectrum (MUST): one-cycle sine excitation x probe pulse-echo response."""
    f = np.asarray(f_mhz, np.float64)
    fc, wb = spec.fc / 1e6, spec.bandwidth * spec.fc / 1e6
    p = np.log(126) / np.log(2 * fc / wb)
    probe = np.exp(-(np.abs(f - fc) / (wb / 2 / np.log(2) ** (1 / p))) ** p)
    T = 1 / fc
    sinc = lambda x: np.sinc(x / np.pi)
    pulse = np.abs(sinc(np.pi * T * (f - fc)) - sinc(np.pi * T * (f + fc)))
    return pulse * probe


def _hann(u, half):
    """Hann apodization over |u| <= half (0 outside)."""
    w = 0.5 * (1 + torch.cos(np.pi * u / half.clamp(min=1e-6)))
    return torch.where(u.abs() <= half, w, torch.zeros_like(w))


class BeamModel:
    """Per-render beam: block / band grid, lateral kernels, bulk delays, aberration statistics.

    slow: (nz, nx) excess slowness 1/c - 1/c0 (us/mm) on the internal grid (rows dz from the
    probe face, columns x mm); A_row: (nz,) one-way attenuation integral (Np/MHz^y) per row;
    loss: optional (nz, nx) two-way boundary transmission loss -ln(T^2) deposited at each crossing
    (vertical lines). It is integrated along every element path, and the apodization-weighted
    mean transmitted amplitude of the transmit and receive apertures gives the two-way intensity
    factor per block and band (transmission_field).
    """

    def __init__(self, spec, slow, x, dz, A_row, power, p, loss=None, block_mm=1.0, band_mm=1.0,
                 n_freq=7, n_path=48, half_kernel_mm=8.0):
        dev = slow.device
        self.dev, self.spec, self.x, self.dz = dev, spec, x, dz
        nz, nx = slow.shape
        dx = float(x[1] - x[0])
        self.dx = dx
        pitch = spec.pitch * 1e3
        width = spec.aperture * 1e3
        n_el = spec.n_elements
        xe = (torch.arange(n_el, device=dev, dtype=torch.float32) - (n_el - 1) / 2) * pitch
        z_max = nz * dz
        self.band_z = torch.arange(band_mm / 2, z_max + band_mm, band_mm, device=dev, dtype=torch.float32)
        self.block_x = torch.linspace(float(x[0]), float(x[-1]), max(2, int(round(width / block_mm)) + 1), device=dev)
        nd, nb = len(self.band_z), len(self.block_x)
        f_rx = p.lateral_f_number
        z_f, f_tx = p.tx_focus_mm, p.tx_f_number
        half_rx = (self.band_z / f_rx / 2).clamp(min=pitch)                     # (nd,)
        z_tx = torch.full_like(self.band_z, z_f) if z_f > 0 else self.band_z      # transmit focus per band
        half_tx = (z_tx / f_tx / 2).clamp(min=pitch, max=width / 2)
        half_rx = half_rx.clamp(max=width / 2)
        half_max = float(max(half_rx.max(), half_tx.max()))
        n_loc = int(np.ceil(2 * half_max / pitch)) + 2
        # local element window per block: indices of the n_loc elements around the block centre
        i0 = torch.clamp(torch.round((self.block_x + width / 2) / pitch - 0.5 - n_loc / 2).long(), 0,
                         max(0, n_el - n_loc))
        idx = i0[:, None] + torch.arange(min(n_loc, n_el), device=dev)[None]     # (nb, nl), inside the array
        xe_loc = xe[idx]                                                          # (nb, nl)
        u = xe_loc - self.block_x[:, None]                                        # element offset from block
        self.half_k = int(np.ceil(half_kernel_mm / dx))
        xi = torch.arange(-self.half_k, self.half_k + 1, device=dev, dtype=torch.float32) * dx   # kernel offsets

        # frequencies and per-band spectral weights (attenuated pulse-echo amplitude spectrum)
        fc = spec.fc / 1e6
        fs = np.linspace(0.3 * fc, 1.7 * fc, n_freq)
        S = torch.tensor(probe_spectrum(fs, spec), device=dev, dtype=torch.float32)
        fs_t = torch.tensor(fs, device=dev, dtype=torch.float32)
        rows = torch.clamp((self.band_z / dz).long(), 0, nz - 1)
        A_band = A_row[rows]                                                      # (nd,)
        wf = S[None] * torch.exp(-2 * A_band[:, None] * fs_t[None] ** power)    # (nd, nf) amplitude
        wf = wf / wf.sum(1, keepdim=True)

        # straight rays are valid for structures larger than the first Fresnel zone; the travel
        # time of a finite-frequency wave averages the medium over it, so thin sheets along the
        # beam do not act as wave guides (FRESNEL_SIGMA_MM)
        sig = FRESNEL_SIGMA_MM
        kz = torch.exp(-0.5 * (torch.arange(-int(3 * sig / dz), int(3 * sig / dz) + 1, device=dev) * dz / sig) ** 2)
        kx = torch.exp(-0.5 * (torch.arange(-int(3 * sig / dx), int(3 * sig / dx) + 1, device=dev) * dx / sig) ** 2)
        sm = F.conv2d(F.pad(slow[None, None], (len(kx) // 2, len(kx) // 2, len(kz) // 2, len(kz) // 2), mode="replicate"),
                      (kz[:, None] * kx[None, :] / (kz.sum() * kx.sum()))[None, None])
        slow_img = sm                                                             # (1,1,nz,nx)
        x0, x1 = float(x[0]), float(x[-1])
        w_el = ELEMENT_FILL * pitch
        K = torch.zeros(nd, nb, len(xi), dtype=torch.complex64, device=dev)
        K0 = torch.zeros(nd, len(xi), dtype=torch.complex64, device=dev)
        shift = torch.zeros(nd, nb, device=dev)
        trans = torch.ones(nd, nb, device=dev)
        if loss is not None:   # one-way cumulative intensity loss along vertical lines
            cum_img = torch.cumsum(loss / 2, 0)[None, None]
        ab_rms = torch.zeros(nd, nb, device=dev)
        s = (torch.arange(n_path, device=dev, dtype=torch.float32) + 0.5) / n_path
        centre = nb // 2
        for d in range(nd):
            z = float(self.band_z[d])
            # excess delay along each element -> (x_b, z) straight path
            px = xe_loc[..., None] + (self.block_x[:, None, None] - xe_loc[..., None]) * s           # (nb, nl, ns)
            pz = z * s.expand_as(px)
            g = torch.stack([(px - x0) / (x1 - x0) * 2 - 1, (pz / z_max) * 2 - 1], -1)
            sl = F.grid_sample(slow_img, g.reshape(1, nb, -1, 2), mode="bilinear", padding_mode="border",
                               align_corners=True).reshape(nb, -1, n_path)
            L = torch.sqrt(u ** 2 + z ** 2)                                                           # (nb, nl)
            tau = sl.mean(-1) * L                                                                     # us
            w_rx = _hann(u, half_rx[d])
            w_tx = _hann(u, half_tx[d])
            zf = float(z_tx[d])
            mean_rx = (w_rx * tau).sum(1) / w_rx.sum(1).clamp(min=1e-9)
            mean_tx = (w_tx * tau).sum(1) / w_tx.sum(1).clamp(min=1e-9)
            shift[d] = C0_MM_US * (mean_tx + mean_rx) / 2
            ab_rms[d] = torch.sqrt((w_rx * (tau - mean_rx[:, None]) ** 2).sum(1) / w_rx.sum(1).clamp(min=1e-9))
            if loss is not None:
                # loss along each path: increments of the cumulative field between consecutive path depths
                sk = torch.arange(n_path + 1, device=dev, dtype=torch.float32) / n_path
                qx = xe_loc[..., None] + (self.block_x[:, None, None] - xe_loc[..., None]) * sk[1:]
                gx = (qx - x0) / (x1 - x0) * 2 - 1
                g_hi = torch.stack([gx, (z * sk[1:] / z_max * 2 - 1).expand_as(gx)], -1)
                g_lo = torch.stack([gx, (z * sk[:-1] / z_max * 2 - 1).expand_as(gx)], -1)
                smp = lambda gg: F.grid_sample(cum_img, gg.reshape(1, nb, -1, 2), mode="bilinear", padding_mode="border",
                                               align_corners=True).reshape(nb, -1, n_path)
                L = (smp(g_hi) - smp(g_lo)).clamp(min=0).sum(-1)                                         # (nb, nl)
                amp = torch.exp(-L / 2)
                t_tx = (w_tx * amp).sum(1) / w_tx.sum(1).clamp(min=1e-9)
                t_rx = (w_rx * amp).sum(1) / w_rx.sum(1).clamp(min=1e-9)
                trans[d] = (t_tx * t_rx) ** 2
            # geometry: distances element -> (x_b + xi, z), focusing laws
            dxp = xi[None, None] - u[..., None]                                                       # (nb, nl, nk)
            r = torch.sqrt(dxp ** 2 + z ** 2)
            r_rx = torch.sqrt(u ** 2 + z ** 2)[..., None]                                             # dynamic receive focus
            r_tx = torch.sqrt(u ** 2 + zf ** 2)[..., None]                                            # transmit focus
            sin_t = dxp / r
            h = torch.zeros(nb, len(xi), dtype=torch.complex64, device=dev)
            h0 = torch.zeros(len(xi), dtype=torch.complex64, device=dev)
            for k, f in enumerate(fs):
                lam = C0_MM_US / f
                kw = 2 * np.pi / lam
                om = 2 * np.pi * f
                direc = torch.sinc(w_el * sin_t / lam)
                ph_rx = kw * (r - r_rx) + om * (tau - mean_rx[:, None])[..., None]
                # relative to the on-axis arrival the beamformer assumes (z / c0 after the focusing
                # law's centre delay): the off-focus wavefront curvature, not its time of flight
                ph_tx = kw * (r - r_tx - (z - zf)) + om * (tau - mean_tx[:, None])[..., None]
                hr = (w_rx[..., None] * direc * torch.polar(torch.ones_like(ph_rx), -ph_rx)).sum(1)
                ht = (w_tx[..., None] * direc * torch.polar(torch.ones_like(ph_tx), -ph_tx)).sum(1)
                h += wf[d, k] * hr * ht
                # ideal reference: same apertures, no aberration, transmit focused at this depth (centre
                # block); the ratio to it keeps the defocus loss off the transmit focus, the aperture
                # truncation at the array ends and the aberration
                ph0 = kw * (r[centre] - r_rx[centre])
                h0r = (w_rx[centre, :, None] * direc[centre] * torch.polar(torch.ones_like(ph0), -ph0)).sum(0)
                h0t = (w_tx[centre, :, None] * direc[centre] * torch.polar(torch.ones_like(ph0), -ph0)).sum(0)
                h0 += wf[d, k] * h0r * h0t
            # normalise by the ideal kernel's DC gain and remove each kernel's piston phase: the
            # bulk delay is applied separately (depth_shift), and kernels of different pistons
            # would cancel when interpolated between bands / blocks
            s0 = h0.sum().abs()
            K[d] = h * torch.polar(torch.ones(nb, device=dev), -torch.angle(h.sum(1)))[:, None] / s0
            K0[d] = h0 * torch.polar(torch.ones((), device=dev), -torch.angle(h0.sum())) / s0
        self.K, self.K0 = K, K0
        self.shift = shift                       # (nd, nb) mm, apparent minus true depth
        self.trans = trans                       # (nd, nb) two-way boundary transmission (intensity)
        self.ab_rms_ns = ab_rms * 1e3            # (nd, nb) receive-aperture rms delay residual, ns
        self.k2sum0 = (K0.abs() ** 2).sum(-1)    # (nd,) energy of the ideal (sum-normalised) kernel

    # ------------------------------------------------------------------ helpers
    def _band_weights(self, z):
        pos = torch.clamp((z - self.band_z[0]) / (self.band_z[1] - self.band_z[0]), 0, len(self.band_z) - 1 - 1e-4)
        d0 = pos.long()
        return d0, (pos - d0)

    def per_row(self, values, z):
        """Interpolate a per-band quantity (nd, ...) to rows at depths z."""
        d0, f = self._band_weights(z)
        f = f.reshape(-1, *([1] * (values.dim() - 1)))
        return values[d0] * (1 - f) + values[d0 + 1] * f

    def lateral(self, rf, z):
        """Spatially varying lateral convolution of complex rf (nz, nx) with the block / band kernels.

        The kernel follows the source position (input-weighted partition of unity over blocks)
        and the row depth (linear between bands).
        """
        nz, nx = rf.shape
        nb, nk = self.K.shape[1], self.K.shape[2]
        nfft = int(2 ** np.ceil(np.log2(nx + nk)))
        hk = self.half_k
        Kp = torch.zeros(*self.K.shape[:2], nfft, dtype=torch.complex64, device=self.dev)
        Kp[..., :hk + 1] = self.K[..., hk:]
        Kp[..., nfft - hk:] = self.K[..., :hk]
        Kf = torch.fft.fft(Kp, dim=-1)                                           # (nd, nb, nfft)
        # hat-function weights of each column for each block
        bx = self.block_x
        pos = torch.clamp((self.x - bx[0]) / (bx[1] - bx[0]), 0, nb - 1 - 1e-4)
        b0 = pos.long()
        fb = pos - b0
        wx = torch.zeros(nb, nx, device=self.dev)
        cols = torch.arange(nx, device=self.dev)
        wx[b0, cols] = 1 - fb
        wx[b0 + 1, cols] += fb
        d0, fd = self._band_weights(z)
        out = torch.zeros(nz, nfft, dtype=torch.complex64, device=self.dev)
        for b in range(nb):
            if not bool((wx[b] > 0).any()):
                continue
            Rb = torch.fft.fft(rf * wx[b][None], n=nfft, dim=-1)
            Kb = Kf[d0, b] * (1 - fd)[:, None] + Kf[d0 + 1, b] * fd[:, None]
            out += Rb * Kb
        return torch.fft.ifft(out, dim=-1)[:, :nx]

    def _blocks_to_columns(self, v_rows):
        bx = self.block_x
        pos = torch.clamp((self.x - bx[0]) / (bx[1] - bx[0]), 0, len(bx) - 1 - 1e-4)
        b0 = pos.long()
        fb = pos - b0
        return v_rows[:, b0] * (1 - fb) + v_rows[:, b0 + 1] * fb

    def transmission_field(self, z):
        """Two-way boundary transmission (nz, nx) at true depths z (1 at the probe face)."""
        t = self._blocks_to_columns(self.per_row(self.trans, z))
        f = torch.clamp(z / float(self.band_z[0]), max=1.0)[:, None]
        return 1 + (t - 1) * f

    def depth_shift(self, z):
        """Apparent minus true depth (nz, nx) at true depths z: bilinear in band and block."""
        s_rows = self.per_row(self.shift, z)                                     # (nz, nb)
        bx = self.block_x
        pos = torch.clamp((self.x - bx[0]) / (bx[1] - bx[0]), 0, len(bx) - 1 - 1e-4)
        b0 = pos.long()
        fb = pos - b0
        s = s_rows[:, b0] * (1 - fb) + s_rows[:, b0 + 1] * fb
        return s * torch.clamp(z / float(self.band_z[0]), max=1.0)[:, None]      # 0 at the probe face


def reverberation(rf_att, r_c, att, face_r, dz):
    """Second-order multiple reflections along each line (columns independent).

    rf_att: (nz, nx) complex echo field before the PSF, attenuated (primary echoes);
    r_c: (nz, nx) complex reflection amplitude of the interfaces (with incidence-angle factor
    and depth phase); att: (nz, nx) two-way intensity transmission to each depth;
    face_r: amplitude reflection coefficient of the probe face.
    Paths (extra round trip D, the echo appears D deeper):
      face:     transmit -> interface at D -> face -> target  : face_r * r(D) * sqrt(att(D))
      internal: target echo up -> interface d1+D -> down from interface d1 (reflection -r) -> ...
                g(D) = -sum_d1 r(d1 + D) conj(r(d1)) sqrt(att(d1 + D) / att(d1))
    Returns the reverberant field (nz, nx) on the same depth grid.
    """
    nz, nx = rf_att.shape
    n = int(2 ** np.ceil(np.log2(2 * nz)))
    sa = att.clamp(min=1e-8).sqrt()
    a = r_c * sa                                                                  # r(d) sqrt(att(d))
    b = r_c / sa
    Fa, Fb = torch.fft.fft(a, n=n, dim=0), torch.fft.fft(b, n=n, dim=0)
    g_int = -torch.fft.ifft(Fa * Fb.conj(), dim=0)[:nz]                           # lag D >= 0
    g = face_r * a + g_int
    g[0] = 0                                                                      # no zero-lag (primary) term
    out = torch.fft.ifft(torch.fft.fft(rf_att, n=n, dim=0) * torch.fft.fft(g, n=n, dim=0), dim=0)[:nz]
    return out
