"""Deterministic sub-resolution scatterer field.

Space is divided into cubic cells of SCATTER_CELL_MM. Every cell holds one
point scatterer whose position (jittered inside the cell) and Gaussian
amplitude come from an integer hash of the cell index and a per-phantom seed.
The field is a function of world position only, so the preview, the PyMUST
generator and repeated visits to the same spot all see the same speckle
source. Amplitudes are later scaled by tissue echogenicity.
"""
import torch

SCATTER_CELL_MM = 0.08
_M32 = 0xFFFFFFFF


def _mix(h):
    h = ((h ^ (h >> 16)) * 0x45D9F3B) & _M32
    h = ((h ^ (h >> 16)) * 0x45D9F3B) & _M32
    return h ^ (h >> 16)


def _hash(ix, iy, iz, seed, salt):
    h = (ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791) ^ (seed * 2654435761 + salt)
    return _mix(h & _M32)


def _uniform(h):
    return (h.to(torch.float32) + 0.5) / 4294967296.0  # float32: MPS has no float64


def cells(points, cell=SCATTER_CELL_MM, seed=0):
    """Scatterer of the cell containing each point.

    points: (..., 3) float tensor, mm. Returns (amplitude (...,), position (..., 3) mm).
    """
    idx = torch.floor(points.float() / cell).to(torch.int64)
    ix, iy, iz = idx.unbind(-1)
    u1 = _uniform(_hash(ix, iy, iz, seed, 1))
    u2 = _uniform(_hash(ix, iy, iz, seed, 2))
    amp = torch.sqrt(-2 * torch.log(u1)) * torch.cos(2 * torch.pi * u2)
    jitter = torch.stack([_uniform(_hash(ix, iy, iz, seed, 3 + k)) for k in range(3)], -1)
    pos = (idx.to(torch.float32) + jitter) * cell
    return amp.to(points.dtype), pos.to(points.dtype)
