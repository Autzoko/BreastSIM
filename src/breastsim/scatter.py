"""Deterministic sub-resolution scatterer field.

Space is divided into cubic cells of SCATTER_CELL_MM. Every cell holds one
point scatterer whose position (jittered inside the cell) and Gaussian
amplitude come from an integer hash of the cell index and a per-phantom seed.
The field is a function of world position only, so the preview, the PyMUST
generator and repeated visits to the same spot all see the same speckle
source. Amplitudes are later scaled by tissue echogenicity.
"""
import torch

SCATTER_CELL_MM = 0.1   # ~10 scatterers per resolution cell at 10-12 MHz
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


_KEY_BITS, _KEY_OFF = 20, 1 << 19     # +-2^19 cells per axis (+-52 m at 0.1 mm)


def cell_index(points, cell=SCATTER_CELL_MM):
    """Integer cell index (..., 3) of each point (mm)."""
    return torch.floor(points.float() / cell).to(torch.int64)


def unique_cells(points, cell=SCATTER_CELL_MM):
    """Distinct cells hit by a cloud of points: (N, 3) int64 indices."""
    idx = cell_index(points.reshape(-1, 3), cell) + _KEY_OFF
    key = (idx[:, 0] << (2 * _KEY_BITS)) | (idx[:, 1] << _KEY_BITS) | idx[:, 2]
    key = torch.unique(key)
    mask = (1 << _KEY_BITS) - 1
    return torch.stack([key >> (2 * _KEY_BITS), (key >> _KEY_BITS) & mask, key & mask], -1) - _KEY_OFF


def from_index(idx, cell=SCATTER_CELL_MM, seed=0):
    """Scatterer (amplitude (...,), position (..., 3) mm) of each cell index."""
    ix, iy, iz = idx.unbind(-1)
    u1 = _uniform(_hash(ix, iy, iz, seed, 1))
    u2 = _uniform(_hash(ix, iy, iz, seed, 2))
    amp = torch.sqrt(-2 * torch.log(u1)) * torch.cos(2 * torch.pi * u2)
    # keep the scatterer strictly inside its cell despite float32 rounding
    jitter = 0.001 + 0.998 * torch.stack([_uniform(_hash(ix, iy, iz, seed, 3 + k)) for k in range(3)], -1)
    return amp, (idx.to(torch.float32) + jitter) * cell


def cells(points, cell=SCATTER_CELL_MM, seed=0):
    """Scatterer of the cell containing each point.

    points: (..., 3) float tensor, mm. Returns (amplitude (...,), position (..., 3) mm).
    """
    amp, pos = from_index(cell_index(points, cell), cell, seed)
    return amp.to(points.dtype), pos.to(points.dtype)
