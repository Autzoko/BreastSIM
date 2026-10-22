"""Linear-array probe description and pose in the canonical phantom frame.

Frame (mm): a = anterior-posterior (chest wall at 0), u / v = coronal plane.
At zero angles the beam points from the nipple side toward the chest wall
(-a), the array runs along +u and the elevation direction is +v.
Both the preview renderer and the data generator build the image plane from
ProbePose.frame(), so a saved scene reproduces the same anatomy.
"""
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class ProbeSpec:
    name: str
    fc: float            # center frequency, Hz
    bandwidth: float     # fractional -6 dB bandwidth (0-1)
    n_elements: int
    pitch: float         # m
    height: float        # elevation aperture, m
    elev_focus: float    # elevation lens focus, m
    c: float = 1540.0

    @property
    def aperture(self):
        return self.n_elements * self.pitch

    @property
    def wavelength(self):
        return self.c / self.fc

    @classmethod
    def from_pymust(cls, name):
        import pymust
        p = pymust.getparam(name)
        return cls(name, float(p.fc), float(p.bandwidth) / 100, int(p.Nelements), float(p.pitch),
                   float(p.height), float(p.focus))

    def to_dict(self):
        return asdict(self)


# Presets mirroring PyMUST's getparam (avoids importing pymust in the server).
PROBES = {
    "L11-5v": ProbeSpec("L11-5v", 7.6e6, 0.77, 128, 0.3e-3, 5e-3, 18e-3),
    "L12-3v": ProbeSpec("L12-3v", 7.54e6, 0.93, 192, 0.2e-3, 5e-3, 20e-3),
}


@dataclass
class ProbePose:
    u: float               # face center, coronal coordinates (mm)
    v: float
    rotation: float = 0.0  # deg, in-plane rotation of the array about the beam axis
    tilt: float = 0.0      # deg, fan the beam toward elevation (about the array axis)
    rock: float = 0.0      # deg, rock the beam toward the array direction (about elevation)
    standoff: float = 0.0  # mm, extra gel between skin and probe face

    def to_dict(self):
        return asdict(self)

    def axes(self):
        """Unit vectors (lateral, axial, elevation) as rows, (a, u, v) components."""
        r, t, k = np.deg2rad([self.rotation, self.tilt, self.rock])
        lat = np.array([0.0, np.cos(r), np.sin(r)])
        elev = np.array([0.0, -np.sin(r), np.cos(r)])
        ax = np.array([-1.0, 0.0, 0.0])
        # rock: rotate axial toward lateral about elevation
        ax, lat = np.cos(k) * ax + np.sin(k) * lat, np.cos(k) * lat - np.sin(k) * ax
        # tilt: rotate axial toward elevation about lateral
        ax, elev = np.cos(t) * ax + np.sin(t) * elev, np.cos(t) * elev - np.sin(t) * ax
        return np.stack([lat, ax, elev])

    def face_height(self, surface, voxel_mm, spec):
        """A coordinate of the probe face: touches the highest skin point under it."""
        lat, _, elev = self.axes()
        half_w, half_h = spec.aperture * 1e3 / 2, spec.height * 1e3 / 2
        s = np.linspace(-1, 1, 33)
        uu = self.u + np.add.outer(s * half_w * lat[1], s * half_h * elev[1])
        vv = self.v + np.add.outer(s * half_w * lat[2], s * half_h * elev[2])
        iu = np.clip(np.round(uu / voxel_mm).astype(int), 0, surface.shape[0] - 1)
        iv = np.clip(np.round(vv / voxel_mm).astype(int), 0, surface.shape[1] - 1)
        h = surface[iu, iv]
        return (float(np.nanmax(h)) if np.isfinite(h).any() else 0.0) + self.standoff

    def frame(self, surface, voxel_mm, spec):
        """(origin, axes): origin is the face center (a, u, v) in mm."""
        a0 = self.face_height(surface, voxel_mm, spec)
        return np.array([a0, self.u, self.v]), self.axes()
