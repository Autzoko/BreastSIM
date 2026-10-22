"""Synthetic phantom shared by the tests (no downloaded data needed)."""
import numpy as np
import pytest

from breastsim import phantom, tissue

VOXEL = 0.2


def make_synthetic(flip=False):
    """Dome-shaped breast on a 0.2 mm grid in raw (source) axis order.

    Source order is (u, v, a) with the chest wall at a = 0 (or at the far
    end when flip=True), so canonicalize() has to find and reorder the AP
    axis. Fat dome with a skin shell, a glandular layer, a ligament sheet
    and a 6 mm tumor 15 mm below the dome top.
    """
    U, V, A = 200, 200, 160                      # 40 x 40 x 32 mm
    u, v, a = np.meshgrid(np.arange(U), np.arange(V), np.arange(A), indexing="ij")
    uc, vc = (U - 1) / 2, (V - 1) / 2
    r = np.hypot((u - uc) * VOXEL, (v - vc) * VOXEL)
    top = 30.0 - 0.012 * r ** 2                  # dome height (mm) over the chest wall
    am = a * VOXEL
    label = np.zeros((U, V, A), np.uint8)
    inside = (am <= top) & (r < 45)
    label[inside] = tissue.FAT
    label[inside & (am > top - 1.0)] = tissue.SKIN
    label[inside & (am > 8) & (am < 14)] = tissue.GLAND
    label[inside & (np.abs(am - 18) < 0.25)] = tissue.LIGAMENT
    center = np.array([uc, vc, (30.0 - 15.0) / VOXEL])
    d = np.sqrt(((np.stack([u, v, a], -1) - center) ** 2).sum(-1)) * VOXEL
    label[d < 3.0] = tissue.TUMOR
    c, rho, al = (tissue.lut({k: v[i] for k, v in tissue.DEFAULT_ACOUSTIC.items()}, default=np.nan) for i in range(3))
    for lab, (cc, rr, aa) in {tissue.LIGAMENT: (1480, 1140, 14.5), tissue.TUMOR: (1548, 945, 31.0)}.items():
        c[lab], rho[lab], al[lab] = cc, rr, aa
    z = c[label] * rho[label] / 1e6
    alpha = al[label]
    if flip:
        label, z, alpha = label[..., ::-1], z[..., ::-1], alpha[..., ::-1]
    return label, z.astype(np.float32), alpha.astype(np.float32)


@pytest.fixture(scope="session")
def synthetic_volume():
    label, z, alpha = make_synthetic()
    label, z, alpha, ori = phantom.canonicalize(label, z, alpha)
    vol = phantom.Volume("synthetic/dome", label, z.astype(np.float16), alpha.astype(np.float16), VOXEL, 1.2,
                         phantom._surface(label, VOXEL), np.isin(label, tissue.LESIONS).any(axis=0),
                         {"description": "synthetic dome", "orientation": ori})
    vol.meta["lesions"] = phantom.find_lesions(label, VOXEL)
    return vol
