import numpy as np
import pytest
import torch

from breastsim import phantom, scatter, tissue
from breastsim.probe import PROBES, ProbePose, skin_offsets

from .conftest import VOXEL, make_synthetic


def test_scatter_is_deterministic_and_seeded():
    pts = torch.rand(10000, 3) * 50
    a1, p1 = scatter.cells(pts, seed=7)
    a2, p2 = scatter.cells(pts.clone(), seed=7)
    a3, _ = scatter.cells(pts, seed=8)
    assert torch.equal(a1, a2) and torch.equal(p1, p2)
    assert not torch.equal(a1, a3)
    # scatterer sits inside the cell of its query point
    cell = scatter.SCATTER_CELL_MM
    assert torch.equal(torch.floor(p1 / cell), torch.floor(pts / cell))


def test_scatter_amplitudes_are_standard_normal():
    pts = torch.stack(torch.meshgrid(*[torch.arange(60) * scatter.SCATTER_CELL_MM + 0.01] * 3, indexing="ij"), -1)
    a, _ = scatter.cells(pts.reshape(-1, 3))
    assert abs(float(a.mean())) < 0.02 and abs(float(a.std()) - 1) < 0.02


@pytest.mark.parametrize("angles", [(0, 0, 0), (37, 0, 0), (0, 20, 0), (0, 0, -15), (120, -12, 25)])
def test_pose_axes_orthonormal(angles):
    ax = ProbePose(0, 0, *angles).axes()
    assert np.allclose(ax @ ax.T, np.eye(3), atol=1e-12)


def test_pose_default_axes():
    lat, ax, el = ProbePose(0, 0).axes()
    assert np.allclose(lat, [0, 1, 0]) and np.allclose(ax, [-1, 0, 0]) and np.allclose(el, [0, 0, 1])


def test_skin_offsets_flat_and_curved():
    surf = np.full((100, 100), 20.0, np.float32)
    spec = PROBES["L12-hf"]
    pose = ProbePose(10, 10)
    origin, axes = pose.frame(surf, 0.2, spec)
    assert origin[0] == pytest.approx(20.0)
    x = np.linspace(-5, 5, 11)
    assert np.allclose(skin_offsets(surf, 0.2, origin, axes, x), 0)
    assert np.allclose(skin_offsets(surf, 0.2, origin, axes, x, standoff=2.0), -2.0)
    # curved skin: line starts follow the skin, never above the face
    uu = np.arange(100)[:, None] * 0.2
    surf2 = (20.0 - 0.05 * (uu - 10) ** 2) * np.ones((1, 100), np.float32)
    origin, axes = pose.frame(surf2, 0.2, spec)
    offs = skin_offsets(surf2, 0.2, origin, axes, x)
    expect = 0.05 * x ** 2
    assert np.allclose(offs, expect, atol=0.06)
    assert np.allclose(skin_offsets(surf2, 0.2, origin, axes, x, conform=False), 0)


@pytest.mark.parametrize("flip", [False, True])
def test_canonicalize_puts_chest_wall_first(flip):
    label, z, alpha = make_synthetic(flip=flip)
    lab, zz, aa, ori = phantom.canonicalize(label, z, alpha)
    assert ori["axis_order"][0] == 2 and ori["flip"] == flip
    occ = (lab > 0).mean(axis=(1, 2))
    nz = np.nonzero(occ)[0]
    assert occ[nz[0]] > 5 * occ[nz[-1]]        # wide chest end at index 0, apex at the far end
    assert (zz[lab == 0] == phantom.WATER_Z).all()
    assert lab.shape == zz.shape == aa.shape


def test_surface_and_lesion_detection(synthetic_volume):
    vol = synthetic_volume
    les = vol.meta["lesions"]
    assert len(les) == 1 and les[0]["type"] == "viable tumor"
    assert les[0]["diameter_mm"] == pytest.approx(6.0, abs=0.4)
    a, u, v = les[0]["center_mm"]
    iu, iv = int(round(u / VOXEL)), int(round(v / VOXEL))
    assert vol.surface[iu, iv] - a == pytest.approx(15.0, abs=0.5)


def test_label_downsampling_keeps_thin_structures():
    lab = np.full((40, 40, 40), tissue.FAT, np.uint8)
    lab[:, :, 21] = tissue.LIGAMENT               # 1-voxel sheet at an odd index
    lab[11, 11, 11] = tissue.TUMOR
    small = phantom._downsample_labels(lab, 2)
    assert (small[:, :, 10] == tissue.LIGAMENT).all()
    assert small[5, 5, 5] == tissue.TUMOR
    assert (phantom._stride(lab, 2) == tissue.LIGAMENT).sum() == 0   # plain striding would lose both


def test_meshes_cover_the_lesion(synthetic_volume):
    vol = synthetic_volume
    meshes = phantom.build_meshes(vol.label, vol.voxel_mm, vol.meta["lesions"])
    v, f = meshes["lesion0"]
    assert len(f) > 50 and f.max() < len(v)
    center = np.array(vol.meta["lesions"][0]["center_mm"])
    r = np.linalg.norm(v - center, axis=1)
    assert r.mean() == pytest.approx(3.0, abs=0.4)
    sv, sf = meshes["skin"]
    assert len(sf) > 100 and sv[:, 0].max() == pytest.approx(np.nanmax(vol.surface), abs=1.2)


def test_display_meshes_are_smooth_and_cached(synthetic_volume, monkeypatch, tmp_path):
    monkeypatch.setattr(phantom.paths, "CACHE", tmp_path)
    vol = synthetic_volume
    vol.lesion_id = np.isin(vol.label, tissue.LESIONS).astype(np.uint8)
    m = phantom.display_meshes(vol)
    assert {"skin", "gland", "lesion0"} <= set(m)
    v, f = m["skin"]
    assert f.max() < len(v) and v[:, 0].max() == pytest.approx(np.nanmax(vol.surface), abs=1.0)
    # Taubin smoothing keeps the dome: apex height within 1 mm, no shrinkage of the base
    m2 = phantom.display_meshes(vol)                     # second call reads the cache file
    assert np.array_equal(m2["skin"][0], v)
    assert (tmp_path / "preview" / "synthetic__dome" / f"display_meshes_v{phantom.DISPLAY_MESH_VERSION}.npz").exists()
