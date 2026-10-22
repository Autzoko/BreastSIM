import numpy as np
import pytest

from breastsim import tissue
from breastsim.presets import PRESETS
from breastsim.probe import ProbePose
from breastsim.realtime.renderer import ImagingParams, PreviewRenderer


@pytest.fixture(scope="module")
def renderer(synthetic_volume):
    r = PreviewRenderer("cpu")
    r.set_volume(synthetic_volume)
    return r


def lesion_pose(vol, **kw):
    a, u, v = vol.meta["lesions"][0]["center_mm"]
    return ProbePose(u=u, v=v, **kw)


def test_output_grid_matches_preset(renderer, synthetic_volume):
    for name in PRESETS:
        p = ImagingParams.from_preset(name, elevation_samples=4)
        img, meta, mask = renderer.render(lesion_pose(synthetic_volume), p)
        assert img.dtype == np.uint8 and img.shape[2] == 3 and img.shape[:2] == mask.shape
        assert meta["px_x_mm"] == pytest.approx(p.px_x_mm, rel=0.02)
        assert meta["px_z_mm"] == pytest.approx(p.px_z_mm, rel=0.02)
        assert img.shape[0] * meta["px_z_mm"] == pytest.approx(p.depth_mm, abs=0.1)


def test_lesion_mask_geometry(renderer, synthetic_volume):
    vol = synthetic_volume
    p = ImagingParams.from_preset("busi", elevation_samples=4)
    img, meta, mask = renderer.render(lesion_pose(vol), p)
    # 6 mm sphere cut through its center: ~28 mm^2, centred laterally, ~15 mm below the skin
    assert meta["lesion_area_mm2"] == pytest.approx(np.pi * 9, rel=0.2)
    rows, cols = np.nonzero(mask)
    assert cols.mean() * meta["px_x_mm"] == pytest.approx(meta["width_mm"] / 2, abs=0.5)
    assert rows.mean() * meta["px_z_mm"] == pytest.approx(15.0, abs=0.7)
    assert meta["lesion_types"] == [tissue.NAMES[tissue.TUMOR]]
    _, meta_far, mask_far = renderer.render(ProbePose(u=lesion_pose(vol).u, v=lesion_pose(vol).v + 12), p)
    assert not mask_far.any() and meta_far["lesion_area_mm2"] == 0


def test_render_is_deterministic(renderer, synthetic_volume):
    p = ImagingParams.from_preset("l11_5v", elevation_samples=6)
    a, _, _ = renderer.render(lesion_pose(synthetic_volume), p)
    b, _, _ = renderer.render(lesion_pose(synthetic_volume), p)
    assert np.array_equal(a, b)


def test_speckle_decorrelates_with_elevation_shift(renderer, synthetic_volume):
    p = ImagingParams.from_preset("l11_5v", elevation_samples=16, overlay="none", speckle_reduction=0)
    base = lesion_pose(synthetic_volume)
    ref, _, _ = renderer.render(base, p)
    win = (slice(20, 120), slice(40, 340))       # fat / gland region, above the lesion

    def corr(dv):
        img, _, _ = renderer.render(ProbePose(u=base.u, v=base.v + dv), p)
        x, y = ref[win][..., 0].astype(float).ravel(), img[win][..., 0].astype(float).ravel()
        return np.corrcoef(x, y)[0, 1]
    c_small, c_large = corr(0.1), corr(4.0)
    assert c_small > 0.6 and c_large < 0.4 and c_small > c_large


def test_tumor_is_hypoechoic_with_posterior_shadow(renderer, synthetic_volume):
    p = ImagingParams.from_preset("busi", elevation_samples=8, overlay="none")
    img, meta, mask = renderer.render(lesion_pose(synthetic_volume), p)
    g = img[..., 0].astype(float)
    rows, cols = np.nonzero(mask)
    c0, c1 = cols.min(), cols.max()
    inside = g[mask].mean()
    beside = g[rows.min():rows.max(), max(0, c0 - 60):c0 - 20].mean()
    below = g[rows.max() + 20:rows.max() + 80, c0 + 5:c1 - 5].mean()
    below_side = g[rows.max() + 20:rows.max() + 80, max(0, c0 - 60):c0 - 20].mean()
    assert inside < beside - 10          # darker than the surrounding fat
    assert below < below_side - 5        # posterior shadowing


def test_conform_removes_gel_gap(renderer, synthetic_volume):
    vol = synthetic_volume
    pose = ProbePose(u=lesion_pose(vol).u + 12, v=lesion_pose(vol).v)   # off the dome top: curved skin below
    p_on = ImagingParams.from_preset("busi", elevation_samples=4, overlay="labels", conform=True)
    p_off = ImagingParams.from_preset("busi", elevation_samples=4, overlay="labels", conform=False)
    _, m_on, _ = renderer.render(pose, p_on)
    _, m_off, _ = renderer.render(pose, p_off)
    assert m_on["skin_offset_mm"][1] > 1.0 and m_off["skin_offset_mm"] == [0.0, 0.0]
