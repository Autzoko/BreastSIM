"""Physics checks of the preview imaging model on simple layered phantoms."""
import numpy as np
import pytest
import torch

from breastsim import phantom, tissue
from breastsim.probe import PROBES, ProbePose
from breastsim.realtime.renderer import ImagingParams, PreviewRenderer, spectral_attenuation_lut

VOX = 0.2


def layered(top_label, bottom_label, boundary_mm, c_top=1540.0, c_bottom=1540.0, z_top=1.40, z_bottom=1.60,
            alpha=0.0, size=(40, 60, 60)):
    """Two flat layers; skin surface at a = size[0], boundary `boundary_mm` below it."""
    shape = tuple(int(s / VOX) for s in size)
    a = np.arange(shape[0]) * VOX
    deep = (size[0] - a) > boundary_mm
    label = np.where(deep, bottom_label, top_label)[:, None, None].repeat(shape[1], 1).repeat(shape[2], 2).astype(np.uint8)
    z = np.where(label == top_label, z_top, z_bottom).astype(np.float16)
    c = np.where(label == top_label, c_top, c_bottom).astype(np.float16)
    al = np.full(shape, alpha, np.float16)
    surface = np.full(shape[1:], size[0], np.float32)
    return phantom.Volume("test/layers", label, z, al, VOX, 1.0, surface, np.zeros(shape[1:], bool), {}, c)


def render(vol, preset="l11_5v", echo=True, **kw):
    r = PreviewRenderer("cpu")
    r.set_volume(vol)
    if not echo:
        r.echo_lut.zero_()                       # specular term only
    pose_kw = kw.pop("pose", {})
    p = ImagingParams.from_preset(preset, overlay="none", speckle_reduction=0, conform=False,
                                  tgc_db_cm_mhz=0, shadow_blur_mm=0, **kw)
    pose = ProbePose(u=vol.label.shape[1] * VOX / 2, v=vol.label.shape[2] * VOX / 2, **pose_kw)
    img, meta, labels = r.render(pose, p)
    return r, img, meta, labels


def boundary_level(r, meta, ref_rows):
    """Peak boundary echo amplitude in units of the unit-speckle rms."""
    inten = r.last_intensity.cpu().numpy()
    prof = np.sqrt(inten[:, inten.shape[1] // 4: 3 * inten.shape[1] // 4].mean(1))
    return prof.max(), prof


def specular_to_speckle(preset, px):
    """Boundary peak / glandular speckle rms, rendered at a fine output grid."""
    vol = layered(tissue.FAT, tissue.GLAND, 15.0)
    r, _, _, _ = render(vol, preset, echo=False, specular=30.0, px_x_mm=px, px_z_mm=px)
    peak, _ = boundary_level(r, None, None)
    r2, _, _, _ = render(layered(tissue.GLAND, tissue.GLAND, 15.0, z_top=1.6, z_bottom=1.6), preset,
                         px_x_mm=px, px_z_mm=px)
    return peak / np.sqrt(r2.last_intensity.cpu().numpy()[int(5 / px):int(25 / px)].mean())


def test_specular_amplitude_is_probe_independent():
    # same boundary (R = 0.0667) imaged by a 7.6 MHz and a 10 MHz probe
    lo, hi = specular_to_speckle("l11_5v", 0.02), specular_to_speckle("busi", 0.02)
    R = (1.60 - 1.40) / (1.60 + 1.40)
    assert hi / lo == pytest.approx(1.0, abs=0.15)
    for v in (lo, hi):                       # nominal: specular * R speckle-rms units
        assert v == pytest.approx(30.0 * R, rel=0.25)


def test_specular_falls_off_with_incidence_angle():
    vol = layered(tissue.FAT, tissue.GLAND, 15.0)
    r0, _, _, _ = render(vol, echo=False)
    p0, _ = boundary_level(r0, None, None)
    r1, _, _, _ = render(vol, echo=False, pose={"rock": 20.0})
    p1, _ = boundary_level(r1, None, None)
    assert 20 * np.log10(p1 / p0) < -8          # oblique smooth boundary reflects away from the aperture


def test_spectral_attenuation_lut_matches_gaussian_theory():
    spec = PROBES["L12-hf"]
    A, E, fbar, sf = spectral_attenuation_lut(spec, power=1.0)
    fc = spec.fc / 1e6
    s_amp = spec.bandwidth * fc / 2.355
    assert E[0] == pytest.approx(1.0) and fbar[0] == pytest.approx(fc, rel=1e-3)
    # |S|^2 exp(-4 A f) with Gaussian |S|^2 of variance s_amp^2/2: centroid fc - 2 A s_amp^2, exact for y = 1
    i = np.searchsorted(A, 0.1)
    assert fbar[i] == pytest.approx(fc - 2 * A[i] * s_amp ** 2, rel=0.01)
    # energy loss is smaller than the centre-frequency estimate exp(-4 A fc)
    assert E[i] > np.exp(-4 * A[i] * fc)
    assert np.all(np.diff(fbar) < 0) and np.all(np.diff(E) < 0)


def test_speed_of_sound_moves_boundaries_like_a_scanner():
    # 10 mm of slow fat (1440 m/s) over gland: the beamformer (1540 m/s) shows the boundary at 10 * 1540/1440 mm
    vol = layered(tissue.FAT, tissue.GLAND, 10.0, c_top=1440.0, c_bottom=1540.0)
    _, _, meta, labels = render(vol)
    col = labels[:, labels.shape[1] // 2]
    row = np.argmax(col == tissue.GLAND)
    assert row * meta["px_z_mm"] == pytest.approx(10.0 * 1540 / 1440, abs=0.15)


def test_output_speckle_is_rayleigh():
    vol = layered(tissue.GLAND, tissue.GLAND, 15.0, z_top=1.6, z_bottom=1.6)
    r, _, meta, _ = render(vol)
    env = np.sqrt(r.last_intensity.cpu().numpy())
    w = env[int(10 / meta["px_z_mm"]):int(30 / meta["px_z_mm"]), 60:-60]
    assert w.mean() / w.std() == pytest.approx(1.91, rel=0.1)


def test_attenuation_slope_matches_spectral_theory():
    alpha = 0.5 * 100 / 8.686                    # 0.5 dB/cm/MHz, y = 1, in Np/m/MHz
    vol = layered(tissue.GLAND, tissue.GLAND, 15.0, z_top=1.6, z_bottom=1.6, alpha=alpha)
    r, _, meta, _ = render(vol)
    inten = r.last_intensity.cpu().numpy()
    zz = (np.arange(inten.shape[0]) + 0.5) * meta["px_z_mm"]
    sel = (zz > 5) & (zz < 35)
    slope = np.polyfit(zz[sel], 10 * np.log10(inten.mean(1))[sel], 1)[0]
    A, E, _, _ = spectral_attenuation_lut(PROBES["L11-5v"], 1.0)
    expect = np.polyfit(zz[sel], 10 * np.log10(np.interp(alpha * zz[sel] * 1e-3, A, E)), 1)[0]
    assert slope == pytest.approx(expect, rel=0.05)


def sphere_in_fat(c_lesion=1560.0, alpha_lesion=None, radius=4.0, size=(40, 60, 60)):
    """Round lesion 15 mm deep in fat (c 1440); lesion attenuation equals fat unless given."""
    shape = tuple(int(s / VOX) for s in size)
    a, u, v = np.meshgrid(*[np.arange(n) * VOX for n in shape], indexing="ij")
    les = (a - (size[0] - 15)) ** 2 + (u - size[1] / 2) ** 2 + (v - size[2] / 2) ** 2 < radius ** 2
    label = np.where(les, tissue.TUMOR, tissue.FAT).astype(np.uint8)
    z = np.where(les, 1.47, 1.35).astype(np.float16)
    c = np.where(les, c_lesion, 1440.0).astype(np.float16)
    fat_alpha = 0.4 * tissue.NP_PER_M_PER_DB_PER_CM
    al = np.where(les, fat_alpha if alpha_lesion is None else alpha_lesion, fat_alpha).astype(np.float16)
    surface = np.full(shape[1:], size[0], np.float32)
    return phantom.Volume("test/sphere", label, z, al, VOX, 1.0, surface, les.any(0), {}, c)


def posterior_profile(vol, **kw):
    r, _, meta, labels = render(vol, **kw)
    inten = r.last_intensity.cpu().numpy()
    rows, cols = np.nonzero(labels == tissue.TUMOR)
    below = slice(rows.max() + int(2 / meta["px_z_mm"]), rows.max() + int(10 / meta["px_z_mm"]))
    return 10 * np.log10(inten[below].mean(0)), cols, meta


def test_refraction_edge_shadows_at_lesion_margins():
    vol = sphere_in_fat()
    prof, cols, meta = posterior_profile(vol)
    prof_off, _, _ = posterior_profile(vol, edge_shadow_db=0.0)
    px = meta["px_x_mm"]
    edges = np.r_[cols.min() - int(0.5 / px):cols.min() + int(0.5 / px), cols.max() - int(0.5 / px):cols.max() + int(0.5 / px)]
    far = np.r_[: cols.min() - int(6 / px), cols.max() + int(6 / px):]
    # dark streaks behind the margins (same attenuation everywhere: pure refraction effect) ...
    assert prof[edges].mean() < prof[far].mean() - 2.0
    # ... that vanish when the boundary-transmission model is off
    assert abs(prof_off[edges].mean() - prof_off[far].mean()) < 1.0


def test_shadowing_lesion_darkens_posterior_region():
    prof_none, cols, meta = posterior_profile(sphere_in_fat(c_lesion=1440.0), edge_shadow_db=0.0)
    prof_sh, _, _ = posterior_profile(sphere_in_fat(c_lesion=1440.0, alpha_lesion=3.0 * tissue.NP_PER_M_PER_DB_PER_CM),
                                      edge_shadow_db=0.0)
    mid = slice(int(np.median(cols)) - 5, int(np.median(cols)) + 5)
    assert prof_sh[mid].mean() < prof_none[mid].mean() - 10      # 3 dB/cm/MHz, 8 mm, 7.6 MHz: strong shadow


def test_posterior_features_follow_dataset_frequencies():
    lesions = [{"id": i, "type": tissue.NAMES[tissue.TUMOR]} for i in range(2000)]
    phantom.lesion_acoustics("x/y", lesions)
    again = phantom.lesion_acoustics("x/y", [{"id": i, "type": tissue.NAMES[tissue.TUMOR]} for i in range(2000)])
    assert [l["posterior"] for l in lesions] == [l["posterior"] for l in again]     # deterministic
    freq = {k: np.mean([l["posterior"] == k for l in lesions]) for k in tissue.POSTERIOR_FEATURE_PROBS[tissue.TUMOR]}
    for k, pr in tissue.POSTERIOR_FEATURE_PROBS[tissue.TUMOR].items():
        assert freq[k] == pytest.approx(pr, abs=0.04)
    for l in lesions:
        lo, hi = tissue.POSTERIOR_ATTENUATION_DB_CM_MHZ[l["posterior"]]
        assert lo <= l["attenuation_db_cm_mhz"] <= hi


def test_lesion_alpha_override():
    from breastsim.realtime.renderer import lesion_alpha
    vol = sphere_in_fat()
    ids = (vol.label == tissue.TUMOR).astype(np.uint8)
    vol.lesion_id = ids
    vol.meta["lesions"] = [{"id": 0, "attenuation_db_cm_mhz": 2.0}]
    a = lesion_alpha(vol)
    assert np.allclose(a[ids == 1], 2.0 * tissue.NP_PER_M_PER_DB_PER_CM, rtol=1e-3)
    assert np.allclose(a[ids == 0], vol.alpha[ids == 0].astype(np.float32))


def test_thin_ligament_parallel_to_beam_casts_only_a_faint_streak():
    # a 0.4 mm ligament sheet along the beam (grazing incidence on both faces) in fat
    shape = tuple(int(s / VOX) for s in (40, 60, 60))
    label = np.full(shape, tissue.FAT, np.uint8)
    mid = shape[1] // 2
    label[: shape[0] - int(5 / VOX), mid - 1:mid + 1, :] = tissue.LIGAMENT
    z = np.where(label == tissue.LIGAMENT, 1.68, 1.35).astype(np.float16)
    c = np.where(label == tissue.LIGAMENT, 1480.0, 1440.0).astype(np.float16)
    al = np.full(shape, 0.4 * tissue.NP_PER_M_PER_DB_PER_CM, np.float16)
    vol = phantom.Volume("test/ligament", label, z, al, VOX, 1.0, np.full(shape[1:], 40.0, np.float32),
                         np.zeros(shape[1:], bool), {}, c)
    r, _, meta, _ = render(vol)
    inten = r.last_intensity.cpu().numpy()
    rows = slice(int(15 / meta["px_z_mm"]), int(30 / meta["px_z_mm"]))
    prof = 10 * np.log10(inten[rows].mean(0))
    centre = prof.shape[0] // 2
    px = meta["px_x_mm"]
    streak = prof[centre - int(0.5 / px):centre + int(0.5 / px)].mean()
    beside = np.r_[prof[: centre - int(5 / px)], prof[centre + int(5 / px):]].mean()
    assert beside - streak < 3.5
