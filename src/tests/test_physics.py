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


def test_spectral_attenuation_lut_matches_pymust_spectrum():
    # the pulse-echo spectrum is MUST's (one-cycle excitation x generalized normal probe response):
    # cross-check the LUT centroids against PyMUST's own spectrum functions
    pymust = pytest.importorskip("pymust")
    spec = PROBES["L11-5v"]
    A, E, fbar, sf = spectral_attenuation_lut(spec, power=1.0)
    param = pymust.getparam("L11-5v")
    f = np.linspace(1e-3, 2 * param.fc, 4000)
    S = np.abs(param.getPulseSpectrumFunction()(2 * np.pi * f) * param.getProbeFunction()(2 * np.pi * f) ** 2)
    for a in (0.0, 0.05, 0.1, 0.2):
        P = S ** 2 * np.exp(-4 * a * (f / 1e6))
        ref = (P * f).sum() / P.sum() / 1e6
        assert np.interp(a, A, fbar) == pytest.approx(ref, rel=0.01)
    assert E[0] == pytest.approx(1.0)
    assert np.all(np.diff(fbar) < 0) and np.all(np.diff(E) < 0)
    # the probe passband bounds the downshift: 30 mm of 0.75 dB/cm/MHz tissue keeps > 60 % of fc
    assert np.interp(0.75 * 3 / 8.686, A, fbar) > 0.6 * spec.fc / 1e6


def test_speed_of_sound_moves_boundaries_like_a_scanner():
    # 10 mm of slow fat (1440 m/s) over gland: the beamformer (1540 m/s) shows the boundary at 10 * 1540/1440 mm
    vol = layered(tissue.FAT, tissue.GLAND, 10.0, c_top=1440.0, c_bottom=1540.0)
    _, _, meta, labels = render(vol)
    col = labels[:, labels.shape[1] // 2]
    row = np.argmax(col == tissue.GLAND)
    assert row * meta["px_z_mm"] == pytest.approx(10.0 * 1540 / 1440, abs=0.15)


def test_output_speckle_is_rayleigh():
    # physics check on a grid fine enough that scan conversion needs no anti-alias filter
    # (coarser display grids low-pass the envelope, as scanners do, raising the SNR)
    vol = layered(tissue.GLAND, tissue.GLAND, 15.0, z_top=1.6, z_bottom=1.6)
    r, _, meta, _ = render(vol, px_x_mm=0.05, px_z_mm=0.05)
    env = np.sqrt(r.last_intensity.cpu().numpy())
    w = env[int(10 / meta["px_z_mm"]):int(30 / meta["px_z_mm"]), 60:-60]
    assert w.mean() / w.std() == pytest.approx(1.91, rel=0.1)


def test_attenuation_slope_matches_spectral_theory():
    alpha = 0.5 * 100 / 8.686                    # 0.5 dB/cm/MHz, y = 1, in Np/m/MHz
    vol = layered(tissue.GLAND, tissue.GLAND, 15.0, z_top=1.6, z_bottom=1.6, alpha=alpha)
    r, _, meta, _ = render(vol)
    inten = r.last_intensity.cpu().numpy()
    # the same medium without attenuation carries the beam's own depth dependence (transmit focal zone)
    r0, _, _, _ = render(layered(tissue.GLAND, tissue.GLAND, 15.0, z_top=1.6, z_bottom=1.6, alpha=0.0))
    zz = (np.arange(inten.shape[0]) + 0.5) * meta["px_z_mm"]
    sel = (zz > 5) & (zz < 35)
    ratio = inten.mean(1) / r0.last_intensity.cpu().numpy().mean(1)
    slope = np.polyfit(zz[sel], 10 * np.log10(ratio)[sel], 1)[0]
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
    prof_off, _, _ = posterior_profile(vol, edge_shadow_db=0.0, aberration=False, reverberation=False)
    px = meta["px_x_mm"]
    edges = np.r_[cols.min() - int(0.5 / px):cols.min() + int(0.5 / px), cols.max() - int(0.5 / px):cols.max() + int(0.5 / px)]
    # reference columns away from the sphere but also away from the array ends, where the truncated
    # aperture lowers the focal gain
    n = prof.shape[0]
    far = np.r_[int(5 / px): cols.min() - int(6 / px), cols.max() + int(6 / px): n - int(5 / px)]
    # dark streaks behind the margins (same attenuation everywhere: pure refraction effect) ...
    assert prof[edges].mean() < prof[far].mean() - 1.5
    # ... that vanish when the boundary-transmission model is off (with the lens-like delays of the
    # sphere, which also defocus behind its margins, and the multiple reflections off as well)
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


def test_chest_wall_muscle_ribs_and_rib_shadows():
    # thin flat breast (12 mm of fat) on the procedural chest wall
    shape = tuple(int(s / VOX) for s in (12, 70, 60))
    label = np.full(shape, tissue.FAT, np.uint8)
    z = np.full(shape, 1.35, np.float16)
    c = np.full(shape, 1440.0, np.float16)
    al = np.full(shape, 0.4 * tissue.NP_PER_M_PER_DB_PER_CM, np.float16)
    vol = phantom.Volume("test/chest", label, z, al, VOX, 1.0, np.full(shape[1:], 12.0, np.float32),
                         np.zeros(shape[1:], bool), {}, c, np.zeros(shape, np.uint8))
    phantom.add_chest_wall(vol)
    assert vol.a0_mm == pytest.approx(-25.0) and vol.label.shape[0] == shape[0] + 125
    r, _, meta, labels = render(vol, depth_mm=40.0, edge_shadow_db=12.0)
    present = set(np.unique(labels).tolist())
    assert {tissue.MUSCLE, tissue.RIB, tissue.LUNG} <= present
    inten = 10 * np.log10(r.last_intensity.cpu().numpy() + 1e-12)
    rib_cols = np.nonzero((labels == tissue.RIB).any(0))[0]
    gap_cols = np.nonzero(~(labels == tissue.RIB).any(0))[0]
    rib_rows = np.nonzero((labels == tissue.RIB).any(1))[0]
    # within the rib depth range: behind the bone surface vs intercostal muscle at the same depth
    # (below the pleura both are dark: air passes nothing but multiple reflections)
    deep = slice(rib_rows.min() + int(2 / meta["px_z_mm"]), rib_rows.max() - int(1 / meta["px_z_mm"]))
    rib_cols = rib_cols[(labels[deep][:, rib_cols] == tissue.RIB).mean(0) > 0.9]
    gap_cols = gap_cols[(labels[deep][:, gap_cols] == tissue.MUSCLE).mean(0) > 0.9]
    # bone shadows: far darker behind the rib surface than the intercostal muscle beside it
    assert inten[deep][:, rib_cols].mean() < inten[deep][:, gap_cols].mean() - 10


def test_transmit_focal_zone_is_brightest_and_narrowest():
    # homogeneous speckle, no attenuation: speckle energy peaks at the transmit focus
    vol = layered(tissue.GLAND, tissue.GLAND, 15.0, z_top=1.6, z_bottom=1.6)
    r, _, meta, _ = render(vol, tx_focus_mm=20.0, depth_mm=40.0, reverberation=False)
    inten = r.last_intensity.cpu().numpy()
    zz = (np.arange(inten.shape[0]) + 0.5) * meta["px_z_mm"]
    prof = np.convolve(10 * np.log10(inten[:, 40:-40].mean(1)), np.ones(25) / 25, mode="same")
    sel = (zz > 5) & (zz < 37)
    assert abs(zz[sel][np.argmax(prof[sel])] - 20.0) < 5.0
    assert prof[sel].max() - prof[(zz > 4) & (zz < 6)].mean() > 1.0


def test_flat_strong_reflector_reverberates_at_twice_its_depth():
    # smooth soft-tissue / high-impedance layer at 12 mm (r ~ 0.5): probe-face replica at 24 mm
    vol = layered(tissue.FAT, tissue.GLAND, 12.0, z_top=1.40, z_bottom=4.2)
    r_on, _, meta, _ = render(vol, echo=False, depth_mm=30.0, aberration=False)
    r_off, _, _, _ = render(vol, echo=False, depth_mm=30.0, aberration=False, reverberation=False)
    zz = (np.arange(r_on.last_intensity.shape[0]) + 0.5) * meta["px_z_mm"]
    on = 10 * np.log10(r_on.last_intensity.cpu().numpy().mean(1) + 1e-20)
    off = 10 * np.log10(r_off.last_intensity.cpu().numpy().mean(1) + 1e-20)
    near = (zz > 23) & (zz < 25)
    assert on[near].max() > off[near].max() + 20


def test_aberration_follows_speed_of_sound_heterogeneity():
    flat = layered(tissue.GLAND, tissue.GLAND, 15.0, z_top=1.6, z_bottom=1.6)
    _, _, m0, _ = render(flat)
    assert m0["aberration_rms_ns"] == pytest.approx(0.0, abs=0.5)
    # fat lobules (1440 m/s) in gland (1540 m/s): tens of ns, as measured in breast (Hinkelman 1995: 67 ns)
    vol = sphere_in_fat()
    _, _, m1, _ = render(vol)
    assert m1["aberration_rms_ns"] > 2.0
