"""Scan presets that approximate the acquisition of public breast US datasets.

Grounded in each dataset's metadata (checked on lab127, 2026-10-22):
- BUSI (Al-Dhabyani et al. 2020): GE LOGIQ E9 / E9 Agile, ML6-15-D matrix
  linear probe; PNG images ~500x500 px.
- BUS-BRA (Gomez-Flores et al. 2024): GE Logiq 7 @10-14 MHz (902), Logiq 5
  @10-12 MHz (809), Toshiba Aplio 300 @12-14 MHz (139), U-Systems @10-14 MHz
  (25); lesion-centred crops, median 286x383 px.
- BrEaST-Lesions_USG (Pawlowska et al. 2024): multi-vendor handheld linear;
  pixel size ~0.078 mm (0.0059-0.0078 cm), images ~485x537 px -> ~38x42 mm.
- TDSC-ABUS 2023: automated 3D breast US, volumes ~865x608x330 voxels,
  0.2 mm along the array, 0.073 mm axial (608 -> 44 mm depth), 0.476 mm
  between frames; breast compressed flat under the membrane.
Probe specs for the clinical scanners are approximations (see probe.PROBES).
"""
from dataclasses import asdict, dataclass


@dataclass
class ScanPreset:
    name: str
    label: str
    probe: str
    depth_mm: float
    px_x_mm: float               # output pixel spacing along the array
    px_z_mm: float               # output pixel spacing in depth
    gain_db: float
    dynamic_range_db: float
    speckle_reduction: float     # Gaussian sigma (output px) on the log image, vendor-style smoothing
    lateral_f_number: float
    conform: bool = True
    tgc_db_cm_mhz: float = 0.5   # TGC slope (one way, dB/cm/MHz^y)
    gray_gamma: float = 1.0      # display gray map exponent
    tx_f_number: float = 2.0     # transmit aperture = depth / F# (focused at every depth)
    compound_deg: tuple = ()     # spatial compounding steering angles, deg
    noise_db: float = -200.0     # receive noise re unit-echogenicity speckle at the face, dB         # probe presses the skin flat
    sweep_step_mm: float = 0.0   # ABUS frame spacing (data generation)
    notes: str = ""

    def to_dict(self):
        return asdict(self)


PRESETS = {p.name: p for p in [
    ScanPreset("busi", "BUSI-like (LOGIQ E9, ML6-15)", "ML6-15", 40.0, 0.1, 0.1, -14, 50, 1.2, 1.8,
               notes="handheld, 50 mm footprint, ~500x500 px"),
    ScanPreset("busbra", "BUS-BRA-like (Logiq 5/7, 10-14 MHz)", "L12-hf", 35.0, 0.1, 0.1, -13, 50, 0.8, 2.0,
               notes="handheld; dataset images are lesion-centred crops"),
    # physical beam (receive F# 2, transmit F# 3 focused at every depth, no compounding: compounding and
    # vendor filtering belong to the post-processing stage) with receive noise at -75 dB re the
    # near-field speckle; chosen with the realism benchmark (breastsim.realism, 2026-10-23: imaging
    # features within the spread between BUSI / BUS-BRA / BrEaST / TDSC-ABUS). The earlier fit of
    # effective F-numbers to whole-image speckle sizes was dropped: it widened the PSF to imitate
    # horizontal anatomy and blurred the deep field. Display chain refitted to BrEaST with this beam.
    ScanPreset("breast", "BrEaST-like (0.078 mm px, 38x42 mm)", "L12-hf", 42.0, 0.078, 0.078, -25.4, 70.0, 0.4, 2.0,
               tgc_db_cm_mhz=0.76, gray_gamma=2.30, tx_f_number=3.0, noise_db=-75.0,
               notes="handheld multi-vendor, native pixel size"),
    ScanPreset("tdsc_abus", "TDSC-ABUS-like (15.4 cm ABUS)", "ABUS-15", 44.4, 0.2, 0.073, -12, 50, 0.6, 2.0,
               sweep_step_mm=0.476, notes="automated sweep, 0.2 x 0.073 mm, 0.476 mm frames"),
    ScanPreset("l11_5v", "Verasonics L11-5v (PyMUST reference)", "L11-5v", 40.0, 0.1, 0.1, -12, 55, 0.0, 2.0,
               notes="raw beamformed look, same probe as the PyMUST generator"),
]}
DEFAULT_PRESET = "breast"
