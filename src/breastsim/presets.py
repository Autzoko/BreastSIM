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
    compound_deg: tuple = ()     # spatial compounding steering angles, deg         # probe presses the skin flat
    sweep_step_mm: float = 0.0   # ABUS frame spacing (data generation)
    notes: str = ""

    def to_dict(self):
        return asdict(self)


PRESETS = {p.name: p for p in [
    ScanPreset("busi", "BUSI-like (LOGIQ E9, ML6-15)", "ML6-15", 40.0, 0.1, 0.1, -14, 50, 1.2, 1.8,
               notes="handheld, 50 mm footprint, ~500x500 px"),
    ScanPreset("busbra", "BUS-BRA-like (Logiq 5/7, 10-14 MHz)", "L12-hf", 35.0, 0.1, 0.1, -13, 50, 0.8, 2.0,
               notes="handheld; dataset images are lesion-centred crops"),
    # speckle / PSF fitted to BrEaST speckle statistics at 0.1 mm/px (2026-10-23, 16 frames x 40 settings):
    # receive F# 4, transmit F# 6 (effective values), smoothing 0.4 px, 3-angle compounding (+-10 deg):
    # axial / lateral speckle 0.28 / 0.86 mm (BrEaST 0.33 / 1.03), aspect 3.1 (2.8), fine-scale
    # energy 0.064 (0.055). Display chain then refitted to BrEaST (src/scripts/calibrate_display.py).
    ScanPreset("breast", "BrEaST-like (0.078 mm px, 38x42 mm)", "L12-hf", 42.0, 0.078, 0.078, -22.9, 70.0, 0.4, 4.0,
               tgc_db_cm_mhz=0.66, gray_gamma=2.54, tx_f_number=6.0, compound_deg=(-10.0, 0.0, 10.0),
               notes="handheld multi-vendor, native pixel size"),
    ScanPreset("tdsc_abus", "TDSC-ABUS-like (15.4 cm ABUS)", "ABUS-15", 44.4, 0.2, 0.073, -12, 50, 0.6, 2.0,
               sweep_step_mm=0.476, notes="automated sweep, 0.2 x 0.073 mm, 0.476 mm frames"),
    ScanPreset("l11_5v", "Verasonics L11-5v (PyMUST reference)", "L11-5v", 40.0, 0.1, 0.1, -12, 55, 0.0, 2.0,
               notes="raw beamformed look, same probe as the PyMUST generator"),
]}
DEFAULT_PRESET = "breast"
