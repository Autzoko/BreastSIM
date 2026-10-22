"""Canonical tissue labels and their imaging properties.

All phantom sources are mapped onto the IDB-8164905 (VICTRE-derived) label
codes, a superset of the other sources. The same tables drive the real-time
preview and the high-fidelity generator, so both see identical content.
"""
import numpy as np

BACKGROUND = 0      # water / coupling gel
FAT = 1
SKIN = 2            # dermis
EPIDERMIS = 3
GLAND = 29
NIPPLE = 33
LIGAMENT = 88
TDLU = 95
DUCT = 125
ARTERY = 150
CYST = 160
FIBROADENOMA = 180
ANGIOGENESIS = 190
TUMOR = 200
NECROSIS = 210
VEIN = 225
# chest wall, generated procedurally behind the breast (not in the source phantoms)
MUSCLE = 40          # pectoralis / intercostal muscle
FASCIA = 45          # pectoral fascia and perimysial septa
RIB = 60
LUNG = 70            # aerated lung behind the pleura

NAMES = {
    BACKGROUND: "background", FAT: "fat", SKIN: "skin", EPIDERMIS: "epidermis",
    GLAND: "glandular", NIPPLE: "nipple", LIGAMENT: "ligament", TDLU: "TDLU",
    DUCT: "duct", ARTERY: "artery", CYST: "simple cyst", FIBROADENOMA: "fibroadenoma",
    ANGIOGENESIS: "peripheral angiogenesis", TUMOR: "viable tumor", NECROSIS: "necrotic core",
    VEIN: "vein", MUSCLE: "muscle", FASCIA: "fascia", RIB: "rib", LUNG: "lung (air)",
}

# Labels that make up the segmentation / detection target. Peripheral
# angiogenesis (IDB-8164905 label_pa) is an optical/functional property of the
# tissue around a tumour, not a sonographic mass, so it is not a lesion label.
LESIONS = (CYST, FIBROADENOMA, TUMOR, NECROSIS)

# Diffuse fraction of boundary echoes (specular otherwise): soft-tissue interfaces
# are smooth, the cortical bone surface and the pleura are rough and stay bright
# at oblique incidence.
ROUGHNESS = {RIB: 0.8, LUNG: 0.6}
DEFAULT_ROUGHNESS = 0.25

# Sheet- or tube-like structures thinner than the ultrasound beam.
THIN_STRUCTURES = (SKIN, EPIDERMIS, LIGAMENT, DUCT, ARTERY, VEIN, FASCIA)

# Relative backscatter amplitude (std of sub-resolution scatterers).
# Breast US appearance: fat hypoechoic, glandular / ligaments / skin
# hyperechoic, solid masses hypoechoic, cysts and vessels anechoic.
ECHOGENICITY = {
    BACKGROUND: 0.0, FAT: 0.35, SKIN: 1.1, EPIDERMIS: 1.3, GLAND: 1.0, NIPPLE: 0.5,
    LIGAMENT: 1.4, TDLU: 0.7, DUCT: 0.1, ARTERY: 0.03, CYST: 0.02, FIBROADENOMA: 0.25,
    ANGIOGENESIS: 0.4, TUMOR: 0.1, NECROSIS: 0.05, VEIN: 0.03,
    MUSCLE: 0.3, FASCIA: 1.3, RIB: 0.0, LUNG: 0.0,
}

# Acoustic properties for sources that only provide labels (OA-Breast):
# (speed of sound m/s, density kg/m^3, attenuation Np/m/MHz^y).
# c / rho from Lou et al. 2017 (OA-Breast), attenuation from Li et al. 2021.
DEFAULT_ACOUSTIC = {
    BACKGROUND: (1500.0, 1000.0, 0.025),
    FAT: (1470.0, 937.0, 4.36),
    SKIN: (1650.0, 1150.0, 21.2),
    GLAND: (1515.0, 1040.0, 8.64),
    VEIN: (1584.0, 1040.0, 1.5),
    # chest wall (Duck 1990 / IT'IS database typical values; attenuation in Np/m/MHz)
    MUSCLE: (1580.0, 1090.0, 11.5),
    FASCIA: (1550.0, 1120.0, 14.5),
    RIB: (3200.0, 1900.0, 230.0),
    LUNG: (343.0, 1.2, 1000.0),
}
DEFAULT_ALPHA_POWER = 1.2

# BI-RADS posterior acoustic features, drawn per lesion (deterministic per
# phantom and lesion) when a phantom is loaded. Frequencies from the BrEaST
# annotations (checked 2026-10-22): malignant n=98 -> none 52 %, shadowing
# 33 %, enhancement 10 %, combined 5 %; fibroadenoma n=30 -> none 67 %,
# enhancement 20 %, combined 7 %, shadowing 7 %. Simple cysts enhance.
# The VICTRE-derived phantoms give every tumour ~2.7 dB/cm/MHz^y (Li et al.
# 2021), i.e. a black shadow behind all of them; clinically posterior
# shadowing depends on the desmoplastic reaction and varies per lesion.
POSTERIOR_FEATURE_PROBS = {
    TUMOR: {"none": 0.52, "shadowing": 0.33, "enhancement": 0.10, "combined": 0.05},
    FIBROADENOMA: {"none": 0.67, "enhancement": 0.20, "combined": 0.07, "shadowing": 0.06},
    CYST: {"enhancement": 0.9, "none": 0.1},
}
# Lesion attenuation slope (dB/cm/MHz) realising each feature relative to
# breast fat (~0.4) and glandular tissue (~0.75).
POSTERIOR_ATTENUATION_DB_CM_MHZ = {
    "shadowing": (1.8, 3.5), "combined": (1.0, 1.8), "none": (0.4, 0.8), "enhancement": (0.05, 0.3),
}
CYST_ATTENUATION_DB_CM_MHZ = (0.0, 0.05)        # fluid, whatever the drawn feature
NP_PER_M_PER_DB_PER_CM = 100 / 8.686

# Overlay colors (RGB) for label views.
COLORS = {
    BACKGROUND: (0, 0, 0), FAT: (230, 200, 120), SKIN: (240, 150, 150), EPIDERMIS: (250, 120, 120),
    GLAND: (120, 170, 240), NIPPLE: (200, 120, 200), LIGAMENT: (250, 250, 250), TDLU: (90, 140, 220),
    DUCT: (60, 90, 200), ARTERY: (220, 40, 40), CYST: (40, 220, 220), FIBROADENOMA: (240, 200, 40),
    ANGIOGENESIS: (250, 120, 40), TUMOR: (250, 40, 40), NECROSIS: (130, 20, 20), VEIN: (60, 40, 200),
    MUSCLE: (170, 60, 70), FASCIA: (235, 210, 210), RIB: (245, 240, 220), LUNG: (40, 40, 60),
}


def lut(table, default=0.0, dtype=np.float32):
    """256-entry lookup array for a {label: value} table."""
    out = np.full(256, default, dtype=dtype)
    for k, v in table.items():
        out[k] = v
    return out


def color_lut():
    out = np.zeros((256, 3), np.uint8)
    for k, v in COLORS.items():
        out[k] = v
    return out
