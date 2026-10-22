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

NAMES = {
    BACKGROUND: "background", FAT: "fat", SKIN: "skin", EPIDERMIS: "epidermis",
    GLAND: "glandular", NIPPLE: "nipple", LIGAMENT: "ligament", TDLU: "TDLU",
    DUCT: "duct", ARTERY: "artery", CYST: "simple cyst", FIBROADENOMA: "fibroadenoma",
    ANGIOGENESIS: "peripheral angiogenesis", TUMOR: "viable tumor", NECROSIS: "necrotic core",
    VEIN: "vein",
}

# Labels that make up the segmentation / detection target.
LESIONS = (CYST, FIBROADENOMA, ANGIOGENESIS, TUMOR, NECROSIS)

# Relative backscatter amplitude (std of sub-resolution scatterers).
# Breast US appearance: fat hypoechoic, glandular / ligaments / skin
# hyperechoic, solid masses hypoechoic, cysts and vessels anechoic.
ECHOGENICITY = {
    BACKGROUND: 0.0, FAT: 0.45, SKIN: 1.1, EPIDERMIS: 1.3, GLAND: 1.0, NIPPLE: 0.6,
    LIGAMENT: 1.4, TDLU: 0.8, DUCT: 0.15, ARTERY: 0.05, CYST: 0.03, FIBROADENOMA: 0.55,
    ANGIOGENESIS: 0.5, TUMOR: 0.3, NECROSIS: 0.15, VEIN: 0.05,
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
}
DEFAULT_ALPHA_POWER = 1.2

# Overlay colors (RGB) for label views.
COLORS = {
    BACKGROUND: (0, 0, 0), FAT: (230, 200, 120), SKIN: (240, 150, 150), EPIDERMIS: (250, 120, 120),
    GLAND: (120, 170, 240), NIPPLE: (200, 120, 200), LIGAMENT: (250, 250, 250), TDLU: (90, 140, 220),
    DUCT: (60, 90, 200), ARTERY: (220, 40, 40), CYST: (40, 220, 220), FIBROADENOMA: (240, 200, 40),
    ANGIOGENESIS: (250, 120, 40), TUMOR: (250, 40, 40), NECROSIS: (130, 20, 20), VEIN: (60, 40, 200),
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
