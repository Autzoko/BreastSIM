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
CALC = 250           # calcification (VICTRE calcification cluster material, inserted inside masses)

NAMES = {
    BACKGROUND: "background", FAT: "fat", SKIN: "skin", EPIDERMIS: "epidermis",
    GLAND: "glandular", NIPPLE: "nipple", LIGAMENT: "ligament", TDLU: "TDLU",
    DUCT: "duct", ARTERY: "artery", CYST: "simple cyst", FIBROADENOMA: "fibroadenoma",
    ANGIOGENESIS: "peripheral angiogenesis", TUMOR: "viable tumor", NECROSIS: "necrotic core",
    VEIN: "vein", MUSCLE: "muscle", FASCIA: "fascia", RIB: "rib", LUNG: "lung (air)", CALC: "calcification",
}

# Labels that make up the segmentation / detection target. Peripheral
# angiogenesis (IDB-8164905 label_pa) is an optical/functional property of the
# tissue around a tumour, not a sonographic mass, so it is not a lesion label.
LESIONS = (CYST, FIBROADENOMA, TUMOR, NECROSIS, CALC)   # calcifications only occur inside masses (part of the mask)

# Diffuse fraction of boundary echoes (specular otherwise): soft-tissue interfaces
# are smooth, the cortical bone surface and the pleura are rough and stay bright
# at oblique incidence.
ROUGHNESS = {RIB: 0.8, LUNG: 0.6, CALC: 0.8}
DEFAULT_ROUGHNESS = 0.25

# Sheet- or tube-like structures thinner than the ultrasound beam.
# Calcifications (0.2-0.6 mm here) are smaller than the beam: they reflect strongly but the beam passes around
# them (no bone-like blocking in the renderer).
THIN_STRUCTURES = (SKIN, EPIDERMIS, LIGAMENT, DUCT, ARTERY, VEIN, FASCIA, CALC)

# Relative backscatter amplitude (std of sub-resolution scatterers).
# Breast US appearance: fat hypoechoic, glandular / ligaments / skin
# hyperechoic, solid masses hypoechoic, cysts and vessels anechoic.
ECHOGENICITY = {
    BACKGROUND: 0.0, FAT: 0.35, SKIN: 1.1, EPIDERMIS: 1.3, GLAND: 1.0, NIPPLE: 0.5,
    LIGAMENT: 1.4, TDLU: 0.7, DUCT: 0.1, ARTERY: 0.03, CYST: 0.02, FIBROADENOMA: 0.25,
    ANGIOGENESIS: 0.4, TUMOR: 0.1, NECROSIS: 0.05, VEIN: 0.03,
    MUSCLE: 0.3, FASCIA: 1.3, RIB: 0.0, LUNG: 0.0, CALC: 3.0,
}
# Test option (not adopted): breast fat backscatter 13-15 dB below fibroglandular tissue in vivo (Anderson,
# Soo, Trahey, UMB 2001, 7-10 MHz) -> fat ~0.2 instead of 0.35 (9.1 dB). Adopting it needs a display refit.
ECHOGENICITY[FAT] = float(__import__("os").environ.get("BREASTSIM_FAT_ECHO", ECHOGENICITY[FAT]))

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

# VICTRE phantoms are label maps only (the IDB-8164905 and USCT phantoms are VICTRE label maps
# with acoustic properties drawn per tissue). Their properties: mean over the 128 IDB-8164905
# phantoms on lab127 (per-tissue spread 1-3 % in c and rho, ~10 % in alpha); muscle from
# DEFAULT_ACOUSTIC. (c m/s, rho kg/m^3, alpha Np/m/MHz^y)
VICTRE_ACOUSTIC = {
    BACKGROUND: (1500.0, 1000.0, 0.03), FAT: (1446.8, 906.0, 4.37), SKIN: (1550.2, 1101.0, 19.87),
    GLAND: (1537.1, 1037.0, 8.29), NIPPLE: (1554.4, 1111.0, 20.68), LIGAMENT: (1474.3, 1003.0, 7.98),
    TDLU: (1532.9, 1033.0, 8.18), DUCT: (1518.4, 1012.0, 7.58), ARTERY: (1536.2, 1012.0, 4.29),
    VEIN: (1536.7, 1012.0, 4.31), TUMOR: (1530.1, 955.0, 23.27), MUSCLE: DEFAULT_ACOUSTIC[MUSCLE],
    FASCIA: DEFAULT_ACOUSTIC[FASCIA],
    CALC: DEFAULT_ACOUSTIC[RIB],      # hydroxyapatite: bone-like values (cortical bone, DEFAULT_ACOUSTIC)
}
# power-law exponent per VICTRE density class, as the IDB-8164905 / USCT phantoms of BI-RADS A-D
VICTRE_DENSITY_BIRADS = {"fatty": "A", "scattered": "B", "hetero": "C", "dense": "D"}
VICTRE_ALPHA_POWER = {"fatty": 1.1151, "scattered": 1.1642, "hetero": 1.2563, "dense": 1.3635}
# Frequency exponent used for imaging (renderer). The phantoms' exponents (USCT / OA-Breast / VICTRE
# classes A-D: 1.12-1.36) are one homogenised y per breast, fitted over 0.2-2.3 MHz for USCT
# solvers that need a single global y (Li et al. 2021, Table IV); extrapolated to 7-12 MHz they
# double the attenuation of dense breasts. With y = 1 the phantoms' per-tissue coefficients (IT'IS)
# read 0.38 (fat) and 0.75 (gland) dB/cm/MHz, the clinical-band values, as linear simulators
# (Field II, PyMUST) quote them.
IMAGING_ALPHA_POWER = 1.0

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
# In vivo attenuation slopes of breast masses (Nam et al., UMB 2013, PMC3676873, 3-13 MHz, reference phantom
# method): carcinomas 1.16 dB/cm/MHz (0.51-1.77, n=10), fibroadenomas 0.94 (0.37-1.63, n=14); the masses with
# posterior shadowing are at the top of that range. 1.8-3.5 rendered complete, full-depth black columns behind
# 10-15 mm masses (real shadows are partial).
POSTERIOR_ATTENUATION_DB_CM_MHZ = {
    "shadowing": (1.3, 1.8), "combined": (0.9, 1.4), "none": (0.4, 0.8), "enhancement": (0.05, 0.3),
}
CYST_ATTENUATION_DB_CM_MHZ = (0.0, 0.05)        # fluid, whatever the drawn feature
NP_PER_M_PER_DB_PER_CM = 100 / 8.686

# Overlay colors (RGB) for label views.
COLORS = {
    BACKGROUND: (0, 0, 0), FAT: (230, 200, 120), SKIN: (240, 150, 150), EPIDERMIS: (250, 120, 120),
    GLAND: (120, 170, 240), NIPPLE: (200, 120, 200), LIGAMENT: (250, 250, 250), TDLU: (90, 140, 220),
    DUCT: (60, 90, 200), ARTERY: (220, 40, 40), CYST: (40, 220, 220), FIBROADENOMA: (240, 200, 40),
    ANGIOGENESIS: (250, 120, 40), TUMOR: (250, 40, 40), NECROSIS: (130, 20, 20), VEIN: (60, 40, 200),
    MUSCLE: (170, 60, 70), FASCIA: (235, 210, 210), RIB: (245, 240, 220), LUNG: (40, 40, 60), CALC: (255, 255, 255),
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
