"""Locations of repository directories and downloaded resources."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("BREASTSIM_DATA", ROOT / "data"))
LIB = ROOT / "lib"
OUTPUTS = ROOT / "outputs"
# Derived caches live next to the big resources (data-001 on lab127).
CACHE = Path(os.environ.get("BREASTSIM_BIG", DATA)) / "cache"

# Resource directories under DATA (large ones are symlinks into BREASTSIM_BIG).
NBP3D = DATA / "nbp3d_usct"            # Dataverse KBYQQ7: 4 USCT phantoms (A-D)
NBP2D = DATA / "nbp2d_usct"            # Dataverse CUFVKE 2D_slices + IDB-5648161
OAT_NBP = DATA / "nbp3d_oat64"         # IDB-8164905: 64 sets, label + acou only
OA_BREAST = DATA / "oa_breast"         # OA-Breast HDF5 phantoms
REAL = DATA / "real"                   # real ultrasound datasets (BUS-BRA, BrEaST, BUSI, ABUS_TDSC)
