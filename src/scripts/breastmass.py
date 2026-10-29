"""breastMass (FDA VICTRE, official binary, unmodified) with its full, documented configuration.

Victre.Pipeline.generate_spiculated fills Victre/breastMass/configs/spiculated.tpl with string.Template, and that
template only has placeholders for imgRes, complexity, seed and alpha: every other breastMass key given through
`arguments_spiculated` (meanShape, meanInitial / spicules, meanLF*, meanSigma2, ...) is silently ignored and the
template's fixed values are used. Affected (same mass for a seed whatever the "kind"): victre_masses.py,
victre_insert.py, victre_insert_at.py, and victre_lesion_typed.py before 2026-10-29.

generate_mass() writes the same template with every key set explicitly (each key must exist in it) and runs the
binary the pipeline uses; runs inside the breastsim/victre container.
"""
import glob
import os
import re
import subprocess

import numpy as np

TEMPLATE = "/opt/VICTRE_PIPELINE/Victre/breastMass/configs/spiculated.tpl"
BINARY = "/opt/VICTRE_PIPELINE/Victre/breastMass/build/breastMass"


def generate_mass(folder, seed, alpha, voxel, complexity=1.0, **keys):
    """Mass volume (bool, cubic) from breastMass with mean radius alpha (mm) at `voxel` mm and the given keys."""
    os.makedirs(folder, exist_ok=True)
    cfg = open(TEMPLATE).read()
    for k, v in {"imgRes": voxel, "complexity": complexity, "seed": int(seed), "alpha": alpha}.items():
        cfg = cfg.replace(f"${k}", str(v))
    for k, v in keys.items():
        cfg, n = re.subn(rf"^{k}=.*$", f"{k}={v}", cfg, flags=re.M)
        if n != 1:
            raise KeyError(f"breastMass key {k!r} not in {TEMPLATE}")
    name = f"mass_cfg_{seed}.in"
    with open(os.path.join(folder, name), "w") as f:
        f.write(cfg)
    subprocess.run(f"cd {folder} && {BINARY} -c {name}", shell=True, check=True, capture_output=True)
    raw = sorted(glob.glob(os.path.join(folder, f"mass_{seed}_*.raw")))[-1]
    side = int(re.search(r"mass_\d+_(\d+)\.raw$", raw)[1])
    vol = np.fromfile(raw, np.uint8).reshape(side, side, side) == 1
    for f in glob.glob(os.path.join(folder, f"mass_{seed}_*")):
        os.remove(f)
    return vol
