#!/usr/bin/env python
"""Rendered lesion contrast vs lesion / host backscatter ratio (breastsim.lesions.CONTRAST_CURVE).

    python src/scripts/lesion_calibrate.py --phantoms 'victre/typ_*' --out outputs/lesion_calibration.json

Every lesion of >= 4 mm of the given phantoms is rendered (preset breast, --poses frames through its centre
at random rotations) with its backscatter amplitude set to ratio x the host rms (circumscribed, no halo, no
posterior feature) and each --rims factor of the boundary rim, and contrast_sd / inner_heterogeneity / edge_sharpness are measured with
breastsim.realism.lesion_features exactly as on the real images. Prints the median per ratio.
"""
import argparse
import fnmatch
import json

import numpy as np
from scipy import ndimage as ndi

from breastsim import lesions, phantom, realism, tissue
from breastsim.probe import ProbePose
from breastsim.realtime.renderer import ImagingParams, PreviewRenderer


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phantoms", required=True)
    ap.add_argument("--ratios", default="0.05,0.1,0.2,0.35,0.5,0.7,1.0,1.4,2.0")
    ap.add_argument("--poses", type=int, default=3)
    ap.add_argument("--rims", default="1.0", help="rim factors (breastsim.lesions EDGE_CURVE), e.g. 1.0,1.4,1.8,2.4")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ratios = [float(x) for x in a.ratios.split(",")]
    pids = [i.id for i in phantom.list_phantoms() if any(fnmatch.fnmatch(i.id, q) for q in a.phantoms.split(","))]
    p = ImagingParams.from_preset("breast", overlay="none")
    rend, rows = PreviewRenderer(), []
    rng = np.random.default_rng(0)
    for pid in pids:
        vol = phantom.load(pid, log=lambda *x: None)
        f = phantom.cache_dir(pid) / "echo_mod.npy"
        base = np.load(f) if f.exists() else np.ones(vol.label.shape, np.float16)
        main = [l for l in vol.meta["lesions"] if l["diameter_mm"] >= 4]
        poses = {l["id"]: rng.uniform(-180, 180, a.poses) for l in main}
        for ratio, rim in [(q, f) for q in ratios for f in [float(x) for x in a.rims.split(",")]]:
            vol.echo_mod = base.copy()
            for l in vol.meta["lesions"]:
                l["recipe"] = {**l["recipe"], "echo_ratio": ratio, "echo": "hypoechoic", "margin": "circumscribed",
                               "halo": False, "rim_factor": rim}
                l["attenuation_db_cm_mhz"] = 0.6
            lesions.apply_content(vol)
            rend.set_volume(vol)
            for l in main:
                for rot in poses[l["id"]]:
                    img, meta, lab = rend.render(ProbePose(u=l["center_mm"][1], v=l["center_mm"][2], rotation=float(rot)), p)
                    les = np.isin(lab, tissue.LESIONS)
                    comp, n = ndi.label(les)
                    if not n:
                        continue
                    les = comp == 1 + np.argmax(ndi.sum(les, comp, range(1, n + 1)))
                    if les.sum() < 50:
                        continue
                    g = img[..., 0].astype(np.float32)
                    f_ = realism.lesion_features(g, les, realism.tissue_mask(g), meta["px_z_mm"])
                    rows.append({"phantom": pid, "lesion": l["id"], "ratio": ratio, "rim": rim, **f_})
        print(pid, len(rows), flush=True)
    json.dump(rows, open(a.out, "w"))
    import pandas as pd
    t = pd.DataFrame(rows).groupby(["ratio", "rim"])[["contrast_sd", "inner_heterogeneity", "edge_sharpness", "halo_sd"]]
    print(t.median().round(3).to_string())
    print("n per ratio:", t.size().to_dict())


if __name__ == "__main__":
    main()
