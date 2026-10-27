#!/usr/bin/env python
"""Lesion type table for the simulator (breastsim.lesions) from the BrEaST annotations.

    python src/scripts/lesion_types.py [--temper 0.5]

Every annotated BrEaST lesion (Pawlowska et al. 2024, CC BY 4.0) is one joint sample of BI-RADS descriptors
(pathology, shape, margin, echo pattern, posterior features, halo, calcifications) together with the lesion
features measured on its image by breastsim.realism (size, depth, h/w, contrast, heterogeneity, halo,
posterior, edge sharpness; outputs/realism/features/BrEaST_lesions.csv, the largest component per case).
The simulator samples a case, then builds geometry and acoustics from that case, so attributes co-occur as
in real data. Rare types are moderately up-weighted: a case of type key k (echo pattern, calcifications in
a mass) gets weight n_k^(temper - 1), i.e. the type frequencies become n_k^temper (temper 1 = real
frequencies, 0 = uniform over types). Writes src/breastsim/lesion_types.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from breastsim import paths

OUT = Path(__file__).resolve().parents[1] / "breastsim" / "lesion_types.json"
FEATS = ["eqdiam_mm", "depth_mm", "h_over_w", "contrast_sd", "inner_heterogeneity", "halo_sd", "posterior_sd",
         "edge_sharpness", "solidity", "irregularity"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--temper", type=float, default=0.5)
    a = ap.parse_args()
    meta = pd.read_excel(next((paths.REAL / "breast_lesions_usg").glob("BrEaST-Lesions-USG-clinical-data-*.xlsx")))
    les = pd.read_csv(paths.OUTPUTS / "realism" / "features" / "BrEaST_lesions.csv")
    les["Image_filename"] = les.id.map(lambda s: Path(s).name)
    les = les.sort_values("eqdiam_mm", ascending=False).drop_duplicates("Image_filename")   # main lesion per case
    m = meta.merge(les[["Image_filename"] + FEATS], on="Image_filename", how="inner")
    m = m[m.Classification != "normal"].copy()
    m["calc"] = m.Calcifications.eq("in a mass")
    m["key"] = m.Echogenicity + np.where(m.calc, " | calcified", "")
    n = m.key.map(m.key.value_counts())
    w = n.astype(float) ** (a.temper - 1)
    m["weight"] = w / w.sum()
    cases = []
    for _, r in m.iterrows():
        cases.append({"case": r.Image_filename, "pathology": r.Classification, "diagnosis": r.Diagnosis,
                      "shape": r.Shape, "margin": r.Margin, "echo": r.Echogenicity, "posterior": r.Posterior_features,
                      "halo": r.Halo == "yes", "calc": bool(r.calc), "key": r.key, "weight": round(float(r.weight), 6),
                      **{f: (None if pd.isna(r[f]) else round(float(r[f]), 4)) for f in FEATS}})
    types = (m.groupby("key").agg(n=("key", "size"), p_sim=("weight", "sum")).assign(p_real=lambda t: t.n / t.n.sum())
             .sort_values("n", ascending=False).round(4))
    doc = {"source": "BrEaST-Lesions_USG annotations (Pawlowska et al., Sci Data 2024, CC BY 4.0) + features "
                     "measured with breastsim.realism on the same images",
           "temper": a.temper, "type_key": "echo pattern | calcifications in a mass",
           "types": types.reset_index().to_dict("records"), "cases": cases}
    OUT.write_text(json.dumps(doc, indent=1))
    print(types.to_string())
    print(f"{len(cases)} cases -> {OUT}")


if __name__ == "__main__":
    main()
