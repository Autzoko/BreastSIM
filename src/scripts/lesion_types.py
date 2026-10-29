#!/usr/bin/env python
"""Lesion type table for the simulator (breastsim.lesions) from the BrEaST annotations.

    python src/scripts/lesion_types.py [--cap 0.12 --boost 2]

Every annotated BrEaST lesion (Pawlowska et al. 2024, CC BY 4.0) is one joint sample of BI-RADS descriptors
(pathology, shape, margin, echo pattern, posterior features, halo, calcifications) together with the lesion
features measured on its image by breastsim.realism (size, depth, h/w, contrast, heterogeneity, halo,
posterior, edge sharpness; outputs/realism/features/BrEaST_lesions.csv, the largest component per case).
The simulator samples a case, then builds geometry and acoustics from that case, so attributes co-occur as
in real data. Rare types are moderately up-weighted: a type key k (echo pattern, calcifications in a mass) of real
share p_k < --rare gets the target share min(--boost x p_k, --cap) (2x, at most 12 % each); the common types share
the rest in their real proportions; case weight = share_k / n_k. The json's "types" list holds the per-type target
shares ("share") for the dataset builder. --cap 0: tempered shares p_k^temper instead. Writes
src/breastsim/lesion_types.json.
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
    ap.add_argument("--temper", type=float, default=0.5, help="used with --cap 0")
    ap.add_argument("--cap", type=float, default=0.12, help="max share of one rare type (2026-10-29 user decision)")
    ap.add_argument("--boost", type=float, default=2.0, help="rare types: this x their real share (until --cap)")
    ap.add_argument("--rare", type=float, default=0.10, help="real share below which a type counts as rare")
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
    p_real = m.key.value_counts() / len(m)
    if a.cap > 0:
        # rare types (real share < --rare): --boost x their real share, at most --cap of the lesion frames each;
        # the common types share the rest in their real proportions
        rare = p_real < a.rare
        share = pd.Series(np.where(rare, np.minimum(a.boost * p_real, a.cap), 0.0), index=p_real.index)
        share[~rare] = p_real[~rare] / p_real[~rare].sum() * (1 - share[rare].sum())
    else:
        share = p_real ** a.temper / (p_real ** a.temper).sum()
    m["weight"] = m.key.map(share) / n
    cases = []
    for _, r in m.iterrows():
        cases.append({"case": r.Image_filename, "pathology": r.Classification, "diagnosis": r.Diagnosis,
                      "shape": r.Shape, "margin": r.Margin, "echo": r.Echogenicity, "posterior": r.Posterior_features,
                      "halo": r.Halo == "yes", "calc": bool(r.calc), "key": r.key, "weight": round(float(r.weight), 6),
                      **{f: (None if pd.isna(r[f]) else round(float(r[f]), 4)) for f in FEATS}})
    types = (m.groupby("key").agg(n=("key", "size"), share=("weight", "sum")).assign(p_real=lambda t: t.n / t.n.sum())
             .sort_values("n", ascending=False).round(4))
    doc = {"source": "BrEaST-Lesions_USG annotations (Pawlowska et al., Sci Data 2024, CC BY 4.0) + features "
                     "measured with breastsim.realism on the same images",
           "sampling": ({"cap": a.cap, "boost": a.boost, "rare_below": a.rare} if a.cap > 0 else {"temper": a.temper}),
           "type_key": "echo pattern | calcifications in a mass",
           "types": types.reset_index().to_dict("records"), "cases": cases}
    OUT.write_text(json.dumps(doc, indent=1))
    print(types.to_string())
    print(f"{len(cases)} cases -> {OUT}")


if __name__ == "__main__":
    main()
