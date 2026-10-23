"""Compare lesion statistics (src/scripts/lesion_stats.py output): real datasets vs simulation, median [p10, p90].

    python src/scripts/lesion_compare.py real/lesions.csv sim/lesions.csv out.png
"""
import sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
real = pd.read_csv(sys.argv[1]); sim = pd.read_csv(sys.argv[2])
d = pd.concat([real, sim], ignore_index=True)
d["group"] = d.dataset + ":" + d.pathology.astype(str)
cols = ["eqdiam_mm", "eqdiam_px", "h_over_w", "angle_deg", "solidity", "circularity", "irregularity", "depth_mm",
        "contrast", "contrast_sd", "inner_heterogeneity", "edge_sharpness", "posterior"]
def q(s):
    s = s.dropna()
    return "-" if len(s) < 5 else f"{s.median():.2f} [{s.quantile(.1):.2f},{s.quantile(.9):.2f}]"
t = d.groupby("group")[cols].agg(q)
t.insert(0, "n", d.groupby("group").size())
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 30)
print(t.T.to_string())
# shares
for name, f in (("taller_than_wide", lambda x: (x.h_over_w > 1).mean()), ("posterior<-5 (shadow)", lambda x: (x.posterior < -5).mean()),
                ("posterior>+5 (enhance)", lambda x: (x.posterior > 5).mean()), ("solidity<0.85 (irregular)", lambda x: (x.solidity < 0.85).mean())):
    print(name, {g: round(float(f(x)), 2) for g, x in d.groupby("group")})
fig, axs = plt.subplots(2, 4, figsize=(20, 8))
for ax, c in zip(axs.ravel(), ["eqdiam_mm", "h_over_w", "solidity", "irregularity", "contrast_sd", "edge_sharpness", "posterior", "depth_mm"]):
    for g, x in d.groupby("group"):
        v = x[c].dropna()
        if len(v) > 5:
            ax.hist(v.clip(v.quantile(.01), v.quantile(.99)), bins=30, histtype="step", density=True, label=f"{g} ({len(v)})",
                    lw=2.5 if g.startswith("SIM") else 1)
    ax.set_title(c)
axs[0, 0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(sys.argv[3], dpi=100)
