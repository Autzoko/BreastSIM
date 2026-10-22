# data/

Open-source resources, fetched by `src/scripts/fetch_data.sh` (not committed).
Large datasets are symlinks into the host's big-storage root
(`$BREASTSIM_BIG`, see `src/scripts/hostenv.sh`).

| Path | Source | License | Notes |
|---|---|---|---|
| `nbp3d_usct/` | [Dataverse KBYQQ7](https://doi.org/10.7910/DVN/KBYQQ7) — Li et al., TUFFC 2021 | CC0 | 4 3D USCT phantoms (BI-RADS A–D), 0.1 mm: label / sos / density / aa (.mat) — 9.6 GB |
| `nbp2d_usct/dataverse/` | [Dataverse CUFVKE](https://doi.org/10.7910/DVN/CUFVKE) | CC0 | 2D slices + imaging system only (916 GB of USCT measurements skipped) |
| `nbp2d_usct/idb/` | [IDB-5648161](https://doi.org/10.13012/B2IDB-5648161_V1) | CC0 | 1,089 2D phantom slices — 3.3 GB zip |
| `oa_breast/` | [OA-Breast](https://anastasio.bioengineering.illinois.edu/downloadable-content/oa-breast-database/) | ODbL / DbCL | 3 MRI-derived phantoms (HDF5), 0.2 mm, no lesions — 4.75 GB |
| `nbp3d_oat64/` | [IDB-8164905](https://doi.org/10.13012/B2IDB-8164905_V1) — Park et al. 2025 | CC0 | 64 sets, 0.125 mm, 1360×1360×680. Kept: `*{h,l}_label.mat`, `*{h,l}_acou.mat`. Dropped: func/opt/phi/p0/p (optoacoustic) |
| `real/busbra/` | [BUS-BRA, Zenodo 8231412](https://zenodo.org/records/8231412) | CC BY 4.0 | 2D real breast US + masks |
| `real/breast_lesions_usg/` | [BrEaST-Lesions_USG, TCIA](https://www.cancerimagingarchive.net/collection/breast-lesions-usg/) | CC BY 4.0 | 2D real breast US + masks + clinical data |
| `real/busi/` | BUSI (manual download) | CC0 | scripted download blocked by host |
| `real/abus_tdsc/` | [TDSC-ABUS 2023](https://tdsc-abus2023.grand-challenge.org/) | challenge terms | 3D ABUS; symlink to existing copy on lab127 |

IDB-8164905 label values: 0 background, 1 fat, 2 dermis, 3 epidermis, 29 glandular,
33 nipple, 88 ligament, 95 TDLU, 125 duct, 150 artery, 160 simple cyst,
180 fibroadenoma, 190 peripheral angiogenesis, 200 viable tumor, 210 necrotic core, 225 vein.
