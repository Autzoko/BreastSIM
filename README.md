# BreastSIM

Breast ultrasound simulation platform for training and studying lesion
segmentation / detection models (2D B-mode and 3D ABUS-like volumes).

Two sides share the same phantom, slice geometry, lesion masks and random seeds,
so a preview and its generated sample show the same content:

- **Real-time side** (`src/breastsim/realtime`) — interactive probe placement with a
  fast approximation (precomputed PSF convolution on GPU).
- **Data-generation side** (`src/breastsim/datagen`) — high-fidelity batch
  simulation (PyMUST/SIMUS sweeps, k-Wave full-wave) written as sharded HDF5.

## Layout

```text
data/   downloaded open-source resources (git-ignored; big ones symlinked)
lib/    third-party source dependencies (git-ignored)
src/    breastsim package + scripts
_ai/    AI working notes (git-ignored)
```

## Hosts

| Host | Role | Env | Big storage |
|---|---|---|---|
| local Mac | development | conda env `breastsim` | `data/` |
| NYUAIRLAB-127 (1× A6000) | real-time server, preprocessing | `~/langtian/envs/breastsim` | `~/data-001/langtian/BreastSIM` |
| JUBAIL (Slurm) | large-scale generation | `/scratch/ll5582/conda/envs/breastsim` | `/scratch/ll5582/BreastSIM_data` |

Code is synced through this git repository; data and envs are prepared per host.

## Setup

```bash
bash src/scripts/setup_lib.sh      # clone lib/ dependencies
bash src/scripts/setup_env.sh      # create/update the conda env for this host
bash src/scripts/fetch_data.sh     # download resources (resumable; --list / --only)
python -m breastsim.phantom        # list phantoms; `python -m breastsim.phantom <id>` builds a preview cache
```

## Real-time preview from the Mac

The server runs on NYUAIRLAB-127's GPU, bound to localhost; the Mac reaches
it through an SSH tunnel:

```bash
bash src/scripts/realtime_remote.sh            # start server in tmux if needed, tunnel, open browser
bash src/scripts/realtime_remote.sh --restart  # pull latest code and restart the server
bash src/scripts/realtime_remote.sh --stop     # stop the remote server
```

Probe Lab (browser): 3D view of skin / glandular tissue / lesions with the
probe and the live scan plane (click the skin to place the probe, drag to
orbit); lesion list with "Go"; scan presets that approximate BUSI, BUS-BRA,
BrEaST and TDSC-ABUS acquisition (`src/breastsim/presets.py`); keys W/S
(elevation), A/D (lateral), Q/E (rotate), R/F (tilt), Z/X (rock), Shift for
larger steps. "Save scene" writes phantom, pose and imaging parameters to
`outputs/scenes/` for the data-generation side.

Checks:

```bash
python -m pytest                                              # unit + server tests (synthetic phantom)
python src/scripts/render_preview.py usct/A --preset breast  # headless frames
PLAYWRIGHT_BROWSERS_PATH=~/langtian/envs/.playwright python src/scripts/e2e_check.py   # UI end-to-end (127)
```
