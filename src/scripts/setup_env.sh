#!/usr/bin/env bash
# Create/update the breastsim conda env for this host, then install
# PyMUST (editable, from lib/) and PyTorch (CUDA on Linux, MPS on macOS).
set -euo pipefail
source "$(dirname "$0")/hostenv.sh"
cd "$BREASTSIM_ROOT"

if [[ "$BREASTSIM_HOST" == local ]]; then
    TARGET=(-n "$BREASTSIM_ENV")
else
    TARGET=(-p "$BREASTSIM_ENV")
fi

if "$CONDA_BIN" env list | grep -q "$BREASTSIM_ENV"; then
    "$CONDA_BIN" env update "${TARGET[@]}" -f environment.yml
else
    "$CONDA_BIN" env create "${TARGET[@]}" -f environment.yml -y
fi

PIP=("${BREASTSIM_PY[@]}" -m pip)
[[ -d lib/PyMUST ]] || bash src/scripts/setup_lib.sh
"${PIP[@]}" install -e lib/PyMUST
"${PIP[@]}" install -e .

if [[ "$BREASTSIM_HOST" == local ]]; then
    "${PIP[@]}" install torch
else
    "${PIP[@]}" install torch --index-url https://download.pytorch.org/whl/cu124
fi

"${BREASTSIM_PY[@]}" -c "import numpy, scipy, h5py, pymust, kwave, torch, breastsim; \
print('python ok | torch', torch.__version__, '| cuda', torch.cuda.is_available())"
