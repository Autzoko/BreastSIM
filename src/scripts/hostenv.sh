#!/usr/bin/env bash
# Per-host settings shared by the setup / fetch scripts. Source this file.
#   BREASTSIM_HOST  local | lab127 | jubail
#   BREASTSIM_ROOT  repository root
#   BREASTSIM_ENV   conda env prefix (or name on the local Mac)
#   BREASTSIM_BIG   storage root for large resources (symlinked into data/)

BREASTSIM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ "$(uname -s)" == "Darwin" ]]; then
    BREASTSIM_HOST=local
    BREASTSIM_ENV=breastsim
    BREASTSIM_BIG="$BREASTSIM_ROOT/data"
    CONDA_BIN="${CONDA_EXE:-/opt/homebrew/Caskroom/miniforge/base/bin/conda}"
elif [[ -d /home/nyuair/data-001 ]]; then
    BREASTSIM_HOST=lab127
    BREASTSIM_ENV=/home/nyuair/langtian/envs/breastsim
    BREASTSIM_BIG=/home/nyuair/data-001/langtian/BreastSIM
    CONDA_BIN=/home/nyuair/miniconda3/bin/conda
elif [[ -d /scratch/ll5582 ]]; then
    BREASTSIM_HOST=jubail
    BREASTSIM_ENV=/scratch/ll5582/conda/envs/breastsim
    BREASTSIM_BIG=/scratch/ll5582/BreastSIM_data
    CONDA_BIN="$(command -v conda || echo conda)"
else
    echo "hostenv.sh: unknown host $(hostname)" >&2
    return 1 2>/dev/null || exit 1
fi

if [[ "$BREASTSIM_HOST" == local ]]; then
    BREASTSIM_PY=("$CONDA_BIN" run -n "$BREASTSIM_ENV" python)
else
    BREASTSIM_PY=("$BREASTSIM_ENV/bin/python")
fi

export BREASTSIM_HOST BREASTSIM_ROOT BREASTSIM_ENV BREASTSIM_BIG CONDA_BIN
