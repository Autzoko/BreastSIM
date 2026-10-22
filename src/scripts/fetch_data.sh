#!/usr/bin/env bash
# Run fetch_data.py at low CPU/IO priority, logging to logs/fetch_data.log.
# Usage: bash src/scripts/fetch_data.sh [--only a,b]
set -euo pipefail
source "$(dirname "$0")/hostenv.sh"
mkdir -p "$BREASTSIM_ROOT/logs" "$BREASTSIM_BIG"
NICE=(nice -n 19)
command -v ionice >/dev/null && NICE+=(ionice -c 3)
"${NICE[@]}" "${BREASTSIM_PY[@]}" -u "$BREASTSIM_ROOT/src/scripts/fetch_data.py" "$@" \
    2>&1 | tee -a "$BREASTSIM_ROOT/logs/fetch_data.log"
