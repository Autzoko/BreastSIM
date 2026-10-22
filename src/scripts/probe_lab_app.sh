#!/usr/bin/env bash
# Native Probe Lab window on the Mac (no browser); rendering on the GPU host over SSH.
# Usage: bash src/scripts/probe_lab_app.sh [--restart] [--host ALIAS]
set -euo pipefail
source "$(dirname "$0")/hostenv.sh"
exec "${BREASTSIM_PY[@]}" "$BREASTSIM_ROOT/src/scripts/probe_lab_app.py" "$@"
