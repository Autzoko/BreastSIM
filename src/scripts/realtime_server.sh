#!/usr/bin/env bash
# Start the real-time preview server on this host (localhost only).
# Usage: PORT=8765 bash src/scripts/realtime_server.sh
set -euo pipefail
source "$(dirname "$0")/hostenv.sh"
cd "$BREASTSIM_ROOT"
mkdir -p logs
exec "${BREASTSIM_PY[@]}" -u -m breastsim.realtime.server --port "${PORT:-8765}" "$@" 2>&1 | tee -a logs/realtime.log
