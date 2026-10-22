#!/usr/bin/env bash
# Clone third-party source dependencies into lib/ (git-ignored).
set -euo pipefail
source "$(dirname "$0")/hostenv.sh"
LIB="$BREASTSIM_ROOT/lib"
mkdir -p "$LIB"

clone() {  # clone <url> <dir> [ref]
    local url=$1 dir=$LIB/$2 ref=${3:-}
    if [[ -d "$dir/.git" ]]; then
        echo "[lib] $2 exists, fetching"; git -C "$dir" fetch --quiet --tags
    else
        echo "[lib] cloning $2"; git clone --quiet "$url" "$dir"
    fi
    [[ -n "$ref" ]] && git -C "$dir" checkout --quiet "$ref"
    echo "[lib] $2 @ $(git -C "$dir" rev-parse --short HEAD)"
}

clone https://github.com/creatis-ULTIM/PyMUST.git PyMUST
clone https://github.com/DIDSR/breastPhantom.git breastPhantom
clone https://github.com/DIDSR/breastMass.git breastMass
clone https://github.com/DIDSR/VICTRE_PIPELINE.git VICTRE_PIPELINE
