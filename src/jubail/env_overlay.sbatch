#!/bin/bash -l
# Build the breastsim Python env inside a Singularity ext3 overlay on JUBAIL (one Lustre file instead of
# ~50k: /scratch has a 500k file-count quota). Follows the NYUAD HPC docs "Conda with Overlay" /
# "Singularity Overlays" (create-overlay, singularity --overlay, base container from
# /share/apps/admin/singularity-images). Run once:  sbatch src/jubail/env_overlay.sbatch
#SBATCH -p compute -A ecomputer -c 8 --mem 16G -t 02:00:00
#SBATCH -J bsim-env -o /scratch/ll5582/BreastSIM_data/logs/%x-%j.out
set -euo pipefail
source /scratch/ll5582/BreastSIM/src/jubail/common.sh
mkdir -p "$(dirname "$BSIM_OVERLAY")"
cd "$(dirname "$BSIM_OVERLAY")"
[[ -e "$BSIM_OVERLAY" ]] || create-overlay -s 20000 -n 400 -o "$(basename "$BSIM_OVERLAY")"
MF=$BREASTSIM_BIG/env/Miniforge3-24.11.3-0-Linux-x86_64.sh   # the base container has no curl
[[ -s $MF ]] || curl -sL -o "$MF" https://github.com/conda-forge/miniforge/releases/download/24.11.3-0/Miniforge3-24.11.3-0-Linux-x86_64.sh
singularity exec -B /scratch --overlay "$BSIM_OVERLAY" "$BSIM_SIF" /bin/bash -c '
set -euo pipefail
if [[ ! -x /ext3/miniforge3/bin/python ]]; then
    bash '"$MF"' -b -p /ext3/miniforge3
fi
printf "export PATH=/ext3/miniforge3/bin:\$PATH\nexport PYTHONNOUSERSITE=1\n" > /ext3/env.sh
source /ext3/env.sh
mamba install -y -q python=3.11 numpy=2.2 scipy h5py scikit-image pandas openpyxl pillow
pip install -q --no-cache-dir torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -q --no-cache-dir segmentation-models-pytorch==0.4.0 timm
conda clean -afy -q
python -c "import torch, segmentation_models_pytorch, h5py, scipy; print(\"ok torch\", torch.__version__)"
'
