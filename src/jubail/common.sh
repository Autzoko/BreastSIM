# Shared settings for the JUBAIL Slurm scripts (source this). Python runs from a conda env inside an
# ext3 overlay, through Singularity (NYUAD HPC docs: "Conda with Overlay"); read-only so many jobs share it.
export BSIM_ROOT=/scratch/ll5582/BreastSIM
export BREASTSIM_BIG=/scratch/ll5582/BreastSIM_data
export BSIM_OVERLAY=$BREASTSIM_BIG/env/breastsim-20G.ext3
export BSIM_SIF=/share/apps/admin/singularity-images/cuda_12.6.2-cudnn-devel-ubuntu22.04.sif
# /home is full: every cache (pip, torch hub weights, matplotlib) goes to /scratch
export PIP_CACHE_DIR=$BREASTSIM_BIG/cache/pip XDG_CACHE_HOME=$BREASTSIM_BIG/cache/xdg TORCH_HOME=$BREASTSIM_BIG/cache/torch
export MPLCONFIGDIR=$BREASTSIM_BIG/cache/mpl SINGULARITYENV_TORCH_HOME=$BREASTSIM_BIG/cache/torch
module load singularity/4.2.0 2>/dev/null || true
# bsim_py <args>: python of the overlay env with the repo on PYTHONPATH (GPU passed through with --nv)
bsim_py() {
    singularity exec --nv --overlay "$BSIM_OVERLAY:ro" -B /scratch "$BSIM_SIF" /bin/bash -c \
        'source /ext3/env.sh; export PYTHONPATH='"$BSIM_ROOT"'/src BREASTSIM_BIG='"$BREASTSIM_BIG"'; python "$@"' bsim "$@"
}
