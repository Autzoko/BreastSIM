#!/bin/bash -l
# Build preview caches (breastsim.phantom) for the given phantom ids, a few in parallel, on a CPU node.
#   sbatch src/jubail/cache.sbatch victre/ds_dense_les0_14 victre/ds_dense_les1_14 ...
#SBATCH -p compute -A ecomputer -c 32 --mem 120G -t 04:00:00
#SBATCH -J bsim-cache -o /scratch/ll5582/BreastSIM_data/logs/%x-%j.out
source /scratch/ll5582/BreastSIM/src/jubail/common.sh
export OMP_NUM_THREADS=4
printf "%s\n" "$@" | xargs -P 6 -I{} bash -c 'source /scratch/ll5582/BreastSIM/src/jubail/common.sh; bsim_py -m breastsim.phantom {}'
