#!/bin/bash -l
# Generic GPU job: python <args> in the overlay env.   sbatch src/jubail/gpu.sbatch src/train/seg.py pretrain ...
#SBATCH -p preempt -A preempt -q preempt-gpu --gres=gpu:1 -c 8 --mem 48G -t 12:00:00 --requeue
#SBATCH -J bsim-gpu -o /scratch/ll5582/BreastSIM_data/logs/%x-%j.out
source /scratch/ll5582/BreastSIM/src/jubail/common.sh
export OMP_NUM_THREADS=8
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
cd "$BSIM_ROOT" && bsim_py "$@"
