#!/bin/bash -l
# Render shards of a simulated dataset (src/scripts/build_sim_dataset.py) on a GPU node.
#   sbatch --array=0-7 src/jubail/render.sbatch <out dir> <n shards>
# Preemptible partition: the job is requeued on preemption and the shard resumes where it stopped.
#SBATCH -p preempt -A preempt -q preempt-gpu --gres=gpu:1 -c 8 --mem 48G -t 08:00:00 --requeue
#SBATCH -J bsim-render -o /scratch/ll5582/BreastSIM_data/logs/%x-%A_%a.out
source /scratch/ll5582/BreastSIM/src/jubail/common.sh
export OMP_NUM_THREADS=8
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
bsim_py "$BSIM_ROOT/src/scripts/build_sim_dataset.py" render --out "$1" --shard "$SLURM_ARRAY_TASK_ID" --shards "$2" ${3:+$3}
