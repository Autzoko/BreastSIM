#!/bin/bash -l
# Fine-tuning grid for one real dataset: init x training-set size x seed, sequential on one GPU
# (each run ~minutes; finished runs are skipped, so a preempted job resumes on requeue).
#   sbatch src/jubail/finetune_grid.sbatch BUSI <sim model scratch-init> <sim model imagenet-init>
#SBATCH -p preempt -A preempt -q preempt-gpu --gres=gpu:1 -c 8 --mem 48G -t 12:00:00 --requeue
#SBATCH -J bsim-ft -o /scratch/ll5582/BreastSIM_data/logs/%x-%j.out
source /scratch/ll5582/BreastSIM/src/jubail/common.sh
ds=$1; sim0=$2; simin=$3; out=$BREASTSIM_BIG/seg/ft
cd "$BSIM_ROOT"
for n in 32 128 all; do
  for seed in 0 1 2; do
    [[ $n == all && $seed != 0 ]] && continue
    for init in none imagenet sim0 simin; do
      case $init in sim0) arg=sim:$sim0;; simin) arg=sim:$simin;; *) arg=$init;; esac
      bsim_py src/train/seg.py finetune --real $BREASTSIM_BIG/realseg/$ds.h5 --init $arg --init-name $init \
          $([[ $n != all ]] && echo --n-train $n) --seed $seed --out $out/${ds}_n${n}_${init}_s$seed
    done
  done
done
