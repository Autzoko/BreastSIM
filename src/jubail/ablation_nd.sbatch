#!/bin/bash -l
# Device ablation: pretrain on the no-devices rendering of the v2 plan (simdata/v2nd), zero-shot evaluate,
# fine-tune n=32 x seeds 0-4 x 4 datasets for both sim inits. Resumable (finished runs are skipped).
#   sbatch --dependency=afterany:<render array> src/jubail/ablation_nd.sbatch
#SBATCH -p preempt -A preempt -q preempt-gpu --gres=gpu:1 -c 8 --mem 48G -t 10:00:00 --requeue
#SBATCH -J bsim-abl -o /scratch/ll5582/BreastSIM_data/logs/%x-%j.out
source /scratch/ll5582/BreastSIM/src/jubail/common.sh
cd "$BSIM_ROOT"
B=$BREASTSIM_BIG; R=$B/realseg_v2
[[ -e $B/simdata/sim256_v2nd.h5 ]] || bsim_py src/train/seg.py cache-sim --sim $B/simdata/v2nd --out $B/simdata/sim256_v2nd.h5
for i in none imagenet; do
  [[ -e $B/seg/v2nd_sim_$i/model.pt ]] || bsim_py src/train/seg.py pretrain --data $B/simdata/sim256_v2nd.h5 --init $i \
      --iters 20000 --real $R/BUSI.h5 $R/BUS-BRA.h5 $R/BrEaST.h5 $R/TDSC-ABUS.h5 --out $B/seg/v2nd_sim_$i
  bsim_py src/train/seg.py evaluate --model $B/seg/v2nd_sim_$i/model.pt --real $R/BUSI.h5 $R/BUS-BRA.h5 $R/BrEaST.h5 \
      $R/TDSC-ABUS.h5 --splits test,train --out $B/seg/zs_v2nd_sim_$i.json
done
for ds in BrEaST TDSC-ABUS BUSI BUS-BRA; do for seed in 0 1 2 3 4; do for i in none imagenet; do
  n=$([[ $i == none ]] && echo v2ndsim0 || echo v2ndsimin)
  bsim_py src/train/seg.py finetune --real $R/$ds.h5 --init sim:$B/seg/v2nd_sim_$i/model.pt --init-name $n \
      --n-train 32 --seed $seed --out $B/seg/ft2/${ds}_n32_${n}_s$seed
done; done; done
