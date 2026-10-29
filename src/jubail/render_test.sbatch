#!/bin/bash -l
# Tiny end-to-end render test of the pinned checkout: cache build, plan (devices, conspicuity, type shares),
# render, stats, contact sheet.   sbatch src/jubail/render_test.sbatch <phantom id> <out dir> [n frames]
#SBATCH -p preempt -A preempt -q preempt-gpu --gres=gpu:1 -c 8 --mem 48G -t 01:00:00
#SBATCH -J bsim-rtest -o /scratch/ll5582/BreastSIM_data/logs/%x-%j.out
source /scratch/ll5582/BreastSIM/src/jubail/common.sh
cd "$BSIM_ROOT" && git log --oneline -1
mkdir -p "$2" && echo "$1" > "$2/phantoms.txt"
nvidia-smi --query-gpu=name --format=csv,noheader
bsim_py -m breastsim.phantom "$1" \
 && bsim_py src/scripts/build_sim_dataset.py plan --out "$2" --phantoms "$2/phantoms.txt" --n "${3:-12}" --pos-frac 0.75 \
      --seed 3 --devices --min-conspicuity --type-shares --shares '{"victre/*": 1.0}' \
 && bsim_py src/scripts/build_sim_dataset.py render --out "$2" \
 && bsim_py src/scripts/build_sim_dataset.py stats --out "$2" \
 && bsim_py src/scripts/build_sim_dataset.py sheet --out "$2" --n 12
