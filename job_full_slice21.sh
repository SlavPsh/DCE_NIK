#!/bin/bash
#SBATCH --job-name=full-sl21
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/net/beegfs/users/P101440/tmp/full_slice21_%j.log
cd /net/beegfs/users/P101440/DCE_NIK
eval "$(micromamba shell hook --shell bash)"
# full-rank wire_ff_res, fixed recipe (all defaults match protocol), slice 21, all spokes (=CS f100)
micromamba run -n torch29 python train_grasp_nik.py \
  --slices 21 \
  --model wire_ff_res \
  --save-dir /net/beegfs/users/P101440/DCE_NIK/results_spoke_full_slice21
echo "TRAIN DONE slice21 full-rank"
