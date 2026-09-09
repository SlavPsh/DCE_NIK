#!/bin/bash
#SBATCH --job-name=full-sl21-matched
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:30:00
#SBATCH --output=/net/beegfs/users/P101440/DCE_NIK/logs/full_sl21_matched_%j.log
cd /net/beegfs/users/P101440/DCE_NIK
export PATH="/net/beegfs/users/P101440/micromamba/bin:$PATH"
export MAMBA_ROOT_PREFIX="/net/beegfs/users/P101440/micromamba"
# SPOKE-MATCHED to grasp v2: identical acquired spokes (keep_f100, 1708), trained on ALL of them,
# NO heldout. the earlier results_spoke_full_slice21 used the default random 70/30 split, which gave
# grasp 43% more data in the in-vivo frontier. same model/recipe as that run otherwise (wire_ff_res).
micromamba run -n torch29 python train_grasp_nik.py \
  --slices 21 \
  --model wire_ff_res \
  --spoke-keep-file /net/beegfs/users/P101440/DCE_NIK/spoke_masks/keep_f100.npy \
  --no-compile \
  --save-dir /net/beegfs/users/P101440/DCE_NIK/results_full_sl21_matched
echo "TRAIN DONE slice21 spoke-matched"
