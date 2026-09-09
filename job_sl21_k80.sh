#!/bin/bash
#SBATCH --job-name=sl21-k80
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:30:00
#SBATCH --output=/net/beegfs/users/P101440/DCE_NIK/logs/sl21_k80_%j.log
cd /net/beegfs/users/P101440/DCE_NIK
export PATH="/net/beegfs/users/P101440/micromamba/bin:$PATH"
export MAMBA_ROOT_PREFIX="/net/beegfs/users/P101440/micromamba"
# train on v%10<8 (1368 spokes, same as grasp), early-stop on the 342 non-kept spokes
micromamba run -n torch29 python train_grasp_nik.py \
  --slices 21 --model wire_ff_res \
  --spoke-keep-file /net/beegfs/users/P101440/DCE_NIK/spoke_masks/keep_f80match.npy \
  --keep-heldout --no-compile \
  --save-dir /net/beegfs/users/P101440/DCE_NIK/results_sl21_k80
echo "TRAIN DONE sl21 k80 with heldout"
