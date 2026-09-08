#!/bin/bash
#SBATCH --job-name=sl21-k80
#SBATCH --partition=luna-gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:30:00
#SBATCH --output=/scratch/rnga/vvpshenov/DCE_NIK/logs/sl21_k80_%j.log
cd /scratch/rnga/vvpshenov/DCE_NIK
export PATH="/scratch/rnga/vvpshenov/micromamba/bin:$PATH"
export MAMBA_ROOT_PREFIX="/scratch/rnga/vvpshenov/micromamba"
# train on v%10<8 (1368 spokes, same as grasp), early-stop on the 342 non-kept spokes
micromamba run -n torch29 python train_grasp_nik.py \
  --slices 21 --model wire_ff_res \
  --spoke-keep-file /scratch/rnga/vvpshenov/DCE_NIK/spoke_masks/keep_f80match.npy \
  --keep-heldout --no-compile \
  --save-dir /scratch/rnga/vvpshenov/DCE_NIK/results_sl21_k80
echo "TRAIN DONE sl21 k80 with heldout"
