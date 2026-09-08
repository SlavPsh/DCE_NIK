#!/bin/bash
#SBATCH --job-name=phitv
#SBATCH --partition=luna-gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:30:00
#SBATCH --array=0-3
#SBATCH --output=/scratch/rnga/vvpshenov/DCE_NIK/logs/phitv_%A_%a.log
cd /scratch/rnga/vvpshenov/DCE_NIK
export PATH="/scratch/rnga/vvpshenov/micromamba/bin:$PATH"
export MAMBA_ROOT_PREFIX="/scratch/rnga/vvpshenov/micromamba"
W=(0.0 0.01 0.05 0.2)
w=${W[$SLURM_ARRAY_TASK_ID]}
# subspace R16 (phi_tv only exists for subspace models), same 1368 spokes + heldout as the k80 run
micromamba run -n torch29 python train_grasp_nik.py \
  --slices 21 --model wire_ff_subspace --rank 16 \
  --spoke-keep-file /scratch/rnga/vvpshenov/DCE_NIK/spoke_masks/keep_f80match.npy \
  --keep-heldout --no-compile \
  --phi-tv-weight $w \
  --save-dir /scratch/rnga/vvpshenov/DCE_NIK/results_phitv_w${w}
echo "DONE phitv w=$w"
