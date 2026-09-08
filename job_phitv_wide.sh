#!/bin/bash
#SBATCH --job-name=phitvw
#SBATCH --partition=luna-gpu-short
#SBATCH --gres=gpu:2g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:30:00
#SBATCH --array=2-5
#SBATCH --output=/scratch/rnga/vvpshenov/DCE_NIK/logs/phitvw_%A_%a.log
cd /scratch/rnga/vvpshenov/DCE_NIK
export PATH="/scratch/rnga/vvpshenov/micromamba/bin:$PATH"
export MAMBA_ROOT_PREFIX="/scratch/rnga/vvpshenov/micromamba"
# temporal-TV weight sweep, same spirit as grasp v2's lam sweep: span decades, let the data pick.
W=(0.0 0.01 0.1 1.0 10.0 100.0)
w=${W[$SLURM_ARRAY_TASK_ID]}
micromamba run -n torch29 python train_grasp_nik.py \
  --slices 21 --model wire_ff_subspace --rank 16 \
  --spoke-keep-file /scratch/rnga/vvpshenov/DCE_NIK/spoke_masks/keep_f80match.npy \
  --keep-heldout --no-compile --phi-tv-weight $w \
  --save-dir /scratch/rnga/vvpshenov/DCE_NIK/results_phitv_w${w}
echo "DONE phitv w=$w"
