#!/bin/bash
#SBATCH -J ktvsmoke
#SBATCH -p luna-gpu-short
#SBATCH --gres gpu:1g.10gb:1
#SBATCH -c 4
#SBATCH --mem 32G
#SBATCH -t 0:40:00
#SBATCH -o /scratch/rnga/vvpshenov/DCE_NIK/logs/ktvsmoke_%j.log
cd /scratch/rnga/vvpshenov/DCE_NIK
export PATH="/scratch/rnga/vvpshenov/micromamba/bin:$PATH"; export MAMBA_ROOT_PREFIX="/scratch/rnga/vvpshenov/micromamba"
micromamba run -n torch29 python train_grasp_nik.py \
  --slices 21 --model wire_ff_res --steps 50 --eval-every 25 --console-every 10 \
  --spoke-keep-file /scratch/rnga/vvpshenov/DCE_NIK/spoke_masks/keep_f80match.npy \
  --keep-heldout --no-compile --ktv21-weight 0.05 \
  --save-dir /scratch/rnga/vvpshenov/DCE_NIK/results_ktv_smoke
echo SMOKE_DONE
