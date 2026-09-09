#!/bin/bash
#SBATCH --job-name=phitv
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:30:00
#SBATCH --array=0-3
#SBATCH --output=/net/beegfs/users/P101440/DCE_NIK/logs/phitv_%A_%a.log
cd /net/beegfs/users/P101440/DCE_NIK
export PATH="/net/beegfs/users/P101440/micromamba/bin:$PATH"
export MAMBA_ROOT_PREFIX="/net/beegfs/users/P101440/micromamba"
W=(0.0 0.01 0.05 0.2)
w=${W[$SLURM_ARRAY_TASK_ID]}
# subspace R16 (phi_tv only exists for subspace models), same 1368 spokes + heldout as the k80 run
micromamba run -n torch29 python train_grasp_nik.py \
  --slices 21 --model wire_ff_subspace --rank 16 \
  --spoke-keep-file /net/beegfs/users/P101440/DCE_NIK/spoke_masks/keep_f80match.npy \
  --keep-heldout --no-compile \
  --phi-tv-weight $w \
  --save-dir /net/beegfs/users/P101440/DCE_NIK/results_phitv_w${w}
echo "DONE phitv w=$w"
