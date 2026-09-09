#!/bin/bash
#SBATCH --job-name=phitvw
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2g.24gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:30:00
#SBATCH --array=2-5
#SBATCH --output=/net/beegfs/users/P101440/DCE_NIK/logs/phitvw_%A_%a.log
cd /net/beegfs/users/P101440/DCE_NIK
export PATH="/net/beegfs/users/P101440/micromamba/bin:$PATH"
export MAMBA_ROOT_PREFIX="/net/beegfs/users/P101440/micromamba"
# temporal-TV weight sweep, same spirit as grasp v2's lam sweep: span decades, let the data pick.
W=(0.0 0.01 0.1 1.0 10.0 100.0)
w=${W[$SLURM_ARRAY_TASK_ID]}
micromamba run -n torch29 python train_grasp_nik.py \
  --slices 21 --model wire_ff_subspace --rank 16 \
  --spoke-keep-file /net/beegfs/users/P101440/DCE_NIK/spoke_masks/keep_f80match.npy \
  --keep-heldout --no-compile --phi-tv-weight $w \
  --save-dir /net/beegfs/users/P101440/DCE_NIK/results_phitv_w${w}
echo "DONE phitv w=$w"
