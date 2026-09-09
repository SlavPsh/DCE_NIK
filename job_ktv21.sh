#!/bin/bash
#SBATCH --job-name=ktv21
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2g.24gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=07:00:00
#SBATCH --array=0-3
#SBATCH --output=/net/beegfs/users/P101440/DCE_NIK/logs/ktv21_%A_%a.log
cd /net/beegfs/users/P101440/DCE_NIK
export PATH="/net/beegfs/users/P101440/micromamba/bin:$PATH"
export MAMBA_ROOT_PREFIX="/net/beegfs/users/P101440/micromamba"
W=(0.0 0.01 0.1 1.0)
w=${W[$SLURM_ARRAY_TASK_ID]}
# l2,1 temporal prior on the k-space output. full-rank wire_ff_res (phi_tv cannot apply here).
# n_k reduced 1024 -> 256 after the OOM; n_t 64 kept.
micromamba run -n torch29 python train_grasp_nik.py \
  --slices 21 --model wire_ff_res \
  --spoke-keep-file /net/beegfs/users/P101440/DCE_NIK/spoke_masks/keep_f80match.npy \
  --keep-heldout --no-compile \
  --ktv21-weight $w --ktv21-nk 256 --ktv21-nt 64 --resume \
  --save-dir /net/beegfs/users/P101440/DCE_NIK/results_ktv21_w${w}
echo "DONE ktv21 w=$w"
