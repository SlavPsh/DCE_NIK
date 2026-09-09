#!/bin/bash
#SBATCH --job-name=nik-spk2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-03:30
#SBATCH --output=/net/beegfs/users/P101440/DCE_NIK/logs/slurm-%x-%j.out
#SBATCH --error=/net/beegfs/users/P101440/DCE_NIK/logs/slurm-%x-%j.out
set -uo pipefail
LABEL="$1"; MODEL="$2"; TAG="$3"; shift 3            # LABEL=f100.. MODEL=wire_ff_res.. TAG=dir suffix
export PATH="/net/beegfs/users/P101440/micromamba/bin:$PATH"
export MAMBA_ROOT_PREFIX="/net/beegfs/users/P101440/micromamba"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd /net/beegfs/users/P101440/DCE_NIK
# apples-to-apples spoke sweep: IDENTICAL acquired spokes as CS/NUFFT (shared keep-file),
# trained on ALL of them (no heldout) so the input data matches exactly.
micromamba run -n torch29 python train_grasp_nik.py \
  --model "$MODEL" --slices 13 \
  --spoke-keep-file /net/beegfs/users/P101440/DCE_NIK/spoke_masks/keep_${LABEL}.npy \
  --save-dir /net/beegfs/users/P101440/DCE_NIK/results_spoke_${TAG}_${LABEL} \
  --resume --no-compile "$@"
echo "EXIT $?"
