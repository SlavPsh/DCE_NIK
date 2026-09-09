#!/bin/bash
#SBATCH --job-name=nik-grasp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2g.24gb:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-07:00
#SBATCH --nice=10000
#SBATCH --output=/net/beegfs/users/P101440/DCE_NIK/logs/slurm-%x-%j.out
#SBATCH --error=/net/beegfs/users/P101440/DCE_NIK/slurm-%x-%j.err

export PATH="/net/beegfs/users/P101440/micromamba/bin:$PATH"
export MAMBA_ROOT_PREFIX="/net/beegfs/users/P101440/micromamba"
eval "$(/net/beegfs/users/P101440/micromamba/bin/micromamba shell hook -s bash)"
micromamba activate torch29

nvidia-smi || true
cd /net/beegfs/users/P101440/DCE_NIK

python train_grasp_nik.py \
  --slices all \
  --steps 8000 \
  --save-dir /net/beegfs/users/P101440/grasp_pro_py/results_nik
echo "EXIT $?"
