#!/bin/bash
#SBATCH --job-name=nik-polar-ablation
#SBATCH --gres=gpu:2g.24gb:1
#SBATCH --partition=gpu
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --time=0-07:00
#SBATCH --nice=10000
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -eu

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi || true

# micromamba setup
export PATH="/net/beegfs/users/P101440/micromamba/bin:$PATH"
export MAMBA_ROOT_PREFIX="/net/beegfs/users/P101440/micromamba"
eval "$(/net/beegfs/users/P101440/micromamba/bin/micromamba shell hook -s bash)"
micromamba activate torch29

which python
python --version

cd /net/beegfs/users/P101440/DCE_NIK

STEPS="${1:-20000}"
echo "Steps: $STEPS"

python run_polar_ablation.py --steps "$STEPS"
