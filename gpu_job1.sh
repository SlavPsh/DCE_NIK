#!/bin/bash
#
# slurm specific parameters should be defined as comment line starting with #SBATCH
#SBATCH --job-name=py-gpu
#SBATCH --gres=gpu:1g.10gb:1          # or gpu:1g.10gb:1, gpu:2g.20gb:1, gpu:4g.40gb:1
#SBATCH --partition=luna-gpu-short # queue/partition
#SBATCH --mem=32G                  # adjust as needed
#SBATCH --cpus-per-task=4          # adjust as needed
#SBATCH --time=0-07:00             # (DD-HH:MM)
#SBATCH --nice=10000               # allow other priority jobs to go first
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -eu

# Print which GPU device(s) SLURM assigned
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi || true

# ---- Micromamba setup ----
export PATH="/scratch/rnga/vvpshenov/micromamba/bin:$PATH"
export MAMBA_ROOT_PREFIX="/scratch/rnga/vvpshenov/micromamba"

# Make micromamba activation work in non-interactive shells
eval "$(/scratch/rnga/vvpshenov/micromamba/bin/micromamba shell hook -s bash)"
micromamba activate ml

# keep pip cache off $HOME
#export PIP_CACHE_DIR="/scratch/rnga/vvpshenov/pip_cache"

which python
python --version

cd /scratch/rnga/vvpshenov/IVIMNET

python Example_1_simple_map.py


