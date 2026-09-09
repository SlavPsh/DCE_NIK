#!/bin/bash
#
# slurm directives below
#SBATCH --job-name=py-gpu
#SBATCH --gres=gpu:1g.12gb:1
#SBATCH --partition=gpu
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-07:00
#SBATCH --nice=10000
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -eu

# gpu device
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi || true

# micromamba setup
export PATH="/net/beegfs/users/P101440/micromamba/bin:$PATH"
export MAMBA_ROOT_PREFIX="/net/beegfs/users/P101440/micromamba"

# non interactive activate
eval "$(/net/beegfs/users/P101440/micromamba/bin/micromamba shell hook -s bash)"
micromamba activate ml

# pip cache off home
#export PIP_CACHE_DIR="/net/beegfs/users/P101440/pip_cache"

which python
python --version

cd /net/beegfs/users/P101440/IVIMNET

python Example_1_simple_map.py


