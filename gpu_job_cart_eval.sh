#!/bin/bash
#SBATCH --job-name=nik-carteval
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --partition=luna-gpu-short
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
export PATH="/scratch/rnga/vvpshenov/micromamba/bin:$PATH"
export MAMBA_ROOT_PREFIX="/scratch/rnga/vvpshenov/micromamba"
eval "$(/scratch/rnga/vvpshenov/micromamba/bin/micromamba shell hook -s bash)"
micromamba activate torch29

which python
python --version

cd /scratch/rnga/vvpshenov/DCE_NIK

# parse args
CONFIG_PATH="${1:?Usage: sbatch gpu_job_cart_eval.sh CONFIG_PATH [sweep|SWEEP_ID] [COUNT]}"
MODE="${2:-single}"
COUNT="${3:-50}"

echo "Config: $CONFIG_PATH"
echo "Mode:   $MODE"

if [ "$MODE" = "single" ]; then
    python train_cart_eval.py "$CONFIG_PATH" --single
elif [ "$MODE" = "sweep" ]; then
    python train_cart_eval.py "$CONFIG_PATH" --count "$COUNT"
else
    # sweep id
    python train_cart_eval.py "$CONFIG_PATH" --sweep-id "$MODE" --count "$COUNT"
fi
