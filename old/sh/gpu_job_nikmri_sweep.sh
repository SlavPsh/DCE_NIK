#!/bin/bash
#SBATCH --job-name=nikmri-sweep
#SBATCH --gres=gpu:2g.24gb:1
#SBATCH --partition=gpu
#SBATCH --mem=24G
#SBATCH --cpus-per-task=1
#SBATCH --time=0-07:00
#SBATCH --nice=10000
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -eu

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi || true

export PATH="/net/beegfs/users/P101440/micromamba/bin:$PATH"
export MAMBA_ROOT_PREFIX="/net/beegfs/users/P101440/micromamba"
eval "$(/net/beegfs/users/P101440/micromamba/bin/micromamba shell hook -s bash)"
micromamba activate torch29

export MPLCONFIGDIR="/home/P101440/tmp/mpl"
mkdir -p "$MPLCONFIGDIR"

which python
python --version

cd /net/beegfs/users/P101440/DCE_NIK

# usage
CONFIG_PATH="${1:?Usage: sbatch gpu_job_nikmri_sweep.sh CONFIG_PATH [SWEEP_ID] [COUNT]}"
SWEEP_ID="${2:-}"
COUNT="${3:-50}"

echo "Config:    $CONFIG_PATH"
echo "Sweep ID:  ${SWEEP_ID:-<new sweep>}"
echo "Run count: $COUNT"

if [ -n "$SWEEP_ID" ]; then
    python train_nik_mri_style.py "$CONFIG_PATH" --sweep-id "$SWEEP_ID" --count "$COUNT"
else
    python train_nik_mri_style.py "$CONFIG_PATH" --count "$COUNT"
fi
