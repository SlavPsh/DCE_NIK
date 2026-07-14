#!/bin/bash
#SBATCH --job-name=nik-dcfpow-v2
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --partition=luna-gpu-short
#SBATCH --mem=48G
#SBATCH --cpus-per-task=1
#SBATCH --time=0-02:00
#SBATCH --array=0-3
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -eu
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
nvidia-smi || true

export PATH="/scratch/rnga/vvpshenov/micromamba/bin:$PATH"
export MAMBA_ROOT_PREFIX="/scratch/rnga/vvpshenov/micromamba"
eval "$(/scratch/rnga/vvpshenov/micromamba/bin/micromamba shell hook -s bash)"
micromamba activate torch29

cd /scratch/rnga/vvpshenov/DCE_NIK

# tight sweep around the sweet spot identified in the first dcf_power run
# (p0.5: best inner; p1.0: best outer). these include radial sharpness in the
# eval, which the previous round did not.
CONFIGS=(
    config/mct_dcfpow_p0p5.toml
    config/mct_dcfpow_p0p7.toml
    config/mct_dcfpow_p0p85.toml
    config/mct_dcfpow_p1p0.toml
)

CFG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "Running config: $CFG"

python train_multicoil_cart.py "$CFG" --single
