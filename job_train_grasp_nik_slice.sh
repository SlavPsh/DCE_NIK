#!/bin/bash
# single-slice WIRE-NIK run.  usage:  sbatch job_train_grasp_nik_slice.sh [SLICE] [STEPS]
#   SLICE defaults to 12, STEPS to 8000.
#SBATCH --job-name=nik-slice
#SBATCH --partition=luna-gpu-short
#SBATCH --gres=gpu:2g.20gb:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-01:00
#SBATCH --nice=10000
#SBATCH --output=/scratch/rnga/vvpshenov/DCE_NIK/slurm-%x-%j.out
#SBATCH --error=/scratch/rnga/vvpshenov/DCE_NIK/slurm-%x-%j.err

SLICE=${1:-13}
STEPS=${2:-8000}

export PATH="/scratch/rnga/vvpshenov/micromamba/bin:$PATH"
export MAMBA_ROOT_PREFIX="/scratch/rnga/vvpshenov/micromamba"
eval "$(/scratch/rnga/vvpshenov/micromamba/bin/micromamba shell hook -s bash)"
micromamba activate torch29

nvidia-smi || true
cd /scratch/rnga/vvpshenov/DCE_NIK

echo "slice=$SLICE steps=$STEPS dcf_power=0.5"
python train_grasp_nik.py \
  --slices "$SLICE" \
  --steps "$STEPS" \
  --save-dir /scratch/rnga/vvpshenov/grasp_pro_py/results_nik
echo "EXIT $?"
