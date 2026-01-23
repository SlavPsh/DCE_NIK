#!/bin/bash
#SBATCH --job-name=dce-gpu
#SBATCH --gres=gpu:2g.20gb:1
#SBATCH --partition=luna-gpu-short
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-07:00
#SBATCH --nice=10000
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi || true

# Micromamba env
export PATH="/scratch/rnga/vvpshenov/micromamba/bin:$PATH"
export MAMBA_ROOT_PREFIX="/scratch/rnga/vvpshenov/micromamba"

eval "$(/scratch/rnga/vvpshenov/micromamba/bin/micromamba shell hook -s bash)"
micromamba activate ml

which python
python --version

cd /scratch/rnga/vvpshenov

# Adjust --data-device to 'cpu' if GPU OOM occurs
python DCE_NIK_try.py \
  --file /scratch/rnga/vvpshenov/XCAT-ERIC/results/simulation_results_20260109T221333.mat \
  --steps 20000 \
  --batch-size 131072 \
  --lr 1e-3 \
  --log-every 10 \
  --grad-clip 1.0 \
  --seed 42 \
  --amp \
  --compile \
  --data-device cuda \
  --device cuda
