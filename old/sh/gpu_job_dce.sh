#!/bin/bash
#SBATCH --job-name=dce-gpu
#SBATCH --gres=gpu:2g.24gb:1
#SBATCH --partition=gpu
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-07:00
#SBATCH --nice=10000
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi || true

# micromamba env
export PATH="/net/beegfs/users/P101440/micromamba/bin:$PATH"
export MAMBA_ROOT_PREFIX="/net/beegfs/users/P101440/micromamba"

eval "$(/net/beegfs/users/P101440/micromamba/bin/micromamba shell hook -s bash)"
micromamba activate ml

which python
python --version

cd /net/beegfs/users/P101440

# cpu fallback for oom
python DCE_NIK_try.py \
  --file /net/beegfs/users/P101440/XCAT-ERIC/results/simulation_results_20260109T221333.mat \
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
