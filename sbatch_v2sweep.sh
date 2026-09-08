#!/bin/bash
#SBATCH -J v2sweep
#SBATCH -p luna-gpu-short
#SBATCH --gres gpu:1g.10gb:1
#SBATCH -c 8
#SBATCH --mem 20G
#SBATCH -t 7:00:00
#SBATCH -o /scratch/rnga/vvpshenov/DCE_NIK/v2sweep_%A_%a.log
cd /scratch/rnga/vvpshenov/DCE_NIK
export OMP_NUM_THREADS=8
G=$(echo "1 2 3 5 8 12 20" | cut -d' ' -f$((SLURM_ARRAY_TASK_ID+1)))
${PY:-/scratch/rnga/vvpshenov/micromamba/envs/torch29/bin/python} -u xph_v2_sweep.py $G
