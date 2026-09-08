#!/bin/bash
#SBATCH -J v2lam
#SBATCH -p luna-gpu-short
#SBATCH --gres gpu:1g.10gb:1
#SBATCH -c 8
#SBATCH --mem 20G
#SBATCH -t 7:00:00
#SBATCH -a 0-14
#SBATCH -o /scratch/rnga/vvpshenov/DCE_NIK/v2lam_%A_%a.log
cd /scratch/rnga/vvpshenov/DCE_NIK
export OMP_NUM_THREADS=8
GS=(5 8 12); LS=(0.02 0.05 0.10 0.25 0.50)
G=${GS[$((SLURM_ARRAY_TASK_ID / 5))]}
export LAM_FRAC=${LS[$((SLURM_ARRAY_TASK_ID % 5))]}
echo "G=$G spokes/frame=$((5*G))  LAM_FRAC=$LAM_FRAC"
${PY:-/scratch/rnga/vvpshenov/micromamba/envs/torch29/bin/python} -u xph_v2_sweep.py $G
