#!/bin/bash
#SBATCH -J v2lam2
#SBATCH -p luna-gpu-short
#SBATCH --gres gpu:1g.10gb:1
#SBATCH -c 8
#SBATCH --mem 20G
#SBATCH -t 4:00:00
#SBATCH -a 0-3
#SBATCH -o /scratch/rnga/vvpshenov/DCE_NIK/v2lam2_%A_%a.log
cd /scratch/rnga/vvpshenov/DCE_NIK
export OMP_NUM_THREADS=8
GS=(40 40 25 60); LS=(0.005 0.010 0.010 0.005)
G=$((${GS[$SLURM_ARRAY_TASK_ID]} / 5))
export LAM_FRAC=${LS[$SLURM_ARRAY_TASK_ID]}
echo "spokes/frame=${GS[$SLURM_ARRAY_TASK_ID]} (G=$G) LAM_FRAC=$LAM_FRAC"
${PY:-/scratch/rnga/vvpshenov/micromamba/envs/torch29/bin/python} -u xph_v2_sweep.py $G
