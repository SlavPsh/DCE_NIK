#!/bin/bash
#SBATCH -J v2spf002
#SBATCH -p gpu
#SBATCH --gres gpu:1g.12gb:1
#SBATCH -c 8
#SBATCH --mem 20G
#SBATCH -t 7:00:00
#SBATCH -a 0-3
#SBATCH -o /net/beegfs/users/P101440/DCE_NIK/v2spf002_%A_%a.log
cd /net/beegfs/users/P101440/DCE_NIK
export OMP_NUM_THREADS=8 LAM_FRAC=0.02
GS=(20 3 2 1)          # 100, 15, 10, 5 spokes/frame. coarse first (fast), G=1 is the 3.3h long pole
G=${GS[$SLURM_ARRAY_TASK_ID]}
echo "G=$G -> $((5*G)) spokes/frame, lam 0.02"
${PY:-/net/beegfs/users/P101440/micromamba/envs/torch29/bin/python} -u xph_v2_sweep.py $G
