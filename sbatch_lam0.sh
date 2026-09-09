#!/bin/bash
#SBATCH -J lam0
#SBATCH -p gpu
#SBATCH --gres gpu:1g.12gb:1
#SBATCH -c 8
#SBATCH --mem 20G
#SBATCH -t 3:00:00
#SBATCH -o /net/beegfs/users/P101440/DCE_NIK/lam0_%j.log
cd /net/beegfs/users/P101440/DCE_NIK
export OMP_NUM_THREADS=8 LAM_FRAC=0.0
${PY:-/net/beegfs/users/P101440/micromamba/envs/torch29/bin/python} -u xph_v2_sweep.py 8
echo "--- objective split across lam ---"
${PY:-/net/beegfs/users/P101440/micromamba/envs/torch29/bin/python} -u xph_lam_meaning.py
