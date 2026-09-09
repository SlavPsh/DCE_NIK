#!/bin/bash
#SBATCH -J xphv2
#SBATCH -p defq
#SBATCH -c 8
#SBATCH --mem 24G
#SBATCH -t 4:00:00
#SBATCH -o /net/beegfs/users/P101440/DCE_NIK/xphv2_%j.log
cd /net/beegfs/users/P101440/DCE_NIK
export OMP_NUM_THREADS=8
${PY:-/net/beegfs/users/P101440/micromamba/envs/torch29/bin/python} -u xph_grasp_v2.py
