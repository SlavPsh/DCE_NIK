#!/bin/bash
#SBATCH --job-name=swcmp
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=/net/beegfs/users/P101440/DCE_NIK/results/realdata_nik_vs_cs_figures/swcmp_%j.log
cd /net/beegfs/users/P101440/DCE_NIK
${PY:-/net/beegfs/users/P101440/micromamba/envs/torch29/bin/python} -u step3_fig.py
