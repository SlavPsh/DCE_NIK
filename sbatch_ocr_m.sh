#!/bin/bash
#SBATCH --job-name=ocrealm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/net/beegfs/users/P101440/DCE_NIK/results/realdata_nik_vs_cs_figures/ocreal_%j.log
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba
cd /net/beegfs/users/P101440/DCE_NIK
${PY:-/net/beegfs/users/P101440/micromamba/envs/torch29/bin/python} -u outcoil_real.py --coilmode ${MODE:-output} --model ${MODELT:-subspace} --steps ${STEPS:-30000}
