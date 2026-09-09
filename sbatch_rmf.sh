#!/bin/bash
#SBATCH --job-name=rmf2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=00:25:00
#SBATCH --output=/net/beegfs/users/P101440/DCE_NIK/results/xcat_physical_nomotion_nik_vs_grasp/slurm/rmf2_%j.log
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba
cd /net/beegfs/users/P101440/DCE_NIK
${PY:-/net/beegfs/users/P101440/micromamba/envs/torch29/bin/python} -u rmf2.py
