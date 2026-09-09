#!/bin/bash
#SBATCH --job-name=k3adc-q
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:35:00
#SBATCH --output=/net/beegfs/users/P101440/DCE_NIK/results/xcat_physical_nomotion_nik_vs_grasp/slurm/k3adcq_%j.log
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba
export K3A_QUICK=1
cd /net/beegfs/users/P101440/DCE_NIK
${PY:-/net/beegfs/users/P101440/micromamba/envs/torch29/bin/python} -u k3a_dc.py
