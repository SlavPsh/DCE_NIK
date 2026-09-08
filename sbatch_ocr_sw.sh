#!/bin/bash
#SBATCH --job-name=ocrsw
#SBATCH --partition=luna-gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/rnga/vvpshenov/DCE_NIK/results/realdata_nik_vs_cs_figures/ocrsw_%j.log
export MAMBA_ROOT_PREFIX=/scratch/rnga/vvpshenov/micromamba
cd /scratch/rnga/vvpshenov/DCE_NIK
${PY:-/scratch/rnga/vvpshenov/micromamba/envs/torch29/bin/python} -u outcoil_real.py --coilmode output --model subspace --steps ${STEPS:-20000} --rank ${RANK:-16} --phi_w0 ${PHIW0:-30} --t_sigma ${TSIG:-0} --act ${ACT:-siren} --gauss_s ${GAUSS_S:-3.0} --t_enc ${TENC:-ff} --tag ${TAG:-_sw}
