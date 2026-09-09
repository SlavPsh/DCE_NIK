#!/bin/bash
#SBATCH --job-name=ocrsw
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/net/beegfs/users/P101440/DCE_NIK/results/realdata_nik_vs_cs_figures/ocrsw_%j.log
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba
cd /net/beegfs/users/P101440/DCE_NIK
${PY:-/net/beegfs/users/P101440/micromamba/envs/torch29/bin/python} -u outcoil_real.py --coilmode output --model subspace --steps ${STEPS:-20000} --rank ${RANK:-16} --phi_w0 ${PHIW0:-30} --t_sigma ${TSIG:-0} --act ${ACT:-siren} --gauss_s ${GAUSS_S:-3.0} --t_enc ${TENC:-ff} --tag ${TAG:-_sw}
