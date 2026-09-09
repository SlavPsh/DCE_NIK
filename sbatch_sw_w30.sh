#!/bin/bash
#SBATCH --job-name=swAw30
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=/net/beegfs/users/P101440/DCE_NIK/results/xcat_physical_nomotion_nik_vs_grasp/slurm/swA_w30_%j.log
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba
cd /net/beegfs/users/P101440/DCE_NIK
${PY:-/net/beegfs/users/P101440/micromamba/envs/torch29/bin/python} -u sense_forward_train.py --steps 2500 --fbatch 6 --rank 16 --seed 0 --smoke 1 --tag _w30 --w0 30 --s0 12 --imsig 5
