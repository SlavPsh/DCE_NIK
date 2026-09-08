#!/bin/bash
#SBATCH -J sbk
#SBATCH -p luna-cpu-short
#SBATCH -c 8
#SBATCH --mem 16G
#SBATCH -t 2:00:00
#SBATCH -o /scratch/rnga/vvpshenov/DCE_NIK/sbk_%j.log
cd /scratch/rnga/vvpshenov/DCE_NIK
${PY:-/scratch/rnga/vvpshenov/micromamba/envs/torch29/bin/python} -u senseB_kernel_check.py
