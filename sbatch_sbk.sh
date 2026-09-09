#!/bin/bash
#SBATCH -J sbk
#SBATCH -p defq
#SBATCH -c 8
#SBATCH --mem 16G
#SBATCH -t 2:00:00
#SBATCH -o /net/beegfs/users/P101440/DCE_NIK/sbk_%j.log
cd /net/beegfs/users/P101440/DCE_NIK
${PY:-/net/beegfs/users/P101440/micromamba/envs/torch29/bin/python} -u senseB_kernel_check.py
