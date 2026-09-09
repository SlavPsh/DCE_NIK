#!/bin/bash
#SBATCH -J rebuildgv2
#SBATCH -p defq
#SBATCH -c 4
#SBATCH --mem 16G
#SBATCH -t 1:00:00
#SBATCH -o /net/beegfs/users/P101440/DCE_NIK/rebuildgv2_%j.log
cd /net/beegfs/users/P101440/DCE_NIK
${PY:-/net/beegfs/users/P101440/micromamba/envs/torch29/bin/python} -u build_gv2_nb.py
echo REBUILD_DONE
