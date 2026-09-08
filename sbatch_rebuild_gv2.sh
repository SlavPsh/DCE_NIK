#!/bin/bash
#SBATCH -J rebuildgv2
#SBATCH -p luna-cpu-tiny
#SBATCH -c 4
#SBATCH --mem 16G
#SBATCH -t 1:00:00
#SBATCH -o /scratch/rnga/vvpshenov/DCE_NIK/rebuildgv2_%j.log
cd /scratch/rnga/vvpshenov/DCE_NIK
${PY:-/scratch/rnga/vvpshenov/micromamba/envs/torch29/bin/python} -u build_gv2_nb.py
echo REBUILD_DONE
