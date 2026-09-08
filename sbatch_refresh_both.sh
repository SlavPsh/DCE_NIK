#!/bin/bash
#SBATCH -J refreshboth
#SBATCH -p luna-cpu-tiny
#SBATCH -c 4
#SBATCH --mem 16G
#SBATCH -t 1:30:00
#SBATCH -o /scratch/rnga/vvpshenov/DCE_NIK/refreshboth_%j.log
cd /scratch/rnga/vvpshenov/DCE_NIK
P=${PY:-/scratch/rnga/vvpshenov/micromamba/envs/torch29/bin/python}
echo "=== refresh grasp-pro notebook (picks up backfilled aorta ttp/pk) ==="; $P -u refresh_nb.py || echo FAILED
echo "=== rebuild grasp-v2 notebook ==="; $P -u build_gv2_nb.py || echo FAILED
echo BOTH_REFRESHED
