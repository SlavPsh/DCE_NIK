#!/bin/bash
#SBATCH -J refreshboth
#SBATCH -p defq
#SBATCH -c 4
#SBATCH --mem 16G
#SBATCH -t 1:30:00
#SBATCH -o /net/beegfs/users/P101440/DCE_NIK/refreshboth_%j.log
cd /net/beegfs/users/P101440/DCE_NIK
P=${PY:-/net/beegfs/users/P101440/micromamba/envs/torch29/bin/python}
echo "=== refresh grasp-pro notebook (picks up backfilled aorta ttp/pk) ==="; $P -u refresh_nb.py || echo FAILED
echo "=== rebuild grasp-v2 notebook ==="; $P -u build_gv2_nb.py || echo FAILED
echo BOTH_REFRESHED
