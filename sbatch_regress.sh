#!/bin/bash
#SBATCH -J regress
#SBATCH -p defq
#SBATCH -c 4
#SBATCH --mem 16G
#SBATCH -t 1:00:00
#SBATCH -o /net/beegfs/users/P101440/DCE_NIK/regress_%j.log
cd /net/beegfs/users/P101440/DCE_NIK
P=${PY:-/net/beegfs/users/P101440/micromamba/envs/torch29/bin/python}
echo "--- step7_fig (defaults) ---"; $P -u step7_fig.py
echo "--- calib_curves (defaults) ---"; $P -u calib_curves.py
echo "--- step1_images (defaults) ---"; $P -u step1_images.py
echo DONE_REGRESS
