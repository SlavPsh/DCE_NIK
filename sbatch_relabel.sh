#!/bin/bash
#SBATCH -J relabel
#SBATCH -p luna-gpu-short
#SBATCH --gres gpu:1g.10gb:1
#SBATCH -c 4
#SBATCH --mem 24G
#SBATCH -t 2:00:00
#SBATCH -o /scratch/rnga/vvpshenov/DCE_NIK/relabel_%j.log
cd /scratch/rnga/vvpshenov/DCE_NIK
P=${PY:-/scratch/rnga/vvpshenov/micromamba/envs/torch29/bin/python}
echo "=== regenerate frontier figures with explicit labels ==="
$P -u xph_frontier.py    2>&1 | tail -3
$P -u invivo_frontier.py 2>&1 | tail -3
echo "=== rebuild v2 notebook ==="
$P -u build_gv2_nb.py 2>&1 | grep -vE "MissingIDField|validate" | tail -4
echo RELABEL_DONE
