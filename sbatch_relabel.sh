#!/bin/bash
#SBATCH -J relabel
#SBATCH -p gpu
#SBATCH --gres gpu:1g.12gb:1
#SBATCH -c 4
#SBATCH --mem 24G
#SBATCH -t 2:00:00
#SBATCH -o /net/beegfs/users/P101440/DCE_NIK/relabel_%j.log
cd /net/beegfs/users/P101440/DCE_NIK
P=${PY:-/net/beegfs/users/P101440/micromamba/envs/torch29/bin/python}
echo "=== regenerate frontier figures with explicit labels ==="
$P -u xph_frontier.py    2>&1 | tail -3
$P -u invivo_frontier.py 2>&1 | tail -3
echo "=== rebuild v2 notebook ==="
$P -u build_gv2_nb.py 2>&1 | grep -vE "MissingIDField|validate" | tail -4
echo RELABEL_DONE
