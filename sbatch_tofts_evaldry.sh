#!/bin/bash
#SBATCH -J tevaldry
#SBATCH -p gpu
#SBATCH --gres=gpu:1g.12gb:1
#SBATCH -c 4
#SBATCH --mem 32G
#SBATCH -t 1:00:00
#SBATCH -o /net/beegfs/users/P101440/DCE_NIK/results/tofts_vs_patlak/logs/evaldry_%j.log
cd /net/beegfs/users/P101440/DCE_NIK; P=/net/beegfs/users/P101440/micromamba/envs/torch29/bin/python
$P -u tofts_eval_phantom.py --sim nomotion; $P -u tofts_eval_invivo.py; echo DRY_DONE
