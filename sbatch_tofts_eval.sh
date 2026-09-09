#!/bin/bash
#SBATCH -J teval
#SBATCH -p gpu
#SBATCH --gres=gpu:1g.12gb:1
#SBATCH -c 4
#SBATCH --mem 32G
#SBATCH -t 3:00:00
#SBATCH -o /net/beegfs/users/P101440/DCE_NIK/results/tofts_vs_patlak/logs/eval_%j.log
cd /net/beegfs/users/P101440/DCE_NIK; P=/net/beegfs/users/P101440/micromamba/envs/torch29/bin/python
echo "### phantom nomotion"; $P -u tofts_eval_phantom.py --sim nomotion
echo "### phantom motion"; $P -u tofts_eval_phantom.py --sim motion
echo "### in vivo"; $P -u tofts_eval_invivo.py
echo EVAL_ALL_DONE
