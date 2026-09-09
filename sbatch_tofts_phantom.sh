#!/bin/bash
#SBATCH -J tph
#SBATCH -p gpu
#SBATCH --gres=gpu:1g.12gb:1
#SBATCH -c 4
#SBATCH --mem 24G
#SBATCH -t 8:00:00
#SBATCH --array=0-8
#SBATCH -o /net/beegfs/users/P101440/DCE_NIK/results/tofts_vs_patlak/logs/phantom_%A_%a.log
# 0-2: nomotion tofts s0-2 | 3-5: motion patlak s0-2 | 6-8: motion tofts s0-2   (nomotion patlak s0-2 = existing w768_ks2.5_s*)
cd /net/beegfs/users/P101440/DCE_NIK; P=/net/beegfs/users/P101440/micromamba/envs/torch29/bin/python
i=$SLURM_ARRAY_TASK_ID; S=$((i % 3))
if [ $i -lt 3 ]; then export XPH_SIM=nomotion; M=wire_ff_tofts; else export XPH_SIM=motion; if [ $i -lt 6 ]; then M=wire_ff_patlak; else M=wire_ff_tofts; fi; fi
TAG=w768_ks2.5_s$S; [ $M = wire_ff_tofts ] && TAG=${TAG}_tofts16
echo "XPH_SIM=$XPH_SIM model=$M seed=$S tag=$TAG"; date
$P -u xph_train.py --model $M --hidden-width 768 --k-sigma 2.5 --seed $S --steps 40000 --ckpt-every 2000 && \
$P -u xph_eval.py --tag $TAG
date; echo "PHANTOM_TASK_DONE $XPH_SIM $TAG"
