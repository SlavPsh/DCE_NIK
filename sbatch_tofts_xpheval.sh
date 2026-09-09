#!/bin/bash
#SBATCH -J txe
#SBATCH -p gpu
#SBATCH --gres=gpu:1g.12gb:1
#SBATCH -c 4
#SBATCH --mem 24G
#SBATCH -t 2:00:00
#SBATCH --array=0-5
#SBATCH -o /net/beegfs/users/P101440/DCE_NIK/results/tofts_vs_patlak/logs/xpheval_%A_%a.log
cd /net/beegfs/users/P101440/DCE_NIK; P=/net/beegfs/users/P101440/micromamba/envs/torch29/bin/python
i=$SLURM_ARRAY_TASK_ID; S=$((i % 3)); if [ $i -lt 3 ]; then export XPH_SIM=nomotion; else export XPH_SIM=motion; fi
echo "XPH_SIM=$XPH_SIM tag w768_ks2.5_s${S}_tofts16"; $P -u xph_eval.py --tag w768_ks2.5_s${S}_tofts16; echo XPHEVAL_TASK_DONE
