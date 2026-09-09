#!/bin/bash
#SBATCH -J tiv
#SBATCH -p gpu
#SBATCH --gres=gpu:2g.24gb:1
#SBATCH -c 8
#SBATCH --mem 48G
#SBATCH -t 14:00:00
#SBATCH --array=0-17
#SBATCH -o /net/beegfs/users/P101440/DCE_NIK/results/tofts_vs_patlak/logs/invivo_%A_%a.log
# i = slice_idx*6 + model_idx*3 + seed ; slices 18,19,21 ; models patlak,tofts ; seeds 0,1,2
cd /net/beegfs/users/P101440/DCE_NIK; P=/net/beegfs/users/P101440/micromamba/envs/torch29/bin/python
i=$SLURM_ARRAY_TASK_ID; SL=(18 19 21); Z=${SL[$((i / 6))]}; MI=$(((i % 6) / 3)); S=$((i % 3))
if [ $MI = 0 ]; then M=wire_ff_patlak; MA="--aif-file /net/beegfs/users/P101440/DCE_NIK/aif_slice$Z.npz --patlak-free 0"; NM=patlak
else M=wire_ff_tofts; MA="--tofts-basis /net/beegfs/users/P101440/DCE_NIK/results/tofts_vs_patlak/basis_sl$Z.npz"; NM=tofts; fi
OUT=/net/beegfs/users/P101440/DCE_NIK/results/tofts_vs_patlak/invivo/${NM}_sl${Z}_s$S; mkdir -p $OUT
echo "slice $Z model $M seed $S -> $OUT"; date
$P -u train_grasp_nik.py --model $M $MA --slices $Z --seed $S --ff-seed $S \
  --spoke-keep-file /net/beegfs/users/P101440/DCE_NIK/spoke_masks/keep_f25.npy \
  --spoke-heldout-file /net/beegfs/users/P101440/DCE_NIK/spoke_masks/val_f25c_m8.npy \
  --no-compile --resume --save-dir $OUT
date; echo "INVIVO_TASK_DONE $NM sl$Z s$S"
