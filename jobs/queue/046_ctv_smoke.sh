#!/bin/bash
#SBATCH -J ctvsmoke
#SBATCH -p defq
#SBATCH -c 8
#SBATCH --mem 32G
#SBATCH -t 0:40:00
# cpu smoke test of the new trainer paths before the gpu runs (045): cosine schedule + coefficient-map tv for 8 steps on slice 21, no wandb
cd /net/beegfs/users/P101440/DCE_NIK
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH OMP_NUM_THREADS=8
OUT=results/tofts_vs_patlak/amp_track/smoke_ctv_sl21; mkdir -p $OUT
micromamba run -n torch29 python -u train_grasp_nik.py --model wire_ff_tofts --tofts-basis results/tofts_vs_patlak/basis_sl21_r8_rms1.npz --slices 21 --seed 0 --ff-seed 0 \
  --steps 8 --eval-every 4 --lr-schedule cosine --coef-tv-weight 0.03 --coef-tv-every 4 --batch-size 4096 --no-wandb \
  --spoke-keep-file spoke_masks/keep_f80match.npy --spoke-heldout-file spoke_masks/val_k80_m8.npy --no-compile --no-restore --save-dir $OUT
echo "SMOKE exit $?"; rm -rf $OUT
