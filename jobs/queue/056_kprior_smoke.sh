#!/bin/bash
#SBATCH -J kpsmoke
#SBATCH -p gpu
#SBATCH --gres=gpu:1g.12gb:1
#SBATCH -c 4
#SBATCH --mem 24G
#SBATCH -t 0:40:00
# gpu smoke test of the support prior + pisco term (8 steps, both on) before 055 reaches the gpu
cd /net/beegfs/users/P101440/DCE_NIK
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
P="micromamba run -n torch29 python -u"; RES=results/tofts_vs_patlak; OUT=$RES/amp_track/smoke_kp_sl21; mkdir -p $OUT
[ -f spoke_masks/support_sl21.npy ] || $P support_mask.py --slices 21 --dilate 4
$P train_grasp_nik.py --model wire_ff_tofts --tofts-basis $RES/basis_sl21_r8_rms1.npz --slices 21 --seed 0 --ff-seed 0 --steps 8 --eval-every 4 --batch-size 8192 --no-wandb \
  --support-weight 0.3 --support-mask spoke_masks/support_sl21.npy --coef-tv-every 4 --pisco-weight 0.1 --pisco-every 2 \
  --spoke-keep-file spoke_masks/keep_f80match.npy --spoke-heldout-file spoke_masks/val_k80_m8.npy --no-compile --no-restore --save-dir $OUT; echo "SMOKE kp exit $?"
rm -rf $OUT
