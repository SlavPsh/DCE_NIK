#!/bin/bash
#SBATCH -J cmsmoke
#SBATCH -p gpu
#SBATCH --gres=gpu:1g.12gb:1
#SBATCH -c 4
#SBATCH --mem 24G
#SBATCH -t 0:30:00
# gpu smoke test of the new output-coil head (nik_model coil_mode='output') and the sub16 input-coil path in train_grasp_nik: 8 steps each, slice 21, no wandb
cd /net/beegfs/users/P101440/DCE_NIK
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
P="micromamba run -n torch29 python -u"; RES=results/tofts_vs_patlak; OUT=$RES/amp_track/smoke_cm_sl21; mkdir -p $OUT
$P train_grasp_nik.py --model wire_ff_tofts --tofts-basis $RES/basis_sl21_r8_rms1.npz --coil-mode output --slices 21 --seed 0 --ff-seed 0 --steps 8 --eval-every 4 --batch-size 8192 --no-wandb \
  --spoke-keep-file spoke_masks/keep_f80match.npy --spoke-heldout-file spoke_masks/val_k80_m8.npy --no-compile --no-restore --save-dir $OUT/oc; echo "SMOKE oc exit $?"
$P train_grasp_nik.py --model wire_ff_subspace --rank 16 --slices 21 --seed 0 --ff-seed 0 --steps 8 --eval-every 4 --batch-size 8192 --no-wandb --warmstart-steps 20 \
  --spoke-keep-file spoke_masks/keep_f80match.npy --spoke-heldout-file spoke_masks/val_k80_m8.npy --no-compile --no-restore --save-dir $OUT/sub16; echo "SMOKE sub16 exit $?"
$P -c "import torch; ck=torch.load('$OUT/oc/model_slice_21.pt', map_location='cpu', weights_only=False); print('ckpt coil_mode', ck.get('coil_mode'), 'a_head', ck['state_dict']['a_head.weight'].shape)"
rm -rf $OUT
