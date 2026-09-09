#!/bin/bash
#SBATCH -J ktvsmoke
#SBATCH -p gpu
#SBATCH --gres gpu:1g.12gb:1
#SBATCH -c 4
#SBATCH --mem 32G
#SBATCH -t 0:40:00
#SBATCH -o /net/beegfs/users/P101440/DCE_NIK/logs/ktvsmoke_%j.log
cd /net/beegfs/users/P101440/DCE_NIK
export PATH="/net/beegfs/users/P101440/micromamba/bin:$PATH"; export MAMBA_ROOT_PREFIX="/net/beegfs/users/P101440/micromamba"
micromamba run -n torch29 python train_grasp_nik.py \
  --slices 21 --model wire_ff_res --steps 50 --eval-every 25 --console-every 10 \
  --spoke-keep-file /net/beegfs/users/P101440/DCE_NIK/spoke_masks/keep_f80match.npy \
  --keep-heldout --no-compile --ktv21-weight 0.05 \
  --save-dir /net/beegfs/users/P101440/DCE_NIK/results_ktv_smoke
echo SMOKE_DONE
