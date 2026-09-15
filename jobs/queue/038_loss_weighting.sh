#!/bin/bash
#SBATCH -J lossw
#SBATCH -p gpu
#SBATCH --gres=gpu:2g.24gb:1
#SBATCH -c 8
#SBATCH --mem 32G
#SBATCH -t 5:00:00
#SBATCH --array=0-3
# loss-weighting test for both amplitude problems (phantom aorta +7%, in vivo pk-arm deficit): envelope exponent 0 (natural spectrum, no
# high-|k| flattening) and ramp dcf weighting (power 1) vs the default (env 0.75, unweighted). same data, same seed, same everything else.
# 0 phantom tofts16 env0, 1 phantom tofts16 dcf1 (each ends with xph_eval), 2 in vivo tofts8 sl21 k80 env0, 3 in vivo tofts8 dcf1 (snapshots, no restore).
# task 0 chains a cpu stage afterany: aorta offset table with the two new phantom recons, amp tracker over all six in vivo variants.
set -uo pipefail
D=/net/beegfs/users/P101440/DCE_NIK; cd $D
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True XPH_SIM=nomotion
P="micromamba run -n torch29 python -u"
RES=results/tofts_vs_patlak; AT=$RES/amp_track; Z=21; XO=results/xcat_physical_nomotion_nik_vs_grasp; SELF=$D/jobs/queue/038_loss_weighting.sh
echo "$(date '+%F %T') host $(hostname) job ${SLURM_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID:-none} stage ${STAGE:-train}"; nvidia-smi -L || true

if [ "${STAGE:-train}" = cpu ]; then
  $P phantom_aorta_offset.py --suffix _lossw --extra "NIK-tofts env0:$D/$XO/arrays/nik_eval_w768_ks2.5_s0_tofts16_env0.npz:rec_best,NIK-tofts dcf1:$D/$XO/arrays/nik_eval_w768_ks2.5_s0_tofts16_dcf1.npz:rec_best"; echo "AORTA exit $?"
  $P tofts_amp_track.py --slice $Z --runs base:$D/$AT/base_sl$Z,wd0:$D/$AT/wd0_sl$Z,lr1e-4:$D/$AT/lr1e-4_sl$Z,atomscale:$D/$AT/atomscale_sl$Z,env0:$D/$AT/env0_sl$Z,dcf1:$D/$AT/dcf1_sl$Z; echo "TRACK exit $?"; exit
fi
i=$SLURM_ARRAY_TASK_ID
if [ "$i" = 0 ]; then
  sbatch --dependency=afterany:$SLURM_ARRAY_JOB_ID --export=ALL,STAGE=cpu --array=0 -J lossw_cpu -p defq --gres="" -c 4 --mem 24G -t 1:00:00 \
    --output=$D/jobs/log/038_loss_weighting_cpu_%j.out --error=$D/jobs/log/038_loss_weighting_cpu_%j.out $SELF && echo "cpu stage chained afterany $SLURM_ARRAY_JOB_ID"
fi
case $i in
  0) export XPH_ENV=0;   TAG=w768_ks2.5_s0_tofts16_env0; $P xph_train.py --hidden-width 768 --k-sigma 2.5 --seed 0 --model wire_ff_tofts --tag-suffix _env0 && $P xph_eval.py --tag $TAG; echo "PH env0 exit $?";;
  1) TAG=w768_ks2.5_s0_tofts16_dcf1; $P xph_train.py --hidden-width 768 --k-sigma 2.5 --seed 0 --model wire_ff_tofts --tag-suffix _dcf1 --dcf-power 1.0 && $P xph_eval.py --tag $TAG; echo "PH dcf1 exit $?";;
  2|3) if [ "$i" = 2 ]; then NM=env0; EX="--envelope-exponent 0"; else NM=dcf1; EX="--use-dcf 1 --dcf-power 1.0"; fi
     OUT=$AT/${NM}_sl$Z; mkdir -p $OUT; echo "variant $NM -> $OUT"
     $P train_grasp_nik.py --model wire_ff_tofts --tofts-basis $RES/basis_sl${Z}_r8.npz --slices $Z --seed 0 --ff-seed 0 --steps 40000 $EX \
       --spoke-keep-file spoke_masks/keep_f80match.npy --spoke-heldout-file spoke_masks/val_k80_m8.npy \
       --snapshot-every 2000 --no-restore --no-compile --resume --save-dir $OUT 2>&1 | tee $OUT/train.log
     echo "$(date '+%F %T') TRAIN_DONE $NM exit ${PIPESTATUS[0]}";;
esac
