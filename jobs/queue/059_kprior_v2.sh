#!/bin/bash
#SBATCH -J kprior2
#SBATCH -p gpu
#SBATCH --gres=gpu:2g.24gb:1
#SBATCH -c 8
#SBATCH --mem 32G
#SBATCH -t 5:00:00
#SBATCH --array=0-3
# k-space priors, second version after the diagnostics: support prior now = energy in the air ring inside the crop / energy in the body (sums; the
# whole-field mean ratio was 3e-4 and inert), weights 1 and 10; pisco now cross-coil only (own-coil neighbours made the relation trivially
# satisfied, residual 0.25%), weights 0.1 and 1. tofts8 unit-rms sl21 k80 seed 0, 12k, snapshots, no restore. task 0 chains the trackers.
set -uo pipefail
D=/net/beegfs/users/P101440/DCE_NIK; cd $D
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P="micromamba run -n torch29 python -u"
RES=results/tofts_vs_patlak; AT=$RES/amp_track; Z=21; SELF=$D/jobs/queue/059_kprior_v2.sh
echo "$(date '+%F %T') host $(hostname) job ${SLURM_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID:-none} stage ${STAGE:-train}"; nvidia-smi -L || true
if [ "${STAGE:-train}" = track ]; then
  $P tofts_amp_track.py --slice $Z --runs supv2_1:$D/$AT/supv2_1_sl$Z,supv2_10:$D/$AT/supv2_10_sl$Z,piscox0.1:$D/$AT/piscox0.1_sl$Z,piscox1:$D/$AT/piscox1_sl$Z; echo "AMP exit $?"
  IQ_VARIANTS=supv2_1,supv2_10,piscox0.1,piscox1 IQ_TAG=_kspace2 $P tofts_iq_track.py --slice $Z; echo "IQ exit $?"; exit
fi
i=$SLURM_ARRAY_TASK_ID
case $i in
  0) NM=supv2_1;   EX="--support-weight 1 --support-mask spoke_masks/support_sl$Z.npy --coef-tv-every 32";;
  1) NM=supv2_10;  EX="--support-weight 10 --support-mask spoke_masks/support_sl$Z.npy --coef-tv-every 32";;
  2) NM=piscox0.1; EX="--pisco-weight 0.1 --pisco-batch 512 --pisco-cross-only 1";;
  3) NM=piscox1;   EX="--pisco-weight 1.0 --pisco-batch 512 --pisco-cross-only 1";;
esac
if [ "$i" = 0 ]; then
  sbatch --dependency=afterany:$SLURM_ARRAY_JOB_ID --export=ALL,STAGE=track --array=0 -J kprior2_cpu -p defq --gres="" -c 4 --mem 24G -t 2:00:00 \
    --output=$D/jobs/log/059_kprior_v2_cpu_%j.out --error=$D/jobs/log/059_kprior_v2_cpu_%j.out $SELF && echo "tracker chained afterany $SLURM_ARRAY_JOB_ID"
fi
OUT=$AT/${NM}_sl$Z; rm -rf $OUT; mkdir -p $OUT; echo "variant $NM -> $OUT"
$P train_grasp_nik.py --model wire_ff_tofts --tofts-basis $RES/basis_sl${Z}_r8_rms1.npz --slices $Z --seed 0 --ff-seed 0 --steps 12000 $EX \
  --spoke-keep-file spoke_masks/keep_f80match.npy --spoke-heldout-file spoke_masks/val_k80_m8.npy \
  --snapshot-every 1000 --no-restore --no-compile --save-dir $OUT 2>&1 | tee $OUT/train.log
echo "$(date '+%F %T') TRAIN_DONE $NM exit ${PIPESTATUS[0]}"
