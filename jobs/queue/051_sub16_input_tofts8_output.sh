#!/bin/bash
#SBATCH -J coilmode
#SBATCH -p gpu
#SBATCH --gres=gpu:2g.24gb:1
#SBATCH -c 8
#SBATCH --mem 32G
#SBATCH -t 2:30:00
#SBATCH --array=0-17
# coil-parameterization consistency (user, 2026-09-17): (a) NIK-sub16 rerun as an INPUT-coil model with the pk-arm trainer and protocol
# (train_grasp_nik --model wire_ff_subspace --rank 16, phi warm-started from the k-centre pca, 10k steps, no restore, wd 3e-3, k80) so every nik
# arm in the in vivo panels shares trainer, coil mode and protocol; (b) NIK-tofts8 with the OUTPUT-coil head (--coil-mode output, unit-rms atoms,
# same protocol) as the test of the output-coil idea on the pk arm. tasks 0-8: sub16 input-coil -> invivo_k80_rms1/sub16_sl<Z>_s<S>;
# tasks 9-17: tofts8 output-coil -> invivo_k80_oc/tofts8_sl<Z>_s<S>. i%9 = slice_idx*3 + seed. task 0 chains eval + panels afterany.
set -uo pipefail
D=/net/beegfs/users/P101440/DCE_NIK; cd $D
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P="micromamba run -n torch29 python -u"
RES=results/tofts_vs_patlak; SELF=$D/jobs/queue/051_sub16_input_tofts8_output.sh
echo "$(date '+%F %T') host $(hostname) job ${SLURM_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID:-none} stage ${STAGE:-train}"; nvidia-smi -L || true

if [ "${STAGE:-train}" = eval ]; then
  $P tofts_eval_invivo.py --spokes k80 --slices 18,19,21 --arms patlak,tofts,tofts8,sub16 --iv-dir invivo_k80_rms1 --basis-suffix _rms1 --suffix _k80_rms1; echo "EVAL rms1 exit $?"
  $P tofts_eval_invivo.py --spokes k80 --slices 18,19,21 --arms tofts8 --iv-dir invivo_k80_oc --basis-suffix _rms1 --suffix _k80_oc; echo "EVAL oc exit $?"
  for Z in 21 18 19; do IV_DIR=invivo_k80_rms1 STORY_TAG=_rms1 $P tofts_span_diag.py --slice $Z --arms tofts8,patlak,sub16; done; echo "SPAN exit $?"
  IV_DIR=invivo_k80_rms1 STORY_TAG=_rms1_t8 OC_DIR=invivo_k80_oc $P story_figs.py --only invivo --t-invivo 90 --tofts tofts8; echo "STORY exit $?"
  exit
fi
i=$SLURM_ARRAY_TASK_ID; j=$((i % 9)); SL=(18 19 21); Z=${SL[$((j / 3))]}; S=$((j % 3))
if [ "$i" = 0 ]; then
  sbatch --dependency=afterany:$SLURM_ARRAY_JOB_ID --export=ALL,STAGE=eval --array=0 -J coilmode_eval \
    --gres=gpu:1g.12gb:1 -c 4 --mem 32G -t 3:00:00 --output=$D/jobs/log/051_coilmode_eval_%j.out --error=$D/jobs/log/051_coilmode_eval_%j.out \
    $SELF && echo "eval chained afterany $SLURM_ARRAY_JOB_ID"
fi
if [ "$i" -lt 9 ]; then NM=sub16; OUT=$RES/invivo_k80_rms1/sub16_sl${Z}_s$S; MA="--model wire_ff_subspace --rank 16"
else NM=tofts8oc; OUT=$RES/invivo_k80_oc/tofts8_sl${Z}_s$S; MA="--model wire_ff_tofts --tofts-basis $RES/basis_sl${Z}_r8_rms1.npz --coil-mode output"; fi
mkdir -p $OUT; echo "slice $Z arm $NM seed $S -> $OUT"
$P train_grasp_nik.py $MA --slices $Z --seed $S --ff-seed $S --steps 10000 --no-restore \
  --spoke-keep-file spoke_masks/keep_f80match.npy --spoke-heldout-file spoke_masks/val_k80_m8.npy \
  --no-compile --resume --save-dir $OUT
echo "$(date '+%F %T') TRAIN_DONE $NM sl$Z s$S exit $?"
