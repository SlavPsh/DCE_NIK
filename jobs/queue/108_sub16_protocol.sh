#!/bin/bash
#SBATCH -J sub16p
#SBATCH -p gpu
#SBATCH --gres=gpu:2g.24gb:1
#SBATCH -c 8
#SBATCH --mem 40G
#SBATCH -t 3:00:00
#SBATCH --array=0-1
# sub16 protocol attribution (user rule 2026-10-02, memory feedback_protocol_per_model): the production protocol (wd 1e-2 + support prior, chosen on
# tofts8) cut sub16's kidney amplitude by 20 to 30% vs its queue-051 protocol (wd 3e-3, no prior). which change does it: 0 wd 3e-3 + prior;
# 1 wd 1e-2, no prior. the 051 runs (invivo_k80_rms1/sub16_sl21_s*) are the wd 3e-3 / no-prior cell, the production run the wd 1e-2 / prior cell.
# p3 slice 21, seed 0, otherwise identical (rank 16 input-coil, pca warm start 100 frames, 10k, final weights, k80, sigma 2.5). task 0 chains the eval.
set -uo pipefail
D=/net/beegfs/users/P101440/DCE_NIK; cd $D
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P="micromamba run -n torch29 python -u"
RES=results/tofts_vs_patlak; OUTR=$RES/sub16_proto; SELF=$D/jobs/queue/108_sub16_protocol.sh; Z=21
GV=/net/beegfs/users/P101440/grasp_v2/results_grasp_v2; GP=/net/beegfs/users/P101440/grasp_pro_py/results_spoke_cs
echo "$(date '+%F %T') host $(hostname) job ${SLURM_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID:-none} stage ${STAGE:-train}"; nvidia-smi -L || true
TAGS=(wd3e-3_prior wd1e-2_noprior)
ITEMS="sub16 051 wd3e-3 noprior:$D/$RES/invivo_k80_rms1/sub16_sl${Z}_s0/nik_slice_${Z}_cplx.npy,sub16 prod wd1e-2 prior:$D/$RES/invivo_prod/sub16_sl${Z}_s0/nik_slice_${Z}_cplx.npy,sub16 wd3e-3 prior:$D/$OUTR/wd3e-3_prior/sub16_sl${Z}_s0/nik_slice_${Z}_cplx.npy,sub16 wd1e-2 noprior:$D/$OUTR/wd1e-2_noprior/sub16_sl${Z}_s0/nik_slice_${Z}_cplx.npy,tofts8 prod:$D/$RES/invivo_prod/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy"
if [ "${STAGE:-train}" = eval ]; then
  for T in "${TAGS[@]}"; do $P tofts_eval_invivo.py --spokes k80 --slices $Z --arms sub16 --iv-dir sub16_proto/$T --basis-suffix _rms1 --suffix _sub16_$T; done; echo "EVAL exit $?"
  IQ_VARIANTS="" IQ_TAG=_sub16_proto IQ_EXTRA="$ITEMS" $P tofts_iq_track.py --slice $Z; echo "IQ exit $?"
  $P compare_runs_fig.py --slice $Z --out $RES/figures/sub16_proto_sl$Z.png --title "p3 slice $Z, sub16 protocol attribution: wd 3e-3 vs 1e-2, with / without the support prior (051 and production cells reused); tofts8 production and GRASP for scale" --items "$ITEMS,GRASP:$GV/gv2_slice${Z}_n12_k80.npy"; echo "FIG exit $?"
  $P sub16_atoms_diag.py --slice $Z --runs "sub16 051:$D/$RES/invivo_k80_rms1/sub16_sl${Z}_s0,sub16 prod:$D/$RES/invivo_prod/sub16_sl${Z}_s0,sub16 wd3e-3 prior:$D/$OUTR/wd3e-3_prior/sub16_sl${Z}_s0,sub16 wd1e-2 noprior:$D/$OUTR/wd1e-2_noprior/sub16_sl${Z}_s0" --tag _proto; echo "ATOMS exit $?"
  exit
fi
i=$SLURM_ARRAY_TASK_ID; T=${TAGS[$i]}
if [ "$i" = 0 ]; then
  sbatch --dependency=afterany:$SLURM_ARRAY_JOB_ID --export=ALL,STAGE=eval --array=0 -J sub16p_eval \
    --gres=gpu:1g.12gb:1 -c 4 --mem 32G -t 2:00:00 --output=$D/jobs/log/108_sub16_protocol_eval_%j.out --error=$D/jobs/log/108_sub16_protocol_eval_%j.out $SELF && echo "eval chained afterany $SLURM_ARRAY_JOB_ID"
fi
case $T in
  wd3e-3_prior)   X="--weight-decay 0.003 --support-weight 1 --support-mask spoke_masks/support_sl${Z}_d6.npy --support-orient rot180 --coef-tv-every 32";;
  wd1e-2_noprior) X="--weight-decay 0.01";;
esac
OUT=$OUTR/$T/sub16_sl${Z}_s0; mkdir -p $OUT; echo "task $i: sub16 $T ($X) -> $OUT"
$P train_grasp_nik.py --model wire_ff_subspace --rank 16 $X --slices $Z --seed 0 --ff-seed 0 --steps 10000 --no-restore \
  --spoke-keep-file spoke_masks/keep_f80match.npy --spoke-heldout-file spoke_masks/val_k80_m8.npy \
  --no-compile --resume --save-dir $OUT 2>&1 | tee $OUT/train.log
echo "$(date '+%F %T') TRAIN_DONE sub16 $T exit ${PIPESTATUS[0]}"
