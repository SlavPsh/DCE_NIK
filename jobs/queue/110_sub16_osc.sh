#!/bin/bash
#SBATCH -J sub16osc
#SBATCH -p gpu
#SBATCH --gres=gpu:2g.24gb:1
#SBATCH -c 8
#SBATCH --mem 40G
#SBATCH -t 3:00:00
#SBATCH --array=0-2
# sub16 oscillation fix on top of its own protocol (wd 3e-3 + prior, FINDINGS 8a / 8c): the 104 tests removed the ripple with no warm start or
# phi-w0 10 but ran under wd 1e-2. p3 slice 21, seed 0: 0 nowarm; 1 w0_10; 2 nowarm + w0_10 -> results/tofts_vs_patlak/sub16_osc/<tag>/sub16_sl21_s0.
# baseline = sub16_proto/wd3e-3_prior (same protocol, pca warm, w0 30). task 0 chains the eval: curve table, atom diag, iq track, figure.
set -uo pipefail
D=/net/beegfs/users/P101440/DCE_NIK; cd $D
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P="micromamba run -n torch29 python -u"
RES=results/tofts_vs_patlak; OUTR=$RES/sub16_osc; SELF=$D/jobs/queue/110_sub16_osc.sh; Z=21
GV=/net/beegfs/users/P101440/grasp_v2/results_grasp_v2
echo "$(date '+%F %T') host $(hostname) job ${SLURM_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID:-none} stage ${STAGE:-train}"; nvidia-smi -L || true
TAGS=(nowarm w0_10 nowarm_w0_10)
BASE=$D/$RES/sub16_proto/wd3e-3_prior/sub16_sl${Z}_s0
if [ "${STAGE:-train}" = eval ]; then
  for T in "${TAGS[@]}"; do $P tofts_eval_invivo.py --spokes k80 --slices $Z --arms sub16 --iv-dir sub16_osc/$T --basis-suffix _rms1 --suffix _sub16osc_$T; done; echo "EVAL exit $?"
  RUNS="sub16 wd3e-3 prior (pca warm w0 30):$BASE,tofts8 prod:$D/$RES/invivo_prod/tofts8_sl${Z}_s0"; IT="sub16 wd3e-3+prior:$BASE/nik_slice_${Z}_cplx.npy,tofts8 prod:$D/$RES/invivo_prod/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy"
  for T in "${TAGS[@]}"; do RUNS="$RUNS,sub16 wd3e-3 $T:$D/$OUTR/$T/sub16_sl${Z}_s0"; IT="$IT,sub16 wd3e-3 $T:$D/$OUTR/$T/sub16_sl${Z}_s0/nik_slice_${Z}_cplx.npy"; done
  $P sub16_atoms_diag.py --slice $Z --runs "$RUNS" --tag _osc; echo "ATOMS exit $?"
  IQ_VARIANTS="" IQ_TAG=_sub16_osc IQ_EXTRA="$IT" $P tofts_iq_track.py --slice $Z; echo "IQ exit $?"
  $P compare_runs_fig.py --slice $Z --out $RES/figures/sub16_osc_sl$Z.png --title "p3 slice $Z, sub16 at wd 3e-3 + prior: oscillation fix (no pca warm start / atom-net w0 10 / both) vs its baseline and tofts8" --items "$IT,GRASP:$GV/gv2_slice${Z}_n12_k80.npy"; echo "FIG exit $?"
  exit
fi
i=$SLURM_ARRAY_TASK_ID; T=${TAGS[$i]}
if [ "$i" = 0 ]; then
  sbatch --dependency=afterany:$SLURM_ARRAY_JOB_ID --export=ALL,STAGE=eval --array=0 -J sub16osc_eval \
    --gres=gpu:1g.12gb:1 -c 4 --mem 32G -t 2:00:00 --output=$D/jobs/log/110_sub16_osc_eval_%j.out --error=$D/jobs/log/110_sub16_osc_eval_%j.out $SELF && echo "eval chained afterany $SLURM_ARRAY_JOB_ID"
fi
case $T in nowarm) X="--no-warmstart";; w0_10) X="--phi-w0 10";; nowarm_w0_10) X="--no-warmstart --phi-w0 10";; esac
OUT=$OUTR/$T/sub16_sl${Z}_s0; mkdir -p $OUT; echo "task $i: sub16 wd3e-3 prior $T ($X) -> $OUT"
$P train_grasp_nik.py --model wire_ff_subspace --rank 16 --weight-decay 0.003 --support-weight 1 --support-mask spoke_masks/support_sl${Z}_d6.npy --support-orient rot180 --coef-tv-every 32 $X \
  --slices $Z --seed 0 --ff-seed 0 --steps 10000 --no-restore \
  --spoke-keep-file spoke_masks/keep_f80match.npy --spoke-heldout-file spoke_masks/val_k80_m8.npy \
  --no-compile --resume --save-dir $OUT 2>&1 | tee $OUT/train.log
echo "$(date '+%F %T') TRAIN_DONE sub16osc $T exit ${PIPESTATUS[0]}"
