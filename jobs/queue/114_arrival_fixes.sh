#!/bin/bash
#SBATCH -J arrfix
#SBATCH -p gpu
#SBATCH --gres=gpu:2g.24gb:1
#SBATCH -c 8
#SBATCH --mem 40G
#SBATCH -t 3:00:00
#SBATCH --array=0-5
# arrival-artifact fixes (FINDINGS 9), tofts8 in-coil + prior, p3 slice 21, seed 0, production protocol otherwise. the per-model rule applies:
# tofts8 only, baseline = invivo_prod/tofts8_sl21_s0 (and the out-coil run as lever 4, no new run). variants -> results/tofts_vs_patlak/arrival_fix/<tag>/tofts<R>_sl21_s0:
# 0 param: basis from the box-deconvolved parametric aif (aif_param_fit.py -> basis_sl21_r8_param_rms1.npz); 1 supfast5 / 2 supfast20: support prior
# weight x5 / x20 on the residual atoms (atoms 3..7); 3 tvfast: huber tv 0.3 on the residual atoms only; 4 rank5; 5 rank6 (bases r5 / r6 from the
# measured aif). STAGE=prep (cpu, task 0 runs it first, inline) builds the aif fit and the bases; task 0 chains the eval afterany: curve tables with
# corrected peaks per variant, arrival diagnostic per variant (windows + per-atom tables), iq track vs the late ruler, comparison figure.
set -uo pipefail
D=/net/beegfs/users/P101440/DCE_NIK; cd $D
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P="micromamba run -n torch29 python -u"
RES=results/tofts_vs_patlak; OUTR=$RES/arrival_fix; SELF=$D/jobs/queue/114_arrival_fixes.sh; Z=21
GV=/net/beegfs/users/P101440/grasp_v2/results_grasp_v2
echo "$(date '+%F %T') host $(hostname) job ${SLURM_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID:-none} stage ${STAGE:-train}"; nvidia-smi -L || true
TAGS=(param supfast5 supfast20 tvfast rank5 rank6); RANKS=(8 8 8 8 5 6)
BASE=$D/$RES/invivo_prod/tofts8_sl${Z}_s0; OC=$D/$RES/invivo_prod_oc/tofts8_sl${Z}_s0
prep() {
  [ -f $D/aif_param_slice$Z.npz ] || $P aif_param_fit.py --slice $Z
  [ -f $RES/basis_sl${Z}_r8_param_rms1.npz ] || { $P nik_tofts_basis.py --aif $D/aif_param_slice$Z.npz --out $RES/basis_sl${Z}_r8_param.npz --ranks 8 > $RES/basis_sl${Z}_r8_param.log 2>&1; $P basis_rms1.py $RES/basis_sl${Z}_r8_param.npz; }
  for RR in 5 6; do [ -f $RES/basis_sl${Z}_r${RR}_rms1.npz ] || { $P nik_tofts_basis.py --aif $D/aif_slice$Z.npz --out $RES/basis_sl${Z}_r${RR}.npz --ranks $RR > $RES/basis_sl${Z}_r${RR}.log 2>&1; $P basis_rms1.py $RES/basis_sl${Z}_r${RR}.npz; }; done   # RR: a loop var R would clobber the task's rank
  ls -la $D/aif_param_slice$Z.npz $RES/basis_sl${Z}_r8_param_rms1.npz $RES/basis_sl${Z}_r5_rms1.npz $RES/basis_sl${Z}_r6_rms1.npz
}
if [ "${STAGE:-train}" = eval ]; then
  [ -d $OUTR/param/tofts6_sl21_s0 ] && [ ! -d $OUTR/param/tofts8_sl21_s0 ] && mv -v $OUTR/param/tofts6_sl21_s0 $OUTR/param/tofts8_sl21_s0   # first submission: the prep loop clobbered R for task 0
  for i in 0 1 2 3 4 5; do T=${TAGS[$i]}; R=${RANKS[$i]}; SFX=_rms1; [ $T = param ] && SFX=_param_rms1
    $P tofts_eval_invivo.py --spokes k80 --slices $Z --arms tofts$R --iv-dir arrival_fix/$T --basis-suffix $SFX --suffix _arrfix_$T; $P add_peak_correction.py invivo_arrfix_$T
    $P arrival_artifact_diag.py --slice $Z --tag _arrfix_$T --model "tofts$R $T:$D/$OUTR/$T/tofts${R}_sl${Z}_s0" --items "tofts$R $T:$D/$OUTR/$T/tofts${R}_sl${Z}_s0/nik_slice_${Z}_cplx.npy,tofts8 prod in-coil:$BASE/nik_slice_${Z}_cplx.npy,tofts8 prod out-coil:$OC/nik_slice_${Z}_cplx.npy,GRASP:$GV/gv2_slice${Z}_n12_k80.npy"
  done; echo "EVAL+DIAG exit $?"
  IT="tofts8 prod in-coil:$BASE/nik_slice_${Z}_cplx.npy,tofts8 prod out-coil:$OC/nik_slice_${Z}_cplx.npy"
  for i in 0 1 2 3 4 5; do T=${TAGS[$i]}; R=${RANKS[$i]}; IT="$IT,tofts$R $T:$D/$OUTR/$T/tofts${R}_sl${Z}_s0/nik_slice_${Z}_cplx.npy"; done
  IQ_VARIANTS="" IQ_TAG=_arrfix IQ_EXTRA="$IT" $P tofts_iq_track.py --slice $Z; echo "IQ exit $?"
  $P compare_runs_fig.py --slice $Z --out $RES/figures/arrival_fix_sl$Z.png --title "p3 slice $Z, tofts8 arrival-artifact levers: parametric aif basis, support x5 / x20 on the residual atoms, tv on the residual atoms, rank 5 / 6; vs production in / out coil and GRASP" --items "$IT,GRASP:$GV/gv2_slice${Z}_n12_k80.npy"; echo "FIG exit $?"
  exit
fi
i=$SLURM_ARRAY_TASK_ID; T=${TAGS[$i]}; R=${RANKS[$i]}
if [ "$i" = 0 ]; then
  prep; echo "PREP exit $?"
  [ -n "${NOCHAIN:-}" ] || sbatch --dependency=afterany:$SLURM_ARRAY_JOB_ID --export=ALL,STAGE=eval --array=0 -J arrfix_eval \
    --gres=gpu:1g.12gb:1 -c 4 --mem 48G -t 4:00:00 --output=$D/jobs/log/114_arrival_fixes_eval_%j.out --error=$D/jobs/log/114_arrival_fixes_eval_%j.out $SELF && echo "eval chained afterany $SLURM_ARRAY_JOB_ID"
else
  for k in $(seq 1 90); do [ -f $RES/basis_sl${Z}_r8_param_rms1.npz ] && [ -f $RES/basis_sl${Z}_r5_rms1.npz ] && [ -f $RES/basis_sl${Z}_r6_rms1.npz ] && break; sleep 20; done   # wait for task 0's prep
fi
PRIOR="--support-weight 1 --support-mask spoke_masks/support_sl${Z}_d6.npy --support-orient rot180 --coef-tv-every 32"
case $T in
  param)     X="--tofts-basis $RES/basis_sl${Z}_r8_param_rms1.npz $PRIOR";;
  supfast5)  X="--tofts-basis $RES/basis_sl${Z}_r8_rms1.npz $PRIOR --support-fast-weight 5";;
  supfast20) X="--tofts-basis $RES/basis_sl${Z}_r8_rms1.npz $PRIOR --support-fast-weight 20";;
  tvfast)    X="--tofts-basis $RES/basis_sl${Z}_r8_rms1.npz $PRIOR --coef-tv-weight 0.3 --tv-slow-weight 0 --tv-fast-weight 1";;
  rank5)     X="--tofts-basis $RES/basis_sl${Z}_r5_rms1.npz $PRIOR";;
  rank6)     X="--tofts-basis $RES/basis_sl${Z}_r6_rms1.npz $PRIOR";;
esac
OUT=$OUTR/$T/tofts${R}_sl${Z}_s0; mkdir -p $OUT; echo "task $i: $T ($X) -> $OUT"
$P train_grasp_nik.py --model wire_ff_tofts $X --slices $Z --seed 0 --ff-seed 0 --steps 10000 --no-restore --weight-decay 0.01 \
  --spoke-keep-file spoke_masks/keep_f80match.npy --spoke-heldout-file spoke_masks/val_k80_m8.npy \
  --no-compile --resume --save-dir $OUT 2>&1 | tee $OUT/train.log
echo "$(date '+%F %T') TRAIN_DONE $T exit ${PIPESTATUS[0]}"
