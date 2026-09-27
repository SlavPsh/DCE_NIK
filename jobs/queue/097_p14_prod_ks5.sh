#!/bin/bash
#SBATCH -J p14ks5
#SBATCH -p gpu
#SBATCH --gres=gpu:2g.24gb:1
#SBATCH -c 8
#SBATCH --mem 64G
#SBATCH -t 4:00:00
#SBATCH --array=0-11
# p14 production rerun with the fourier-feature sigma scaled to the 512 grid (sigma 5 instead of 2.5; blur away from the centre, FINDINGS section 7):
# 0-8 tofts8 in-coil + prior (3 slices x 3 seeds) -> p14/invivo_prod_ks5; 9-11 tofts8 out-coil + prior seed 0 -> p14/invivo_prod_ks5_oc. everything
# else = production protocol (10k, final weights, wd 1e-2, unit-rms atoms, support prior 1). task 0 chains the eval afterany on the gpu partition:
# curve tables with corrected peaks, iq tracks, comparison figures (with the sigma-2.5 production arm), radial sharpness diag.
set -uo pipefail
D=/net/beegfs/users/P101440/DCE_NIK; cd $D
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH DCE_DS=p14
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P="micromamba run -n torch29 python -u"
RES=results/tofts_vs_patlak; OUTR=$RES/p14; SELF=$D/jobs/queue/097_p14_prod_ks5.sh; REFD=/net/beegfs/users/P101440/grasp_pro_py/results_ref_p14
GV=/net/beegfs/users/P101440/grasp_v2/results_grasp_v2_p14; GP=/net/beegfs/users/P101440/grasp_pro_py/results_spoke_cs_p14
echo "$(date '+%F %T') host $(hostname) job ${SLURM_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID:-none} stage ${STAGE:-train}"; nvidia-smi -L || true

if [ "${STAGE:-train}" = eval ]; then
  $P tofts_eval_invivo.py --spokes k80 --slices 21,24,27 --arms tofts8 --iv-dir p14/invivo_prod_ks5 --basis-suffix _rms1 --suffix _p14_prod_ks5; echo "EVAL exit $?"
  $P tofts_eval_invivo.py --spokes k80 --slices 21,24,27 --arms tofts8 --iv-dir p14/invivo_prod_ks5_oc --basis-suffix _rms1 --suffix _p14_prod_ks5_oc; echo "EVAL oc exit $?"
  $P add_peak_correction.py invivo_p14_prod_ks5 invivo_p14_prod_ks5_oc; echo "PEAKCORR exit $?"
  for Z in 21 24 27; do
    ITEMS="tofts8 sigma2.5 (prod):$D/$OUTR/invivo_prod/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,tofts8 sigma5:$D/$OUTR/invivo_prod_ks5/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,tofts8 sigma5 s1:$D/$OUTR/invivo_prod_ks5/tofts8_sl${Z}_s1/nik_slice_${Z}_cplx.npy,tofts8 sigma5 s2:$D/$OUTR/invivo_prod_ks5/tofts8_sl${Z}_s2/nik_slice_${Z}_cplx.npy,tofts8 sigma5 out-coil:$D/$OUTR/invivo_prod_ks5_oc/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,GRASP:$GV/gv2_slice${Z}_n12_k80.npy,GRASP-Pro:$GP/cs_slice${Z}_f80match.npy"
    IQ_VARIANTS="" IQ_TAG=_p14_ks5 IQ_EXTRA="$ITEMS" $P tofts_iq_track.py --slice $Z
    $P compare_runs_fig.py --slice $Z --out $RES/figures/p14_ks5_sl$Z.png --title "p14 slice $Z, tofts8 in-coil + prior, fourier sigma 5 (grid-scaled) vs the sigma-2.5 production run: images at 90 s, liver zoom, roi curves" --items "$ITEMS"
    $P radial_blur_diag.py --slice $Z --tag _ks5 --items "$ITEMS"
  done; echo "IQ+FIGS exit $?"
  exit
fi
i=$SLURM_ARRAY_TASK_ID; SL=(21 24 27)
if [ "$i" = 0 ]; then
  sbatch --dependency=afterany:$SLURM_ARRAY_JOB_ID --export=ALL,STAGE=eval --array=0 -J p14ks5_eval \
    --gres=gpu:1g.12gb:1 -c 4 --mem 48G -t 4:00:00 --output=$D/jobs/log/097_p14_prod_ks5_eval_%j.out --error=$D/jobs/log/097_p14_prod_ks5_eval_%j.out $SELF && echo "eval chained afterany $SLURM_ARRAY_JOB_ID"
fi
PRIOR() { echo "--support-weight 1 --support-mask spoke_masks/support_p14_sl$1_d6.npy --support-orient rot180 --coef-tv-every 32"; }
if [ $i -lt 9 ]; then j=$i; Z=${SL[$((j/3))]}; S=$((j%3)); NM=tofts8ks5; OUT=$OUTR/invivo_prod_ks5/tofts8_sl${Z}_s$S; MA="--model wire_ff_tofts --tofts-basis $RES/basis_p14_sl${Z}_r8_rms1.npz --k-sigma 5 $(PRIOR $Z)"
else j=$((i-9)); Z=${SL[$j]}; S=0; NM=tofts8ks5oc; OUT=$OUTR/invivo_prod_ks5_oc/tofts8_sl${Z}_s0; MA="--model wire_ff_tofts --tofts-basis $RES/basis_p14_sl${Z}_r8_rms1.npz --k-sigma 5 --coil-mode output $(PRIOR $Z)"
fi
mkdir -p $OUT; echo "task $i: $NM slice $Z seed $S -> $OUT"
$P train_grasp_nik.py $MA --slices $Z --seed $S --ff-seed $S --steps 10000 --no-restore --weight-decay 0.01 --out-dir $REFD \
  --spoke-keep-file spoke_masks/keep_k80_p14.npy --spoke-heldout-file spoke_masks/val_k80_m8_p14.npy \
  --no-compile --resume --save-dir $OUT 2>&1 | tee $OUT/train.log
echo "$(date '+%F %T') TRAIN_DONE $NM sl$Z s$S exit ${PIPESTATUS[0]}"
