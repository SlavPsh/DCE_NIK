#!/bin/bash
#SBATCH -J p14ks5a
#SBATCH -p gpu
#SBATCH --gres=gpu:2g.24gb:1
#SBATCH -c 8
#SBATCH --mem 64G
#SBATCH -t 4:00:00
#SBATCH --array=0-14
# p14 at fourier sigma 5 (grid-scaled, FINDINGS section 7), the remaining arms so the p14 comparison is consistent: 0-5 tofts8 output-coil + prior
# seeds 1, 2 (3 slices) -> p14/invivo_prod_ks5_oc; 6-8 patlak + prior; 9-11 sub16 input-coil + prior; 12-14 nik-free (no prior possible) -> p14/invivo_prod_ks5.
# tofts8 in-coil (3 seeds) and out-coil seed 0 at sigma 5 exist from queue 097. otherwise the production protocol. task 0 chains the full eval afterany:
# tables with corrected peaks (both coil modes), iq tracks, comparison figures, radial diag, story figures.
set -uo pipefail
D=/net/beegfs/users/P101440/DCE_NIK; cd $D
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH DCE_DS=p14
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P="micromamba run -n torch29 python -u"
RES=results/tofts_vs_patlak; OUTR=$RES/p14; SELF=$D/jobs/queue/099_p14_ks5_arms.sh; REFD=/net/beegfs/users/P101440/grasp_pro_py/results_ref_p14
GV=/net/beegfs/users/P101440/grasp_v2/results_grasp_v2_p14; GP=/net/beegfs/users/P101440/grasp_pro_py/results_spoke_cs_p14
echo "$(date '+%F %T') host $(hostname) job ${SLURM_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID:-none} stage ${STAGE:-train}"; nvidia-smi -L || true

if [ "${STAGE:-train}" = eval ]; then
  $P tofts_eval_invivo.py --spokes k80 --slices 21,24,27 --arms tofts8,patlak,sub16,free --iv-dir p14/invivo_prod_ks5 --basis-suffix _rms1 --suffix _p14_ks5; echo "EVAL exit $?"
  $P tofts_eval_invivo.py --spokes k80 --slices 21,24,27 --arms tofts8 --iv-dir p14/invivo_prod_ks5_oc --basis-suffix _rms1 --suffix _p14_ks5_oc; echo "EVAL oc exit $?"
  $P add_peak_correction.py invivo_p14_ks5 invivo_p14_ks5_oc; echo "PEAKCORR exit $?"
  for Z in 21 24 27; do
    ITEMS="tofts8 in-coil+prior:$D/$OUTR/invivo_prod_ks5/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,tofts8 out-coil+prior:$D/$OUTR/invivo_prod_ks5_oc/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,patlak+prior:$D/$OUTR/invivo_prod_ks5/patlak_sl${Z}_s0/nik_slice_${Z}_cplx.npy,sub16+prior:$D/$OUTR/invivo_prod_ks5/sub16_sl${Z}_s0/nik_slice_${Z}_cplx.npy,NIK-free:$D/$OUTR/invivo_prod_ks5/free_sl${Z}_s0/nik_slice_${Z}_cplx.npy,GRASP:$GV/gv2_slice${Z}_n12_k80.npy,GRASP-Pro:$GP/cs_slice${Z}_f80match.npy"
    IQ_VARIANTS="" IQ_TAG=_p14_ks5all IQ_EXTRA="$ITEMS" $P tofts_iq_track.py --slice $Z
    $P compare_runs_fig.py --slice $Z --out $RES/figures/p14_ks5all_sl$Z.png --title "p14 (liver slab) slice $Z, k80, production protocol at fourier sigma 5 (grid-scaled): images at 90 s, liver / spleen zoom, roi curves (liver, spleen, aorta)" --items "$ITEMS"
    $P radial_blur_diag.py --slice $Z --tag _ks5all --items "$ITEMS"
  done; echo "IQ+FIGS exit $?"
  exit
fi
i=$SLURM_ARRAY_TASK_ID; SL=(21 24 27)
if [ "$i" = 0 ]; then
  sbatch --dependency=afterany:$SLURM_ARRAY_JOB_ID --export=ALL,STAGE=eval --array=0 -J p14ks5a_eval \
    --gres=gpu:1g.12gb:1 -c 4 --mem 48G -t 4:00:00 --output=$D/jobs/log/099_p14_ks5_arms_eval_%j.out --error=$D/jobs/log/099_p14_ks5_arms_eval_%j.out $SELF && echo "eval chained afterany $SLURM_ARRAY_JOB_ID"
fi
PRIOR() { echo "--support-weight 1 --support-mask spoke_masks/support_p14_sl$1_d6.npy --support-orient rot180 --coef-tv-every 32"; }
if   [ $i -lt 6 ];  then j=$i;        Z=${SL[$((j/2))]}; S=$((j%2+1)); NM=tofts8oc; OUT=$OUTR/invivo_prod_ks5_oc/tofts8_sl${Z}_s$S; MA="--model wire_ff_tofts --tofts-basis $RES/basis_p14_sl${Z}_r8_rms1.npz --coil-mode output $(PRIOR $Z)"
elif [ $i -lt 9 ];  then j=$((i-6));  Z=${SL[$j]}; S=0; NM=patlak; OUT=$OUTR/invivo_prod_ks5/patlak_sl${Z}_s0; MA="--model wire_ff_patlak --aif-file $D/aif_p14_slice$Z.npz --patlak-free 0 $(PRIOR $Z)"
elif [ $i -lt 12 ]; then j=$((i-9));  Z=${SL[$j]}; S=0; NM=sub16;  OUT=$OUTR/invivo_prod_ks5/sub16_sl${Z}_s0;  MA="--model wire_ff_subspace --rank 16 $(PRIOR $Z)"
else                     j=$((i-12)); Z=${SL[$j]}; S=0; NM=free;   OUT=$OUTR/invivo_prod_ks5/free_sl${Z}_s0;   MA="--model wire_ff_res"
fi
mkdir -p $OUT; echo "task $i: $NM slice $Z seed $S sigma 5 -> $OUT"
$P train_grasp_nik.py $MA --k-sigma 5 --slices $Z --seed $S --ff-seed $S --steps 10000 --no-restore --weight-decay 0.01 --out-dir $REFD \
  --spoke-keep-file spoke_masks/keep_k80_p14.npy --spoke-heldout-file spoke_masks/val_k80_m8_p14.npy \
  --no-compile --resume --save-dir $OUT 2>&1 | tee $OUT/train.log
echo "$(date '+%F %T') TRAIN_DONE $NM sl$Z s$S exit ${PIPESTATUS[0]}"
