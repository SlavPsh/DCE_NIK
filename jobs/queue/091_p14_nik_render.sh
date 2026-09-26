#!/bin/bash
#SBATCH -J p14nikR
#SBATCH -p gpu
#SBATCH --gres=gpu:2g.24gb:1
#SBATCH -c 8
#SBATCH --mem 64G
#SBATCH -t 1:30:00
#SBATCH --array=0-20
# render-only rerun of 069: every task trained to step 10000 but the final coil-combine render oom-killed at 40G (four full 512 grid complex128
# copies); recon_nik_cart is now frame-chunked. --resume loads ckpt_slice_Z.pt (step 10000), skips the loop, renders, saves model + nik_slice_cplx.
# p14 nik arms under the production protocol (10k steps, final weights, wd 1e-2, unit-rms atoms, support prior 1 on the coefficient-map arms, k80 views):
# 0-8 tofts8 input-coil + prior (3 slices x 3 seeds); 9-11 tofts8 output-coil + prior (seed 0); 12-14 patlak + prior; 15-17 sub16 input-coil + prior;
# 18-20 nik-free (no prior possible). slices 21 / 24 / 27 -> results/tofts_vs_patlak/p14/invivo_prod{,_oc}/<arm>_sl<Z>_s<S>. 512 grid: ~1.8x the p3 cost.
# task 0 chains the evaluation afterany (tables with corrected peaks need mf_peak_check first, so it runs it), iq tracker, comparison figures per slice.
set -uo pipefail
D=/net/beegfs/users/P101440/DCE_NIK; cd $D
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH DCE_DS=p14
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P="micromamba run -n torch29 python -u"
RES=results/tofts_vs_patlak; OUTR=$RES/p14; SELF=$D/jobs/queue/091_p14_nik_render.sh; REFD=/net/beegfs/users/P101440/grasp_pro_py/results_ref_p14
GV=/net/beegfs/users/P101440/grasp_v2/results_grasp_v2_p14; GP=/net/beegfs/users/P101440/grasp_pro_py/results_spoke_cs_p14
echo "$(date '+%F %T') host $(hostname) job ${SLURM_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID:-none} stage ${STAGE:-train}"; nvidia-smi -L || true

if [ "${STAGE:-train}" = eval ]; then
  for Z in 21 24 27; do $P mf_peak_check.py --slice $Z --windows 31,21,15,11; done; echo "PEAK exit $?"
  $P tofts_eval_invivo.py --spokes k80 --slices 21,24,27 --arms tofts8,patlak,sub16,free --iv-dir p14/invivo_prod --basis-suffix _rms1 --suffix _p14_prod; echo "EVAL prod exit $?"
  $P tofts_eval_invivo.py --spokes k80 --slices 21,24,27 --arms tofts8 --iv-dir p14/invivo_prod_oc --basis-suffix _rms1 --suffix _p14_prod_oc; echo "EVAL oc exit $?"
  $P add_peak_correction.py invivo_p14_prod invivo_p14_prod_oc; echo "PEAKCORR exit $?"
  $P roi_check_invivo.py --slices 21,24,27 --t-show 90 || true
  for Z in 21 24 27; do
    IQ_VARIANTS="" IQ_TAG=_p14_prod_sl$Z IQ_EXTRA="tofts8 in-coil+prior:$D/$OUTR/invivo_prod/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,tofts8 out-coil+prior:$D/$OUTR/invivo_prod_oc/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,patlak+prior:$D/$OUTR/invivo_prod/patlak_sl${Z}_s0/nik_slice_${Z}_cplx.npy,sub16+prior:$D/$OUTR/invivo_prod/sub16_sl${Z}_s0/nik_slice_${Z}_cplx.npy,NIK-free:$D/$OUTR/invivo_prod/free_sl${Z}_s0/nik_slice_${Z}_cplx.npy" $P tofts_iq_track.py --slice $Z
    $P compare_runs_fig.py --slice $Z --out $RES/figures/p14_prod_sl$Z.png --title "p14 (liver slab) slice $Z, k80, production protocol: images at 90 s, liver / spleen zoom, roi curves (liver, spleen, aorta)" --items "tofts8 in-coil + prior:$D/$OUTR/invivo_prod/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,tofts8 out-coil + prior:$D/$OUTR/invivo_prod_oc/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,patlak + prior:$D/$OUTR/invivo_prod/patlak_sl${Z}_s0/nik_slice_${Z}_cplx.npy,sub16 + prior:$D/$OUTR/invivo_prod/sub16_sl${Z}_s0/nik_slice_${Z}_cplx.npy,NIK-free:$D/$OUTR/invivo_prod/free_sl${Z}_s0/nik_slice_${Z}_cplx.npy,GRASP:$GV/gv2_slice${Z}_n12_k80.npy,GRASP-Pro:$GP/cs_slice${Z}_f80match.npy"
  done; echo "IQ+FIGS exit $?"
  exit
fi
i=$SLURM_ARRAY_TASK_ID; SL=(21 24 27)
if [ "$i" = 0 ]; then
  sbatch --dependency=afterany:$SLURM_ARRAY_JOB_ID --export=ALL,STAGE=eval --array=0 -J p14nik_eval \
    --gres=gpu:1g.12gb:1 -c 4 --mem 48G -t 6:00:00 --output=$D/jobs/log/091_p14_nik_eval_%j.out --error=$D/jobs/log/091_p14_nik_eval_%j.out $SELF && echo "eval chained afterany $SLURM_ARRAY_JOB_ID"
fi
PRIOR() { echo "--support-weight 1 --support-mask spoke_masks/support_p14_sl$1_d6.npy --support-orient rot180 --coef-tv-every 32"; }
if   [ $i -lt 9 ];  then j=$i;        Z=${SL[$((j/3))]}; S=$((j%3)); NM=tofts8;   OUT=$OUTR/invivo_prod/tofts8_sl${Z}_s$S;    MA="--model wire_ff_tofts --tofts-basis $RES/basis_p14_sl${Z}_r8_rms1.npz $(PRIOR $Z)"
elif [ $i -lt 12 ]; then j=$((i-9));  Z=${SL[$j]}; S=0; NM=tofts8oc; OUT=$OUTR/invivo_prod_oc/tofts8_sl${Z}_s0;  MA="--model wire_ff_tofts --tofts-basis $RES/basis_p14_sl${Z}_r8_rms1.npz --coil-mode output $(PRIOR $Z)"
elif [ $i -lt 15 ]; then j=$((i-12)); Z=${SL[$j]}; S=0; NM=patlak;   OUT=$OUTR/invivo_prod/patlak_sl${Z}_s0;    MA="--model wire_ff_patlak --aif-file $D/aif_p14_slice$Z.npz --patlak-free 0 $(PRIOR $Z)"
elif [ $i -lt 18 ]; then j=$((i-15)); Z=${SL[$j]}; S=0; NM=sub16;    OUT=$OUTR/invivo_prod/sub16_sl${Z}_s0;     MA="--model wire_ff_subspace --rank 16 $(PRIOR $Z)"
else                     j=$((i-18)); Z=${SL[$j]}; S=0; NM=free;     OUT=$OUTR/invivo_prod/free_sl${Z}_s0;      MA="--model wire_ff_res"
fi
for f in $RES/basis_p14_sl${Z}_r8_rms1.npz spoke_masks/support_p14_sl${Z}_d6.npy spoke_masks/keep_k80_p14.npy $REFD/slice_$Z.npz; do [ -f $f ] || { echo "missing $f (068 not done)"; exit 1; }; done
[ -f $OUT/ckpt_slice_$Z.pt ] || { echo "missing $OUT/ckpt_slice_$Z.pt"; exit 2; }; echo "task $i: $NM slice $Z seed $S -> $OUT (render only from the step-10000 ckpt)"
$P train_grasp_nik.py $MA --slices $Z --seed $S --ff-seed $S --steps 10000 --no-restore --weight-decay 0.01 --out-dir $REFD \
  --spoke-keep-file spoke_masks/keep_k80_p14.npy --spoke-heldout-file spoke_masks/val_k80_m8_p14.npy \
  --no-compile --resume --save-dir $OUT 2>&1 | tee -a $OUT/train.log
echo "$(date '+%F %T') TRAIN_DONE $NM sl$Z s$S exit ${PIPESTATUS[0]}"
