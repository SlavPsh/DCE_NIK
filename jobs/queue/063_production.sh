#!/bin/bash
#SBATCH -J prod
#SBATCH -p gpu
#SBATCH --gres=gpu:2g.24gb:1
#SBATCH -c 8
#SBATCH --mem 32G
#SBATCH -t 3:00:00
#SBATCH --array=0-38
# PRODUCTION RUN 2026-09-21 (user decisions): standard protocol = k80 (keep v%10<8, val v%10==8), 10k steps, final weights kept, wd 1e-2, unit-rms atoms,
# support prior 1 on every coefficient-map arm (mask dilated 6 px = about 12 mm, above the anterior-wall breathing excursion; 4 px = 8 mm was the test value),
# orientation rot180 (verified by support_diag.py).
# tasks: 0-8 tofts8 input-coil + prior (3 slices x 3 seeds) -> invivo_prod/tofts8_*; 9-17 tofts8 output-coil + prior -> invivo_prod_oc/tofts8_*;
# 18-26 tofts8 input-coil, wd 1e-2, NO prior (matched base) -> invivo_prod_base/tofts8_*; 27-29 patlak + prior (seed 0) -> invivo_prod/patlak_*;
# 30-32 sub16 input-coil + prior (seed 0) -> invivo_prod/sub16_*; 33-35 nik-free standard protocol (wire_ff_res, wd 1e-2, 10k, no restore, val spokes;
# no coefficient maps -> no prior possible) -> invivo_prod/free_*; 36-38 tofts8 output-coil matched base (no prior, seed 0) -> invivo_prod_base/tofts8oc_*.
# note for the record: per-weight wd on the 8x larger output head = 8x aggregate shrinkage on that head; flag if the coil modes diverge.
# task 0 chains the evaluation afterany: k80 tables + corrected reference peaks (three dirs), span diagnostic, image-quality tracker per slice,
# comparison figures per slice, story panels with the protocol note.
set -uo pipefail
D=/net/beegfs/users/P101440/DCE_NIK; cd $D
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P="micromamba run -n torch29 python -u"
RES=results/tofts_vs_patlak; SELF=$D/jobs/queue/063_production.sh; GV=/net/beegfs/users/P101440/grasp_v2/results_grasp_v2; GP=/net/beegfs/users/P101440/grasp_pro_py/results_spoke_cs
echo "$(date '+%F %T') host $(hostname) job ${SLURM_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID:-none} stage ${STAGE:-train}"; nvidia-smi -L || true

if [ "${STAGE:-train}" = eval ]; then
  $P tofts_eval_invivo.py --spokes k80 --slices 18,19,21 --arms tofts8,patlak,sub16,free --iv-dir invivo_prod --basis-suffix _rms1 --suffix _prod; echo "EVAL prod exit $?"
  $P tofts_eval_invivo.py --spokes k80 --slices 18,19,21 --arms tofts8 --iv-dir invivo_prod_oc --basis-suffix _rms1 --suffix _prod_oc; echo "EVAL oc exit $?"
  $P tofts_eval_invivo.py --spokes k80 --slices 18,19,21 --arms tofts8,tofts8oc --iv-dir invivo_prod_base --basis-suffix _rms1 --suffix _prod_base; echo "EVAL base exit $?"
  $P add_peak_correction.py invivo_prod invivo_prod_oc invivo_prod_base; echo "PEAKCORR exit $?"
  for Z in 21 18 19; do IV_DIR=invivo_prod STORY_TAG=_prod $P tofts_span_diag.py --slice $Z --arms tofts8,patlak,sub16; done; echo "SPAN exit $?"
  for Z in 21 18 19; do
    IQ_VARIANTS="" IQ_TAG=_prod_sl$Z IQ_EXTRA="tofts8 prod in-coil+prior:$D/$RES/invivo_prod/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,tofts8 prod out-coil+prior:$D/$RES/invivo_prod_oc/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,tofts8 base wd1e-2 no prior:$D/$RES/invivo_prod_base/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,tofts8oc base no prior:$D/$RES/invivo_prod_base/tofts8oc_sl${Z}_s0/nik_slice_${Z}_cplx.npy,patlak prod+prior:$D/$RES/invivo_prod/patlak_sl${Z}_s0/nik_slice_${Z}_cplx.npy,sub16 prod+prior:$D/$RES/invivo_prod/sub16_sl${Z}_s0/nik_slice_${Z}_cplx.npy,NIK-free prod:$D/$RES/invivo_prod/free_sl${Z}_s0/nik_slice_${Z}_cplx.npy" $P tofts_iq_track.py --slice $Z
    $P compare_runs_fig.py --slice $Z --out $RES/figures/prod_sl$Z.png --title "slice $Z, k80, production protocol (10k, no restore, wd 1e-2, support prior 1 on coefficient-map arms; NIK-free cannot take the prior): images at 90 s, kidney zoom, roi curves" --items "tofts8 in-coil + prior:$D/$RES/invivo_prod/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,tofts8 out-coil + prior:$D/$RES/invivo_prod_oc/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,tofts8 in-coil no prior:$D/$RES/invivo_prod_base/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,patlak + prior:$D/$RES/invivo_prod/patlak_sl${Z}_s0/nik_slice_${Z}_cplx.npy,sub16 + prior:$D/$RES/invivo_prod/sub16_sl${Z}_s0/nik_slice_${Z}_cplx.npy,NIK-free:$D/$RES/invivo_prod/free_sl${Z}_s0/nik_slice_${Z}_cplx.npy,GRASP:$GV/gv2_slice${Z}_n12_k80.npy,GRASP-Pro:$GP/cs_slice${Z}_f80match.npy"
  done; echo "IQ+FIGS exit $?"
  IV_DIR=invivo_prod STORY_TAG=_prod OC_DIR=invivo_prod_oc FREE_PATH=$D/$RES/invivo_prod/free_sl21_s0/nik_slice_21_cplx.npy STORY_NOTE="every NIK arm: train_grasp_nik.py, k80, 10k steps, final weights, wd 1e-2; support prior 1 on tofts8 / patlak / sub16 (coefficient-map arms), NIK-free cannot take it; coil mode in the column name (input unless stated)" $P story_figs.py --only invivo --t-invivo 90 --tofts tofts8; echo "STORY exit $?"
  exit
fi

i=$SLURM_ARRAY_TASK_ID; SL=(18 19 21)
if [ "$i" = 0 ]; then
  sbatch --dependency=afterany:$SLURM_ARRAY_JOB_ID --export=ALL,STAGE=eval --array=0 -J prod_eval \
    --gres=gpu:1g.12gb:1 -c 4 --mem 32G -t 5:00:00 --output=$D/jobs/log/063_production_eval_%j.out --error=$D/jobs/log/063_production_eval_%j.out \
    $SELF && echo "eval chained afterany $SLURM_ARRAY_JOB_ID"
fi
for Z in 18 19 21; do [ -f spoke_masks/support_sl${Z}_d6.npy ] || $P support_mask.py --slices $Z --dilate 6; done
PRIOR() { echo "--support-weight 1 --support-mask spoke_masks/support_sl$1_d6.npy --support-orient rot180 --coef-tv-every 32"; }
if   [ $i -lt 9 ];  then j=$i;        Z=${SL[$((j/3))]}; S=$((j%3)); NM=tofts8;       OUT=$RES/invivo_prod/tofts8_sl${Z}_s$S;         MA="--model wire_ff_tofts --tofts-basis $RES/basis_sl${Z}_r8_rms1.npz $(PRIOR $Z)"
elif [ $i -lt 18 ]; then j=$((i-9));  Z=${SL[$((j/3))]}; S=$((j%3)); NM=tofts8oc;     OUT=$RES/invivo_prod_oc/tofts8_sl${Z}_s$S;      MA="--model wire_ff_tofts --tofts-basis $RES/basis_sl${Z}_r8_rms1.npz --coil-mode output $(PRIOR $Z)"
elif [ $i -lt 27 ]; then j=$((i-18)); Z=${SL[$((j/3))]}; S=$((j%3)); NM=tofts8base;   OUT=$RES/invivo_prod_base/tofts8_sl${Z}_s$S;    MA="--model wire_ff_tofts --tofts-basis $RES/basis_sl${Z}_r8_rms1.npz"
elif [ $i -lt 30 ]; then j=$((i-27)); Z=${SL[$j]}; S=0; NM=patlak;       OUT=$RES/invivo_prod/patlak_sl${Z}_s0;          MA="--model wire_ff_patlak --aif-file $D/aif_slice$Z.npz --patlak-free 0 $(PRIOR $Z)"
elif [ $i -lt 33 ]; then j=$((i-30)); Z=${SL[$j]}; S=0; NM=sub16;        OUT=$RES/invivo_prod/sub16_sl${Z}_s0;           MA="--model wire_ff_subspace --rank 16 $(PRIOR $Z)"
elif [ $i -lt 36 ]; then j=$((i-33)); Z=${SL[$j]}; S=0; NM=free;         OUT=$RES/invivo_prod/free_sl${Z}_s0;            MA="--model wire_ff_res"
else                     j=$((i-36)); Z=${SL[$j]}; S=0; NM=tofts8ocbase; OUT=$RES/invivo_prod_base/tofts8oc_sl${Z}_s0;   MA="--model wire_ff_tofts --tofts-basis $RES/basis_sl${Z}_r8_rms1.npz --coil-mode output"
fi
mkdir -p $OUT; echo "task $i: $NM slice $Z seed $S -> $OUT"
$P train_grasp_nik.py $MA --slices $Z --seed $S --ff-seed $S --steps 10000 --no-restore --weight-decay 0.01 \
  --spoke-keep-file spoke_masks/keep_f80match.npy --spoke-heldout-file spoke_masks/val_k80_m8.npy \
  --no-compile --resume --save-dir $OUT 2>&1 | tee $OUT/train.log
echo "$(date '+%F %T') TRAIN_DONE $NM sl$Z s$S exit ${PIPESTATUS[0]}"
