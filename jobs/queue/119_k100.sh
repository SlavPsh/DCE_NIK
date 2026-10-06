#!/bin/bash
#SBATCH -J k100
#SBATCH -p gpu
#SBATCH --gres=gpu:2g.24gb:1
#SBATCH -c 8
#SBATCH --mem 40G
#SBATCH -t 3:00:00
#SBATCH --array=0-4
# all spokes instead of k80 (user idea 2026-10-06): the k80 split only served the held-out early stop, which is no longer a selection criterion, and the
# image ruler is now the model-free late nufft. every nik arm on p3 slice 21 at k100 under its own protocol (tofts8 in / out coil + prior wd 1e-2,
# patlak + prior, sub16 wd 3e-3 + prior, free wd 1e-2; sigma 2.5; plateau scheduler on the running train loss since there are no held-out spokes),
# cs references at all spokes (grasp n12, grasp-pro 14 spf K5 = cs_slice21_f100). caveat: the late ruler shares its spokes with the training set,
# equally for every arm. -> results/tofts_vs_patlak/invivo_k100{,_oc}/<arm>_sl21_s0. task 0 chains the eval: curve tables (k100 cs columns), arrival diag
# (tofts8 in / out), iq track and comparison figure vs the k80 production arms.
set -uo pipefail
D=/net/beegfs/users/P101440/DCE_NIK; cd $D
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P="micromamba run -n torch29 python -u"
RES=results/tofts_vs_patlak; SELF=$D/jobs/queue/119_k100.sh; Z=21; GV=/net/beegfs/users/P101440/grasp_v2/results_grasp_v2; GP=/net/beegfs/users/P101440/grasp_pro_py/results_spoke_cs
echo "$(date '+%F %T') host $(hostname) job ${SLURM_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID:-none} stage ${STAGE:-train}"; nvidia-smi -L || true
if [ "${STAGE:-train}" = eval ]; then
  $P tofts_eval_invivo.py --spokes k100 --slices $Z --arms tofts8,patlak,sub16,free --iv-dir invivo_k100 --basis-suffix _rms1 --suffix _k100; $P add_peak_correction.py invivo_k100
  $P tofts_eval_invivo.py --spokes k100 --slices $Z --arms tofts8 --iv-dir invivo_k100_oc --basis-suffix _rms1 --suffix _k100_oc; $P add_peak_correction.py invivo_k100_oc; echo "EVAL exit $?"
  for A in "tofts8 k100 in-coil:$D/$RES/invivo_k100/tofts8_sl${Z}_s0" "tofts8 k100 out-coil:$D/$RES/invivo_k100_oc/tofts8_sl${Z}_s0"; do nm=${A%%:*}; d=${A#*:}; tag=$(echo $nm | tr ' ' '_')
    $P arrival_artifact_diag.py --slice $Z --tag _$tag --model "$A" --items "$nm:$d/nik_slice_${Z}_cplx.npy,tofts8 k80 in-coil:$D/$RES/invivo_prod/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,GRASP all:$GV/gv2_slice${Z}_n12.npy"
  done; echo "DIAG exit $?"
  IT="tofts8 k100 in-coil:$D/$RES/invivo_k100/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,tofts8 k100 out-coil:$D/$RES/invivo_k100_oc/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,patlak k100:$D/$RES/invivo_k100/patlak_sl${Z}_s0/nik_slice_${Z}_cplx.npy,sub16 (wd 3e-3) k100:$D/$RES/invivo_k100/sub16_sl${Z}_s0/nik_slice_${Z}_cplx.npy,NIK-free k100:$D/$RES/invivo_k100/free_sl${Z}_s0/nik_slice_${Z}_cplx.npy,GRASP all spokes:$GV/gv2_slice${Z}_n12.npy,GRASP-Pro all spokes:$GP/cs_slice${Z}_f100.npy"
  IQ_VARIANTS="" IQ_TAG=_k100 IQ_EXTRA="$IT,tofts8 k80 in-coil:$D/$RES/invivo_prod/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,sub16 (wd 3e-3) k80:$D/$RES/invivo_prod_sub16wd/sub16_sl${Z}_s0/nik_slice_${Z}_cplx.npy,NIK-free k80:$D/$RES/invivo_prod/free_sl${Z}_s0/nik_slice_${Z}_cplx.npy" $P tofts_iq_track.py --slice $Z; echo "IQ exit $?"
  $P compare_runs_fig.py --slice $Z --out $RES/figures/k100_sl$Z.png --title "p3 slice $Z, every arm on ALL spokes (k100) vs the k80 production tofts8: images at 90 s, kidney zoom, roi curves" --items "$IT,tofts8 k80 in-coil:$D/$RES/invivo_prod/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy"; echo "FIG exit $?"
  exit
fi
i=$SLURM_ARRAY_TASK_ID
if [ "$i" = 0 ]; then
  sbatch --dependency=afterany:$SLURM_ARRAY_JOB_ID --export=ALL,STAGE=eval --array=0 -J k100_eval \
    --gres=gpu:1g.12gb:1 -c 4 --mem 48G -t 3:00:00 --output=$D/jobs/log/119_k100_eval_%j.out --error=$D/jobs/log/119_k100_eval_%j.out $SELF && echo "eval chained afterany $SLURM_ARRAY_JOB_ID"
fi
PRIOR="--support-weight 1 --support-mask spoke_masks/support_sl${Z}_d6.npy --support-orient rot180 --coef-tv-every 32"
case $i in
  0) NM=tofts8;   OUT=$RES/invivo_k100/tofts8_sl${Z}_s0;    MA="--model wire_ff_tofts --tofts-basis $RES/basis_sl${Z}_r8_rms1.npz $PRIOR --weight-decay 0.01";;
  1) NM=tofts8oc; OUT=$RES/invivo_k100_oc/tofts8_sl${Z}_s0; MA="--model wire_ff_tofts --tofts-basis $RES/basis_sl${Z}_r8_rms1.npz --coil-mode output $PRIOR --weight-decay 0.01";;
  2) NM=patlak;   OUT=$RES/invivo_k100/patlak_sl${Z}_s0;    MA="--model wire_ff_patlak --aif-file $D/aif_slice$Z.npz --patlak-free 0 $PRIOR --weight-decay 0.01";;
  3) NM=sub16;    OUT=$RES/invivo_k100/sub16_sl${Z}_s0;     MA="--model wire_ff_subspace --rank 16 $PRIOR --weight-decay 0.003";;
  4) NM=free;     OUT=$RES/invivo_k100/free_sl${Z}_s0;      MA="--model wire_ff_res --weight-decay 0.01";;
esac
mkdir -p $OUT; echo "task $i: $NM k100 -> $OUT"
$P train_grasp_nik.py $MA --slices $Z --seed 0 --ff-seed 0 --steps 10000 --no-restore \
  --spoke-keep-file spoke_masks/keep_f100.npy \
  --no-compile --resume --save-dir $OUT 2>&1 | tee $OUT/train.log
echo "$(date '+%F %T') TRAIN_DONE $NM k100 exit ${PIPESTATUS[0]}"
