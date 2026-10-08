#!/bin/bash
#SBATCH -J aiffeat
#SBATCH -p gpu
#SBATCH --gres=gpu:2g.24gb:1
#SBATCH -c 8
#SBATCH --mem 40G
#SBATCH -t 3:00:00
#SBATCH --array=0-1
# give the free-basis arms the aif (FINDINGS next steps 9), k100 standard, p3 slice 21, seed 0, each at its own protocol:
# 0 nik-free + aif features: new class wire_ff_res_aif (aif(t) and its integral appended to the time encoding; the production nik-free class is untouched), wd 1e-2, no prior
# 1 sub16 as patlak span + 13 free atoms: --model wire_ff_patlak --patlak-free 13 (exists), wd 3e-3 + prior (sub16 protocol), no pca warm start
# -> results/tofts_vs_patlak/aif_feat/{free,sub16}_sl21_s0. task 0 chains the eval: curve table with corrected peaks, arrival windows, iq vs the late ruler, figure vs the k100 free / sub16 / tofts8 / grasp.
set -uo pipefail
D=/net/beegfs/users/P101440/DCE_NIK; cd $D
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P="micromamba run -n torch29 python -u"; RES=results/tofts_vs_patlak; OUTR=$RES/aif_feat; SELF=$D/jobs/queue/129_aif_features.sh; Z=21; GV=/net/beegfs/users/P101440/grasp_v2/results_grasp_v2
echo "$(date '+%F %T') host $(hostname) job ${SLURM_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID:-none} stage ${STAGE:-train}"; nvidia-smi -L || true
K=$D/$RES/invivo_k100
if [ "${STAGE:-train}" = eval ]; then
  $P tofts_eval_invivo.py --spokes k100 --slices $Z --arms free,sub16 --iv-dir aif_feat --basis-suffix _rms1 --suffix _aiffeat; $P add_peak_correction.py invivo_aiffeat; echo "EVAL exit $?"
  IT="NIK-free + aif features:$D/$OUTR/free_sl${Z}_s0/nik_slice_${Z}_cplx.npy,NIK-free k100:$K/free_sl${Z}_s0/nik_slice_${Z}_cplx.npy,patlak span + 13 free atoms:$D/$OUTR/sub16_sl${Z}_s0/nik_slice_${Z}_cplx.npy,sub16 (wd 3e-3) k100:$K/sub16_sl${Z}_s0/nik_slice_${Z}_cplx.npy,tofts8 k100:$K/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,GRASP all spokes:$GV/gv2_slice${Z}_n12.npy"
  $P arrival_artifact_diag.py --slice $Z --tag _aiffeat --items "$IT"; echo "DIAG exit $?"
  IQ_VARIANTS="" IQ_TAG=_aiffeat IQ_EXTRA="$IT" $P tofts_iq_track.py --slice $Z; echo "IQ exit $?"
  $P compare_runs_fig.py --slice $Z --out $RES/figures/aif_feat_sl$Z.png --title "p3 slice $Z, k100: free-basis arms given the aif (nik-free + aif features; patlak span + 13 free atoms) vs their k100 baselines, tofts8 and GRASP" --items "$IT"; echo "FIG exit $?"
  exit
fi
i=$SLURM_ARRAY_TASK_ID
if [ "$i" = 0 ]; then
  sbatch --dependency=afterany:$SLURM_ARRAY_JOB_ID --export=ALL,STAGE=eval --array=0 -J aiffeat_eval \
    --gres=gpu:1g.12gb:1 -c 4 --mem 48G -t 2:00:00 --output=$D/jobs/log/129_aif_features_eval_%j.out --error=$D/jobs/log/129_aif_features_eval_%j.out $SELF && echo "eval chained afterany $SLURM_ARRAY_JOB_ID"
fi
PRIOR="--support-weight 1 --support-mask spoke_masks/support_sl${Z}_d6.npy --support-orient rot180 --coef-tv-every 32"
if [ $i = 0 ]; then NM=free;  OUT=$OUTR/free_sl${Z}_s0;  MA="--model wire_ff_res_aif --aif-file $D/aif_slice$Z.npz --weight-decay 0.01"
else                NM=sub16; OUT=$OUTR/sub16_sl${Z}_s0; MA="--model wire_ff_patlak --aif-file $D/aif_slice$Z.npz --patlak-free 13 $PRIOR --weight-decay 0.003"; fi
mkdir -p $OUT; echo "task $i: $NM with aif ($MA) -> $OUT"
$P train_grasp_nik.py $MA --slices $Z --seed 0 --ff-seed 0 --steps 10000 --no-restore \
  --spoke-keep-file spoke_masks/keep_f100.npy --no-compile --resume --save-dir $OUT 2>&1 | tee $OUT/train.log
echo "$(date '+%F %T') TRAIN_DONE $NM aif exit ${PIPESTATUS[0]}"
