#!/bin/bash
#SBATCH -J pf13
#SBATCH -p gpu
#SBATCH --gres=gpu:2g.24gb:1
#SBATCH -c 8
#SBATCH --mem 64G
#SBATCH -t 4:00:00
#SBATCH --array=0-4
# per-model rule check of the 129 candidate (patlak span + 13 free atoms beat sub16 on p3 slice 21: aorta peak 0.54 -> 0.68, ripple gone, HaarPSI 0.776 -> 0.815):
# the same arm on p3 18 / 19 (sigma 2.5) and p14 21 / 24 / 27 (sigma 5), k100, seed 0, sub16 protocol (wd 3e-3 + prior, no pca warm start), own aif per slice.
# -> results/tofts_vs_patlak/aif_feat/sub16_sl<Z>_s0 (p3) and p14/aif_feat/sub16_sl<Z>_s0. task 0 chains the eval: curve tables with corrected peaks
# (invivo_aiffeat3.md p3 18/19/21, invivo_p14_aiffeat.md), arrival windows, iq vs the late ruler, panel per slice vs sub16 / tofts8 k100 and grasp all spokes.
set -uo pipefail
D=/net/beegfs/users/P101440/DCE_NIK; cd $D
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P="micromamba run -n torch29 python -u"; RES=results/tofts_vs_patlak; SELF=$D/jobs/queue/134_patlak13_confirm.sh
GV3=/net/beegfs/users/P101440/grasp_v2/results_grasp_v2; GV14=/net/beegfs/users/P101440/grasp_v2/results_grasp_v2_p14; REFD14=/net/beegfs/users/P101440/grasp_pro_py/results_ref_p14
echo "$(date '+%F %T') host $(hostname) job ${SLURM_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID:-none} stage ${STAGE:-train}"; nvidia-smi -L || true
if [ "${STAGE:-train}" = eval ]; then
  for DS in p3 p14; do
    if [ $DS = p3 ]; then SLS="18 19 21"; IV=aif_feat; SUF=_aiffeat3; R=$D/$RES; GV=$GV3; else SLS="21 24 27"; IV=p14/aif_feat; SUF=_p14_aiffeat; R=$D/$RES/p14; GV=$GV14; fi
    export DCE_DS=$DS
    $P tofts_eval_invivo.py --spokes k100 --slices $(echo $SLS | tr ' ' ',') --arms sub16 --iv-dir $IV --basis-suffix _rms1 --suffix $SUF; $P add_peak_correction.py invivo$SUF; echo "EVAL $DS exit $?"
    for Z in $SLS; do
      IT="patlak span + 13 free atoms:$R/aif_feat/sub16_sl${Z}_s0/nik_slice_${Z}_cplx.npy,sub16 (wd 3e-3) k100:$R/invivo_k100/sub16_sl${Z}_s0/nik_slice_${Z}_cplx.npy,tofts8 k100:$R/invivo_k100/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,NIK-free k100:$R/invivo_k100/free_sl${Z}_s0/nik_slice_${Z}_cplx.npy,GRASP all spokes:$GV/gv2_slice${Z}_n12.npy"
      $P arrival_artifact_diag.py --slice $Z --tag _aiffeat --items "$IT"; echo "DIAG $DS $Z exit $?"
      IQ_VARIANTS="" IQ_TAG=${SUF}_sl$Z IQ_EXTRA="$IT" $P tofts_iq_track.py --slice $Z; echo "IQ $DS $Z exit $?"
      $P compare_runs_fig.py --slice $Z --out $RES/figures/aif_feat_${DS}_sl$Z.png --title "$DS slice $Z, k100: patlak span + 13 free atoms (sub16 protocol) vs sub16, tofts8, nik-free and GRASP all spokes; images at 90 s, tissue zoom, roi curves" --items "$IT"
    done
  done; echo "FIG exit $?"; exit
fi
i=$SLURM_ARRAY_TASK_ID
if [ "$i" = 0 ]; then
  sbatch --dependency=afterany:$SLURM_ARRAY_JOB_ID --export=ALL,STAGE=eval --array=0 -J pf13_eval \
    --gres=gpu:1g.12gb:1 -c 4 --mem 64G -t 4:00:00 --output=$D/jobs/log/134_patlak13_confirm_eval_%j.out --error=$D/jobs/log/134_patlak13_confirm_eval_%j.out $SELF && echo "eval chained afterany $SLURM_ARRAY_JOB_ID"
fi
if [ $i -lt 2 ]; then ZS=(18 19); Z=${ZS[$i]}; export DCE_DS=p3; OUT=$RES/aif_feat/sub16_sl${Z}_s0; AIF=$D/aif_slice$Z.npz; SUP=spoke_masks/support_sl${Z}_d6.npy; KEEP=spoke_masks/keep_f100.npy; X=""
else ZS=(21 24 27); Z=${ZS[$((i-2))]}; export DCE_DS=p14; OUT=$RES/p14/aif_feat/sub16_sl${Z}_s0; AIF=$D/aif_p14_slice$Z.npz; SUP=spoke_masks/support_p14_sl${Z}_d6.npy; KEEP=spoke_masks/keep_f100_p14.npy; X="--k-sigma 5 --out-dir $REFD14"; fi
[ -f $OUT/nik_slice_${Z}_cplx.npy ] && { echo "exists $OUT"; exit 0; }
mkdir -p $OUT; echo "task $i: patlak + 13 free, $DCE_DS slice $Z -> $OUT"
$P train_grasp_nik.py --model wire_ff_patlak --aif-file $AIF --patlak-free 13 --weight-decay 0.003 --support-weight 1 --support-mask $SUP --support-orient rot180 --coef-tv-every 32 $X \
  --slices $Z --seed 0 --ff-seed 0 --steps 10000 --no-restore --spoke-keep-file $KEEP --no-compile --resume --save-dir $OUT 2>&1 | tee $OUT/train.log
echo "$(date '+%F %T') TRAIN_DONE patlak13 $DCE_DS sl$Z exit ${PIPESTATUS[0]}"
