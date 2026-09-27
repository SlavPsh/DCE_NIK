#!/bin/bash
#SBATCH -J p3bw
#SBATCH -p gpu
#SBATCH --gres=gpu:2g.24gb:1
#SBATCH -c 8
#SBATCH --mem 40G
#SBATCH -t 3:00:00
#SBATCH --array=0-1
# fourier-feature sigma test on p3 slice 21 (384 grid, where sigma 2.5 was tuned; the periphery still falls to 0.8 of the all-spoke fine scale):
# 0 sigma 3.5, 1 sigma 5; tofts8 in-coil + prior, seed 0, production protocol otherwise -> results/tofts_vs_patlak/bw_test/ks<s>. task 0 chains the
# eval afterany on the gpu partition: radial sharpness diag + comparison figure against the sigma-2.5 production run, grasp, grasp-pro.
set -uo pipefail
D=/net/beegfs/users/P101440/DCE_NIK; cd $D
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P="micromamba run -n torch29 python -u"
RES=results/tofts_vs_patlak; OUTR=$RES/bw_test; SELF=$D/jobs/queue/098_p3_bandwidth.sh; Z=21
GV=/net/beegfs/users/P101440/grasp_v2/results_grasp_v2; GP=/net/beegfs/users/P101440/grasp_pro_py/results_spoke_cs
echo "$(date '+%F %T') host $(hostname) job ${SLURM_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID:-none} stage ${STAGE:-train}"; nvidia-smi -L || true
ITEMS="tofts8 sigma2.5 (prod):$D/$RES/invivo_prod/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,sigma3.5:$D/$OUTR/ks3.5/nik_slice_${Z}_cplx.npy,sigma5:$D/$OUTR/ks5/nik_slice_${Z}_cplx.npy,GRASP:$GV/gv2_slice${Z}_n12_k80.npy,GRASP-Pro:$GP/cs_slice${Z}_f80match.npy"
if [ "${STAGE:-train}" = eval ]; then
  $P radial_blur_diag.py --slice $Z --tag _bw --items "$ITEMS"; echo "RADIAL exit $?"
  $P compare_runs_fig.py --slice $Z --out $RES/figures/p3_bw_sl$Z.png --title "p3 slice $Z, tofts8 in-coil + prior: fourier sigma 2.5 (production) vs 3.5 vs 5, images at 90 s, kidney zoom, roi curves" --items "$ITEMS"; echo "FIG exit $?"
  exit
fi
i=$SLURM_ARRAY_TASK_ID
if [ "$i" = 0 ]; then
  sbatch --dependency=afterany:$SLURM_ARRAY_JOB_ID --export=ALL,STAGE=eval --array=0 -J p3bw_eval \
    --gres=gpu:1g.12gb:1 -c 4 --mem 32G -t 1:00:00 --output=$D/jobs/log/098_p3_bandwidth_eval_%j.out --error=$D/jobs/log/098_p3_bandwidth_eval_%j.out $SELF && echo "eval chained afterany $SLURM_ARRAY_JOB_ID"
fi
case $i in 0) TAG=ks3.5; KS=3.5;; 1) TAG=ks5; KS=5;; esac
OUT=$OUTR/$TAG; mkdir -p $OUT; echo "task $i: sigma $KS -> $OUT"
$P train_grasp_nik.py --model wire_ff_tofts --tofts-basis $RES/basis_sl${Z}_r8_rms1.npz --support-weight 1 --support-mask spoke_masks/support_sl${Z}_d6.npy --support-orient rot180 --coef-tv-every 32 --k-sigma $KS \
  --slices $Z --seed 0 --ff-seed 0 --steps 10000 --no-restore --weight-decay 0.01 \
  --spoke-keep-file spoke_masks/keep_f80match.npy --spoke-heldout-file spoke_masks/val_k80_m8.npy \
  --no-compile --resume --save-dir $OUT 2>&1 | tee $OUT/train.log
echo "$(date '+%F %T') TRAIN_DONE sigma $KS exit ${PIPESTATUS[0]}"
