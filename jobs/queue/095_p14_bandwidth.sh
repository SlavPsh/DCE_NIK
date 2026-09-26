#!/bin/bash
#SBATCH -J p14bw
#SBATCH -p gpu
#SBATCH --gres=gpu:2g.24gb:1
#SBATCH -c 8
#SBATCH --mem 64G
#SBATCH -t 4:00:00
#SBATCH --array=0-3
# blur away from the image centre (radial_blur_sl*_prod.md: nik fine-scale energy vs cs100 falls from 1.1 at the centre to 0.6 to 0.75 at 112-128 px on
# p14, 0.8 to 0.9 at 80-96 px on p3; all backbones alike, intensity flat, so not the support prior): the k-space representation must oscillate x_px/2
# cycles per unit coordinate for content x_px from the centre, and the fourier-feature sigma (2.5) and wire w0 (62) are fixed cycles per unit, not per
# grid. test on p14 slice 24, tofts8 in-coil + prior, seed 0, production protocol otherwise: 0 k-sigma 3.5; 1 k-sigma 5; 2 w0 90; 3 k-sigma 3.5 + w0 90.
# task 0 chains the eval afterany: radial diag + comparison figure (curves, haarpsi, air) against the production baseline (k-sigma 2.5, w0 62).
set -uo pipefail
D=/net/beegfs/users/P101440/DCE_NIK; cd $D
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH DCE_DS=p14
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P="micromamba run -n torch29 python -u"
RES=results/tofts_vs_patlak; OUTR=$RES/p14/bw_test; SELF=$D/jobs/queue/095_p14_bandwidth.sh; REFD=/net/beegfs/users/P101440/grasp_pro_py/results_ref_p14; Z=24
GV=/net/beegfs/users/P101440/grasp_v2/results_grasp_v2_p14; GP=/net/beegfs/users/P101440/grasp_pro_py/results_spoke_cs_p14
echo "$(date '+%F %T') host $(hostname) job ${SLURM_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID:-none} stage ${STAGE:-train}"; nvidia-smi -L || true
BASE=$D/$RES/p14/invivo_prod/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy
ITEMS="tofts8 base (ks2.5 w62):$BASE,ks3.5:$D/$OUTR/ks3.5/nik_slice_${Z}_cplx.npy,ks5:$D/$OUTR/ks5/nik_slice_${Z}_cplx.npy,w90:$D/$OUTR/w90/nik_slice_${Z}_cplx.npy,ks3.5 w90:$D/$OUTR/ks3.5_w90/nik_slice_${Z}_cplx.npy,GRASP:$GV/gv2_slice${Z}_n12_k80.npy,GRASP-Pro:$GP/cs_slice${Z}_f80match.npy"
if [ "${STAGE:-train}" = eval ]; then
  $P radial_blur_diag.py --slice $Z --tag _bw --items "$ITEMS"; echo "RADIAL exit $?"
  $P compare_runs_fig.py --slice $Z --out $RES/figures/p14_bw_sl$Z.png --title "p14 slice $Z, tofts8 in-coil + prior: fourier-feature sigma / wire w0 test (images at 90 s, liver zoom, roi curves)" --items "$ITEMS"; echo "FIG exit $?"
  exit
fi
i=$SLURM_ARRAY_TASK_ID
if [ "$i" = 0 ]; then
  sbatch --dependency=afterany:$SLURM_ARRAY_JOB_ID --export=ALL,STAGE=eval --array=0 -J p14bw_eval -p defq -c 4 --mem 32G -t 1:00:00 \
    --output=$D/jobs/log/095_p14_bandwidth_eval_%j.out --error=$D/jobs/log/095_p14_bandwidth_eval_%j.out $SELF && echo "eval chained afterany $SLURM_ARRAY_JOB_ID"
fi
case $i in 0) TAG=ks3.5; X="--k-sigma 3.5";; 1) TAG=ks5; X="--k-sigma 5";; 2) TAG=w90; X="--w0 90";; 3) TAG=ks3.5_w90; X="--k-sigma 3.5 --w0 90";; esac
OUT=$OUTR/$TAG; mkdir -p $OUT; echo "task $i: $TAG -> $OUT"
$P train_grasp_nik.py --model wire_ff_tofts --tofts-basis $RES/basis_p14_sl${Z}_r8_rms1.npz --support-weight 1 --support-mask spoke_masks/support_p14_sl${Z}_d6.npy --support-orient rot180 --coef-tv-every 32 $X \
  --slices $Z --seed 0 --ff-seed 0 --steps 10000 --no-restore --weight-decay 0.01 --out-dir $REFD \
  --spoke-keep-file spoke_masks/keep_k80_p14.npy --spoke-heldout-file spoke_masks/val_k80_m8_p14.npy \
  --no-compile --resume --save-dir $OUT 2>&1 | tee $OUT/train.log
echo "$(date '+%F %T') TRAIN_DONE $TAG exit ${PIPESTATUS[0]}"
