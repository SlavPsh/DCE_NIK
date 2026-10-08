#!/bin/bash
#SBATCH -J jointz
#SBATCH -p gpu
#SBATCH --gres=gpu:2g.24gb:1
#SBATCH -c 8
#SBATCH --mem 48G
#SBATCH -t 7:00:00
#SBATCH --array=0-9
# joint-z nik-tofts (plan item 3; train_joint_z.py, new class JointZTofts, production trainer untouched), p3, k100 standard, one shared basis (slice 19)
# for every run, seed 0, in-coil, support prior 1 per z, wd 1e-2, sigma 2.5, compute matched (one micro-batch per slice per step):
#  0 joint {18,19} query 20, z_sigma 1   | 1 joint {18,19,21} query 20, z_sigma 1 | 2 triplet z_sigma 0.5 | 3 triplet z_sigma 2 | 4 triplet categorical embedding (no z smoothness)
#  5 triplet wide (hidden 896, ~3x params) | 6-9 single-slice tofts8 with the SAME shared basis on 18 / 19 / 21 / 20 (the fair baselines; slice 20 = the interpolation reference)
# -> results/tofts_vs_patlak/jointz/<tag>/tofts8_sl<Z>_s0/. task 0 runs the prep first (slice 20: precompute, rulers, model-free series, support mask, grasp all spokes),
# the other tasks wait for jointz/.prep_done. task 0 chains the eval: curve tables (corrected peaks, slice 20 peak check), iq tracks vs the late ruler, per-slice figures.
set -uo pipefail
D=/net/beegfs/users/P101440/DCE_NIK; cd $D
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH DCE_DS=p3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P="micromamba run -n torch29 python -u"; RES=results/tofts_vs_patlak; OUTR=$RES/jointz; SELF=$D/jobs/queue/132_jointz.sh; REFD=/net/beegfs/users/P101440/grasp_pro_py/results_ref
GV=/net/beegfs/users/P101440/grasp_v2/results_grasp_v2; BASIS=$RES/basis_sl19_r8_rms1.npz
echo "$(date '+%F %T') host $(hostname) job ${SLURM_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID:-none} stage ${STAGE:-train}"; nvidia-smi -L || true
TAGS=(pair18_19 trip zs0.5 zs2 embed wide single single single single); mkdir -p $OUTR
if [ "${STAGE:-train}" = eval ]; then
  $P mf_peak_check.py --slice 20 --windows 31,21,15,11 || true
  for T in pair18_19 trip zs0.5 zs2 embed wide single; do SLS=18,19,21,20; [ $T = pair18_19 ] && SLS=18,19,20
    $P tofts_eval_invivo.py --spokes k100 --slices $SLS --arms tofts8 --iv-dir jointz/$T --basis-suffix _rms1 --suffix _jointz_$T; $P add_peak_correction.py invivo_jointz_$T; done; echo "EVAL exit $?"
  for Z in 18 19 21 20; do
    IT="single shared basis:$D/$OUTR/single/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,joint 18+19+21 zs1:$D/$OUTR/trip/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,joint zs0.5:$D/$OUTR/zs0.5/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,joint zs2:$D/$OUTR/zs2/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,joint embed:$D/$OUTR/embed/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,joint wide:$D/$OUTR/wide/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,GRASP all spokes:$GV/gv2_slice${Z}_n12.npy"
    [ -f $OUTR/pair18_19/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy ] && IT="joint 18+19 zs1:$D/$OUTR/pair18_19/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,$IT"
    [ $Z != 20 ] && IT="tofts8 k100 own basis:$D/$RES/invivo_k100/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,$IT"
    IQ_VARIANTS="" IQ_TAG=_jointz_sl$Z IQ_EXTRA="$IT" $P tofts_iq_track.py --slice $Z
    NOTE="trained"; [ $Z = 20 ] && NOTE="NEVER TRAINED ON by the joint models (interpolation test)"
    $P compare_runs_fig.py --slice $Z --out $RES/figures/jointz_sl$Z.png --title "p3 slice $Z ($NOTE), k100: joint-z tofts8 (shared basis sl19) vs single-slice tofts8; images at 90 s, kidney zoom, roi curves" --items "$IT"
  done; echo "FIGS exit $?"
  exit
fi
i=$SLURM_ARRAY_TASK_ID; T=${TAGS[$i]}
if [ "$i" = 0 ]; then
  [ -f $REFD/slice_20.npz ] || (cd /net/beegfs/users/P101440/grasp_pro_py && $P precompute_ref.py --file /net/beegfs/users/P101440/dce_data/orig/meas_p3_dce.dat --out results_ref --slices 20)
  [ -d results_nufft_slice20 ] && [ -f results_nufft_slice20/nufft_late.npy ] || $P build_rulers.py 20
  [ -f step2_slice20.npz ] || $P step2_kidney.py 20
  [ -f spoke_masks/support_sl20_d6.npy ] || $P support_mask.py --slices 20 --dilate 6
  [ -f $GV/gv2_slice20_n12.npy ] || (cd /net/beegfs/users/P101440/grasp_v2 && SET=sweep NLINE=12 LAM_FRAC=0.25 SLICES=20 OMP_NUM_THREADS=8 $P grasp_v2_real.py | grep -E "SAVED|cached|DONE")
  ls -la $REFD/slice_20.npz results_nufft_slice20/nufft_late.npy step2_slice20.npz spoke_masks/support_sl20_d6.npy $GV/gv2_slice20_n12.npy && touch $OUTR/.prep_done; echo "PREP exit $?"
  sbatch --dependency=afterany:$SLURM_ARRAY_JOB_ID --export=ALL,STAGE=eval --array=0 -J jointz_eval \
    --gres=gpu:1g.12gb:1 -c 4 --mem 48G -t 4:00:00 --output=$D/jobs/log/132_jointz_eval_%j.out --error=$D/jobs/log/132_jointz_eval_%j.out $SELF && echo "eval chained afterany $SLURM_ARRAY_JOB_ID"
else
  for k in $(seq 1 360); do [ -f $OUTR/.prep_done ] && break; sleep 30; done; [ -f $OUTR/.prep_done ] || { echo "prep not done"; exit 3; }
fi
if [ $i -le 5 ]; then
  case $i in 0) SLS=18,19; X="--z-sigma 1";; 1) SLS=18,19,21; X="--z-sigma 1";; 2) SLS=18,19,21; X="--z-sigma 0.5";; 3) SLS=18,19,21; X="--z-sigma 2";; 4) SLS=18,19,21; X="--z-mode embed";; 5) SLS=18,19,21; X="--z-sigma 1 --hidden 896";; esac
  OUT=$OUTR/$T; mkdir -p $OUT; echo "task $i: joint-z $T slices $SLS ($X) -> $OUT"
  $P train_joint_z.py --slices $SLS --query-slices 20 --basis $BASIS --out $OUT $X --steps 10000 --batch-size 65536 --weight-decay 0.01 --seed 0 --ff-seed 0 \
    --spoke-keep-file spoke_masks/keep_f100.npy --support-weight 1 --support-every 32 2>&1 | tee $OUT/train.log
  echo "$(date '+%F %T') TRAIN_DONE jointz $T exit ${PIPESTATUS[0]}"
else
  ZS=(18 19 21 20); Z=${ZS[$((i-6))]}; OUT=$OUTR/single/tofts8_sl${Z}_s0; mkdir -p $OUT; echo "task $i: single-slice tofts8 shared basis slice $Z -> $OUT"
  $P train_grasp_nik.py --model wire_ff_tofts --tofts-basis $BASIS --support-weight 1 --support-mask spoke_masks/support_sl${Z}_d6.npy --support-orient rot180 --coef-tv-every 32 --weight-decay 0.01 \
    --slices $Z --seed 0 --ff-seed 0 --steps 10000 --no-restore --spoke-keep-file spoke_masks/keep_f100.npy --no-compile --resume --save-dir $OUT 2>&1 | tee $OUT/train.log
  echo "$(date '+%F %T') TRAIN_DONE single sl$Z exit ${PIPESTATUS[0]}"
fi
