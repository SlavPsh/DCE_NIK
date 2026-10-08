#!/bin/bash
#SBATCH -J peak3s
#SBATCH -p gpu
#SBATCH --gres=gpu:2g.24gb:1
#SBATCH -c 8
#SBATCH --mem 40G
#SBATCH -t 3:00:00
#SBATCH --array=0-5
# tofts8 first-pass peak, the two one-seed gains confirmed with 3 seeds at the k100 standard (FINDINGS 9 side findings): 0-2 support prior x20 on the
# residual atoms (seeds 0-2), 3-5 smooth-delay prior 0.1 (seeds 0-2). p3 slice 21, in-coil + prior, wd 1e-2, k100. baseline = invivo_k100/tofts8_sl21_s{0,1,2}.
# -> results/tofts_vs_patlak/peak_levers/{supfast20,dly0.1}/tofts8_sl21_s<S>. task 0 chains the eval: curve tables (3-seed mean +- sd, corrected peaks), iq, figure.
set -uo pipefail
D=/net/beegfs/users/P101440/DCE_NIK; cd $D
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P="micromamba run -n torch29 python -u"; RES=results/tofts_vs_patlak; OUTR=$RES/peak_levers; SELF=$D/jobs/queue/130_peak_levers_3seeds.sh; Z=21; GV=/net/beegfs/users/P101440/grasp_v2/results_grasp_v2
echo "$(date '+%F %T') host $(hostname) job ${SLURM_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID:-none} stage ${STAGE:-train}"; nvidia-smi -L || true
if [ "${STAGE:-train}" = eval ]; then
  for T in supfast20 dly0.1; do $P tofts_eval_invivo.py --spokes k100 --slices $Z --arms tofts8 --iv-dir peak_levers/$T --basis-suffix _rms1 --suffix _peak_$T; $P add_peak_correction.py invivo_peak_$T; done; echo "EVAL exit $?"
  IT="tofts8 k100 s0:$D/$RES/invivo_k100/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,support x20 s0:$D/$OUTR/supfast20/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,support x20 s1:$D/$OUTR/supfast20/tofts8_sl${Z}_s1/nik_slice_${Z}_cplx.npy,delay-tv 0.1 s0:$D/$OUTR/dly0.1/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,delay-tv 0.1 s1:$D/$OUTR/dly0.1/tofts8_sl${Z}_s1/nik_slice_${Z}_cplx.npy,GRASP all spokes:$GV/gv2_slice${Z}_n12.npy"
  IQ_VARIANTS="" IQ_TAG=_peak3s IQ_EXTRA="$IT" $P tofts_iq_track.py --slice $Z; echo "IQ exit $?"
  $P compare_runs_fig.py --slice $Z --out $RES/figures/peak_levers_sl$Z.png --title "p3 slice $Z, k100: tofts8 first-pass levers with seeds (support x20 on the residual atoms, smooth-delay prior 0.1) vs the k100 tofts8 and GRASP" --items "$IT"; echo "FIG exit $?"
  exit
fi
i=$SLURM_ARRAY_TASK_ID; S=$((i % 3)); T=supfast20; X="--support-fast-weight 20"; [ $i -ge 3 ] && { T=dly0.1; X="--delay-tv-weight 0.1"; }
if [ "$i" = 0 ]; then
  sbatch --dependency=afterany:$SLURM_ARRAY_JOB_ID --export=ALL,STAGE=eval --array=0 -J peak3s_eval \
    --gres=gpu:1g.12gb:1 -c 4 --mem 48G -t 2:00:00 --output=$D/jobs/log/130_peak_levers_3seeds_eval_%j.out --error=$D/jobs/log/130_peak_levers_3seeds_eval_%j.out $SELF && echo "eval chained afterany $SLURM_ARRAY_JOB_ID"
fi
OUT=$OUTR/$T/tofts8_sl${Z}_s$S; mkdir -p $OUT; echo "task $i: tofts8 $T seed $S -> $OUT"
$P train_grasp_nik.py --model wire_ff_tofts --tofts-basis $RES/basis_sl${Z}_r8_rms1.npz --support-weight 1 --support-mask spoke_masks/support_sl${Z}_d6.npy --support-orient rot180 --coef-tv-every 32 $X \
  --weight-decay 0.01 --slices $Z --seed $S --ff-seed $S --steps 10000 --no-restore \
  --spoke-keep-file spoke_masks/keep_f100.npy --no-compile --resume --save-dir $OUT 2>&1 | tee $OUT/train.log
echo "$(date '+%F %T') TRAIN_DONE $T s$S exit ${PIPESTATUS[0]}"
