#!/bin/bash
#SBATCH -J schedctl
#SBATCH -p gpu
#SBATCH --gres=gpu:2g.24gb:1
#SBATCH -c 8
#SBATCH --mem 40G
#SBATCH -t 3:00:00
#SBATCH --array=0-1
# scheduler control for the k100 test (FINDINGS 10): k100 changed the spokes AND the plateau scheduler signal (train loss instead of held-out).
# same k80 input as production but the scheduler on the running train loss (--sched-on train): 0 tofts8 in-coil + prior, 1 nik-free. p3 slice 21,
# seed 0 -> results/tofts_vs_patlak/sched_control/<arm>_sl21_s0. task 0 chains the eval: curve tables, iq track and figure vs the k80 production and k100 runs.
set -uo pipefail
D=/net/beegfs/users/P101440/DCE_NIK; cd $D
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P="micromamba run -n torch29 python -u"
RES=results/tofts_vs_patlak; OUTR=$RES/sched_control; SELF=$D/jobs/queue/122_sched_control.sh; Z=21; GV=/net/beegfs/users/P101440/grasp_v2/results_grasp_v2
echo "$(date '+%F %T') host $(hostname) job ${SLURM_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID:-none} stage ${STAGE:-train}"; nvidia-smi -L || true
if [ "${STAGE:-train}" = eval ]; then
  $P tofts_eval_invivo.py --spokes k80 --slices $Z --arms tofts8,free --iv-dir sched_control --basis-suffix _rms1 --suffix _schedctl; $P add_peak_correction.py invivo_schedctl; echo "EVAL exit $?"
  IT="tofts8 k80 prod (sched heldout):$D/$RES/invivo_prod/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,tofts8 k80 sched train:$D/$OUTR/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,tofts8 k100:$D/$RES/invivo_k100/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,NIK-free k80 prod:$D/$RES/invivo_prod/free_sl${Z}_s0/nik_slice_${Z}_cplx.npy,NIK-free k80 sched train:$D/$OUTR/free_sl${Z}_s0/nik_slice_${Z}_cplx.npy,NIK-free k100:$D/$RES/invivo_k100/free_sl${Z}_s0/nik_slice_${Z}_cplx.npy"
  $P arrival_artifact_diag.py --slice $Z --tag _schedctl --items "$IT"; echo "DIAG exit $?"
  IQ_VARIANTS="" IQ_TAG=_schedctl IQ_EXTRA="$IT" $P tofts_iq_track.py --slice $Z; echo "IQ exit $?"
  $P compare_runs_fig.py --slice $Z --out $RES/figures/sched_control_sl$Z.png --title "p3 slice $Z, scheduler control: k80 with the plateau scheduler on held-out (production) vs on the train loss vs k100, tofts8 and NIK-free" --items "$IT,GRASP:$GV/gv2_slice${Z}_n12_k80.npy"; echo "FIG exit $?"
  exit
fi
i=$SLURM_ARRAY_TASK_ID
if [ "$i" = 0 ]; then
  sbatch --dependency=afterany:$SLURM_ARRAY_JOB_ID --export=ALL,STAGE=eval --array=0 -J schedctl_eval \
    --gres=gpu:1g.12gb:1 -c 4 --mem 48G -t 2:00:00 --output=$D/jobs/log/122_sched_control_eval_%j.out --error=$D/jobs/log/122_sched_control_eval_%j.out $SELF && echo "eval chained afterany $SLURM_ARRAY_JOB_ID"
fi
PRIOR="--support-weight 1 --support-mask spoke_masks/support_sl${Z}_d6.npy --support-orient rot180 --coef-tv-every 32"
case $i in
  0) NM=tofts8; OUT=$OUTR/tofts8_sl${Z}_s0; MA="--model wire_ff_tofts --tofts-basis $RES/basis_sl${Z}_r8_rms1.npz $PRIOR";;
  1) NM=free;   OUT=$OUTR/free_sl${Z}_s0;   MA="--model wire_ff_res";;
esac
mkdir -p $OUT; echo "task $i: $NM k80, scheduler on the train loss -> $OUT"
$P train_grasp_nik.py $MA --sched-on train --weight-decay 0.01 --slices $Z --seed 0 --ff-seed 0 --steps 10000 --no-restore \
  --spoke-keep-file spoke_masks/keep_f80match.npy --spoke-heldout-file spoke_masks/val_k80_m8.npy \
  --no-compile --resume --save-dir $OUT 2>&1 | tee $OUT/train.log
echo "$(date '+%F %T') TRAIN_DONE $NM schedctl exit ${PIPESTATUS[0]}"
