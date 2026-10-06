#!/bin/bash
#SBATCH -J freets
#SBATCH -p gpu
#SBATCH --gres=gpu:2g.24gb:1
#SBATCH -c 8
#SBATCH --mem 40G
#SBATCH -t 3:00:00
#SBATCH --array=0-2
# nik-free temporal bandwidth (FINDINGS 9 follow-up): the time input is encoded with fourier features of sigma 1.5 cycles per unit t (375 s = 2 units),
# i.e. periods of ~125 s typical and little below 45 s, while the first pass is a 5 to 10 s rise; the aorta peak of nik-free is 0.6 to 0.7 of the
# corrected reference. the old "> 3 kills dynamics" note was judged on held-out k-space under an older protocol. retest under the production
# nik-free protocol (wire_ff_res, wd 1e-2, 10k, final weights, k80, p3 slice 21, seed 0): t_sigma 2.5 / 4 / 6 vs the production 1.5.
# -> results/tofts_vs_patlak/free_tsigma/ts<v>/free_sl21_s0. task 0 chains the eval: curve table with corrected peaks, arrival-window metrics, iq vs the late ruler, figure.
set -uo pipefail
D=/net/beegfs/users/P101440/DCE_NIK; cd $D
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P="micromamba run -n torch29 python -u"
RES=results/tofts_vs_patlak; OUTR=$RES/free_tsigma; SELF=$D/jobs/queue/120_free_tsigma.sh; Z=21; GV=/net/beegfs/users/P101440/grasp_v2/results_grasp_v2
echo "$(date '+%F %T') host $(hostname) job ${SLURM_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID:-none} stage ${STAGE:-train}"; nvidia-smi -L || true
TS=(2.5 4 6); BASE=$D/$RES/invivo_prod/free_sl${Z}_s0; T8=$D/$RES/invivo_prod/tofts8_sl${Z}_s0
if [ "${STAGE:-train}" = eval ]; then
  IT="NIK-free t_sigma 1.5 (prod):$BASE/nik_slice_${Z}_cplx.npy"
  for V in "${TS[@]}"; do T=ts$V
    $P tofts_eval_invivo.py --spokes k80 --slices $Z --arms free --iv-dir free_tsigma/$T --basis-suffix _rms1 --suffix _freets_$T; $P add_peak_correction.py invivo_freets_$T
    IT="$IT,NIK-free t_sigma $V:$D/$OUTR/$T/free_sl${Z}_s0/nik_slice_${Z}_cplx.npy"
  done; echo "EVAL exit $?"
  $P arrival_artifact_diag.py --slice $Z --tag _freets --items "$IT,tofts8 prod:$T8/nik_slice_${Z}_cplx.npy,GRASP:$GV/gv2_slice${Z}_n12_k80.npy"; echo "DIAG exit $?"
  IQ_VARIANTS="" IQ_TAG=_freets IQ_EXTRA="$IT" $P tofts_iq_track.py --slice $Z; echo "IQ exit $?"
  $P compare_runs_fig.py --slice $Z --out $RES/figures/free_tsigma_sl$Z.png --title "p3 slice $Z, NIK-free temporal bandwidth: t_sigma 1.5 (production) vs 2.5 / 4 / 6; tofts8 and GRASP for scale" --items "$IT,tofts8 prod:$T8/nik_slice_${Z}_cplx.npy,GRASP:$GV/gv2_slice${Z}_n12_k80.npy"; echo "FIG exit $?"
  exit
fi
i=$SLURM_ARRAY_TASK_ID; V=${TS[$i]}; T=ts$V
if [ "$i" = 0 ]; then
  sbatch --dependency=afterany:$SLURM_ARRAY_JOB_ID --export=ALL,STAGE=eval --array=0 -J freets_eval \
    --gres=gpu:1g.12gb:1 -c 4 --mem 48G -t 3:00:00 --output=$D/jobs/log/120_free_tsigma_eval_%j.out --error=$D/jobs/log/120_free_tsigma_eval_%j.out $SELF && echo "eval chained afterany $SLURM_ARRAY_JOB_ID"
fi
OUT=$OUTR/$T/free_sl${Z}_s0; mkdir -p $OUT; echo "task $i: nik-free t_sigma $V -> $OUT"
$P train_grasp_nik.py --model wire_ff_res --t-sigma $V --weight-decay 0.01 --slices $Z --seed 0 --ff-seed 0 --steps 10000 --no-restore \
  --spoke-keep-file spoke_masks/keep_f80match.npy --spoke-heldout-file spoke_masks/val_k80_m8.npy \
  --no-compile --resume --save-dir $OUT 2>&1 | tee $OUT/train.log
echo "$(date '+%F %T') TRAIN_DONE free $T exit ${PIPESTATUS[0]}"
