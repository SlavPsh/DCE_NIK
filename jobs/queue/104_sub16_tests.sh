#!/bin/bash
#SBATCH -J sub16t
#SBATCH -p gpu
#SBATCH --gres=gpu:2g.24gb:1
#SBATCH -c 8
#SBATCH --mem 40G
#SBATCH -t 3:00:00
#SBATCH --array=0-6
# sub16 fluctuation tests (plan 2026-10-02, FINDINGS idea 8): p3 slice 21, seed 0, production protocol (10k, final weights, wd 1e-2, support
# prior 1, k80), rank 16 input-coil, sigma 2.5 (384 grid). variants, all into results/tofts_vs_patlak/sub16_tests/<tag> (the production sub16 in
# invivo_prod/sub16_sl21_s0 is untouched and is the baseline): 0 nowarm (--no-warmstart); 1 nocap (pca warm start on 5-spoke frames, no 100 cap);
# 2 toftsinit (first 11 atoms = tofts basis, 5 pca); 3 phitv0.03; 4 phitv0.3 (huber temporal tv on the atoms); 5 w0_10 (atom-net siren w0 10);
# 6 ortho (qr gauge). task 0 chains the eval afterany: atom diagnostic (respiratory content, curve oscillation), curve table, iq track vs the
# late ruler, comparison figure, all against the production sub16 and tofts8.
set -uo pipefail
D=/net/beegfs/users/P101440/DCE_NIK; cd $D
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P="micromamba run -n torch29 python -u"
RES=results/tofts_vs_patlak; OUTR=$RES/sub16_tests; SELF=$D/jobs/queue/104_sub16_tests.sh; Z=21
GV=/net/beegfs/users/P101440/grasp_v2/results_grasp_v2; GP=/net/beegfs/users/P101440/grasp_pro_py/results_spoke_cs
echo "$(date '+%F %T') host $(hostname) job ${SLURM_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID:-none} stage ${STAGE:-train}"; nvidia-smi -L || true
TAGS=(nowarm nocap toftsinit phitv0.03 phitv0.3 w0_10 ortho)
RUNS="sub16 prod (pca warm, 100 fr):$D/$RES/invivo_prod/sub16_sl${Z}_s0,tofts8 prod:$D/$RES/invivo_prod/tofts8_sl${Z}_s0"; ITEMS="sub16 prod:$D/$RES/invivo_prod/sub16_sl${Z}_s0/nik_slice_${Z}_cplx.npy,tofts8 prod:$D/$RES/invivo_prod/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy"
for T in "${TAGS[@]}"; do RUNS="$RUNS,sub16 $T:$D/$OUTR/$T"; ITEMS="$ITEMS,sub16 $T:$D/$OUTR/$T/nik_slice_${Z}_cplx.npy"; done
if [ "${STAGE:-train}" = eval ]; then
  $P sub16_atoms_diag.py --slice $Z --runs "$RUNS" --tag _tests; echo "ATOMS exit $?"
  IQ_VARIANTS="" IQ_TAG=_sub16_tests IQ_EXTRA="$ITEMS" $P tofts_iq_track.py --slice $Z; echo "IQ exit $?"
  $P compare_runs_fig.py --slice $Z --out $RES/figures/sub16_tests_sl$Z.png --title "p3 slice $Z, sub16 fluctuation tests (warm start, atom tv, atom w0, ortho) vs the production sub16 and tofts8: images at 90 s, kidney zoom, roi curves" --items "$ITEMS,GRASP:$GV/gv2_slice${Z}_n12_k80.npy"; echo "FIG exit $?"
  exit
fi
i=$SLURM_ARRAY_TASK_ID; T=${TAGS[$i]}
if [ "$i" = 0 ]; then
  sbatch --dependency=afterany:$SLURM_ARRAY_JOB_ID --export=ALL,STAGE=eval --array=0 -J sub16t_eval \
    --gres=gpu:1g.12gb:1 -c 4 --mem 32G -t 2:00:00 --output=$D/jobs/log/104_sub16_tests_eval_%j.out --error=$D/jobs/log/104_sub16_tests_eval_%j.out $SELF && echo "eval chained afterany $SLURM_ARRAY_JOB_ID"
fi
case $T in
  nowarm)    X="--no-warmstart";;
  nocap)     X="--warmstart-frames 0";;
  toftsinit) X="--warmstart-source tofts --tofts-basis $RES/basis_sl${Z}_r8_rms1.npz";;
  phitv0.03) X="--phi-tv-weight 0.03";;
  phitv0.3)  X="--phi-tv-weight 0.3";;
  w0_10)     X="--phi-w0 10";;
  ortho)     X="--phi-ortho";;
esac
OUT=$OUTR/$T; mkdir -p $OUT; echo "task $i: sub16 $T ($X) -> $OUT"
$P train_grasp_nik.py --model wire_ff_subspace --rank 16 --support-weight 1 --support-mask spoke_masks/support_sl${Z}_d6.npy --support-orient rot180 --coef-tv-every 32 $X \
  --slices $Z --seed 0 --ff-seed 0 --steps 10000 --no-restore --weight-decay 0.01 \
  --spoke-keep-file spoke_masks/keep_f80match.npy --spoke-heldout-file spoke_masks/val_k80_m8.npy \
  --no-compile --resume --save-dir $OUT 2>&1 | tee $OUT/train.log
echo "$(date '+%F %T') TRAIN_DONE sub16 $T exit ${PIPESTATUS[0]}"
