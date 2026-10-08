#!/bin/bash
#SBATCH -J phwd
#SBATCH -p gpu
#SBATCH --gres=gpu:2g.24gb:1
#SBATCH -c 8
#SBATCH --mem 32G
#SBATCH -t 6:00:00
#SBATCH --array=0-4
# plan item 2 (first-pass peak), the truth side: per-atom weight decay on the tofts amplitude head (xph_train.py --wd-fast, new flags, defaults
# unchanged), tested on the no-motion phantom (truth curves, no reference clipping). phantom standard = tofts16 w768 ks2.5, 40k steps x batch 16384
# (= the in vivo 10k x 65536 samples), best-VAL ckpt, unit-norm atoms, wd 3e-3 (phantom_nomotion.json seeds 0-2 = baseline, not rerun). new arms,
# seed 0, unit-rms atoms (the in vivo standard, basis_xph_rms1.npz): 0 wd 3e-3 | 1 wd 1e-2 (the in vivo wd) | 2 wd 1e-2 + wd-fast 0 |
# 3 wd 1e-2 + wd-fast 1e-3 | 4 wd 3e-3 + wd-fast 0. each task trains then runs xph_eval (prunes to best + last ckpt); task 0 chains a cpu stage:
# phantom_peak_levers.py -> results/tofts_vs_patlak/phantom_peak_levers.md + figures/phantom_peak_levers.png (peak ratios vs truth, curves, image metrics).
# differences to the in vivo tofts8 protocol, stated: rank 16 not 8, no support prior, oracle aif, 5 spokes per frame, no motion.
set -uo pipefail
D=/net/beegfs/users/P101440/DCE_NIK; cd $D
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True XPH_SIM=nomotion
P="micromamba run -n torch29 python -u"; RES=results/tofts_vs_patlak; SELF=$D/jobs/queue/133_phantom_peak_wd.sh
echo "$(date '+%F %T') host $(hostname) job ${SLURM_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID:-none} stage ${STAGE:-train}"; nvidia-smi -L || true
TAGS=(rms1_wd3e-3 rms1_wd1e-2 rms1_wd1e-2_wdf0 rms1_wd1e-2_wdf1e-3 rms1_wd3e-3_wdf0); B=w768_ks2.5_s0_tofts16
if [ "${STAGE:-train}" = cpu ]; then
  IT="phantom standard s0 (unit-norm atoms wd 3e-3):$B,phantom standard s1:w768_ks2.5_s1_tofts16,phantom standard s2:w768_ks2.5_s2_tofts16"
  for T in "${TAGS[@]}"; do IT="$IT,$T:${B}_$T"; done
  $P phantom_peak_levers.py --items "$IT" --out $RES/phantom_peak_levers.md --fig $RES/figures/phantom_peak_levers.png \
    --title "phantom no-motion z15, tofts16 w768 ks2.5 40k steps best-VAL: atom scale (unit-norm vs unit-rms), wd 3e-3 vs 1e-2, per-atom wd on the residual atoms (wdf) vs truth"; echo "LEVERS exit $?"; exit
fi
i=$SLURM_ARRAY_TASK_ID; T=${TAGS[$i]}
$P basis_rms1.py $RES/basis_xph.npz
if [ "$i" = 0 ]; then
  sbatch --dependency=afterany:$SLURM_ARRAY_JOB_ID --export=ALL,STAGE=cpu --array=0 -J phwd_cpu -p defq --gres="" -c 4 --mem 24G -t 1:00:00 \
    --output=$D/jobs/log/133_phantom_peak_wd_cpu_%j.out --error=$D/jobs/log/133_phantom_peak_wd_cpu_%j.out $SELF && echo "cpu stage chained afterany $SLURM_ARRAY_JOB_ID"
fi
case $i in 0) X="--weight-decay 0.003";; 1) X="--weight-decay 0.01";; 2) X="--weight-decay 0.01 --wd-fast 0";; 3) X="--weight-decay 0.01 --wd-fast 0.001";; 4) X="--weight-decay 0.003 --wd-fast 0";; esac
echo "task $i: $T ($X)"
$P xph_train.py --hidden-width 768 --k-sigma 2.5 --seed 0 --model wire_ff_tofts --basis-file $D/$RES/basis_xph_rms1.npz $X --tag-suffix _$T && $P xph_eval.py --tag ${B}_$T
echo "$(date '+%F %T') PH $T exit $?"
