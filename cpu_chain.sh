#!/usr/bin/env bash
# Serial CPU orchestrator for Task 4 non-neural recons (avoid thrash on the loaded node).
# Order: [running] direct+nufft f25 -> csfit f25 -> direct+nufft f100 -> csfit f100
cd /net/beegfs/users/P101440/DCE_NIK
L=results/task4_xcat_nomotion_pilot/logs
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
run() { echo "[chain] $(date +%H:%M) START $1"; micromamba run -n torch29 python -u $2 > $L/$1.log 2>&1; echo "[chain] $(date +%H:%M) END   $1 ($(grep -c DONE $L/$1.log) done-marker)"; }

# 1) wait for the already-running direct+nufft f25
until grep -qE "DONE|Error|Traceback|Killed" $L/direct_f25.log 2>/dev/null; do sleep 15; done
echo "[chain] direct_f25 finished: $(grep -iE 'DONE|Error' $L/direct_f25.log | tail -1)"

# 2) csfit f25 (primary condition)
[ -f $L/csfit_f25.log ] && grep -q DONE $L/csfit_f25.log || run csfit_f25 "task4_csfit.py --frac f25"
# 3) direct+nufft f100
[ -f $L/direct_f100.log ] && grep -q DONE $L/direct_f100.log || run direct_f100 "task4_baselines.py --frac f100"
# 4) csfit f100
[ -f $L/csfit_f100.log ] && grep -q DONE $L/csfit_f100.log || run csfit_f100 "task4_csfit.py --frac f100"
echo "[chain] ALL CPU RECONS DONE"
