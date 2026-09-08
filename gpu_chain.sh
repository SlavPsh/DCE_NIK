#!/usr/bin/env bash
# GPU orchestrator: use the single MIG slice without memory conflict.
# [running] NIK f25 -> (gap) GPU non-neural recons all fracs -> NIK f100.
cd /scratch/rnga/vvpshenov/DCE_NIK
L=results/task4_xcat_nomotion_pilot/logs; A=results/task4_xcat_nomotion_pilot/arrays
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4

# 1) wait for NIK f25 full run to finish
until grep -qE "^DONE|Error|Traceback|OutOfMemory|Killed" $L/nik_f25_seed0.log 2>/dev/null; do sleep 20; done
echo "[gpu_chain] $(date +%H:%M) NIK f25 finished: $(grep -iE '^DONE|Error' $L/nik_f25_seed0.log | tail -1)"

# 2) GPU non-neural recons (direct-discrete + CS-fit + NUFFT-sanity), both fracs (GPU free now)
echo "[gpu_chain] $(date +%H:%M) START gpu_recon"
micromamba run -n torch29 python -u task4_gpu_recon.py --frac both > $L/gpu_recon.log 2>&1
echo "[gpu_chain] $(date +%H:%M) END gpu_recon: $(grep -c GPU_RECON_DONE $L/gpu_recon.log) done-marker"

# 3) NIK f100 (GPU free again)
if [ ! -f $A/nik_F0_f100_seed0.npz ]; then
  echo "[gpu_chain] $(date +%H:%M) START NIK f100"
  micromamba run -n torch29 python -u task4_nik.py --frac f100 --seed 0 --steps 40000 > $L/nik_f100_seed0.log 2>&1
  echo "[gpu_chain] $(date +%H:%M) END NIK f100"
fi
echo "[gpu_chain] ALL GPU WORK DONE"
