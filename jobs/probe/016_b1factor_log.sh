#!/bin/bash
# retrieval only: the check_b1_factor.py log(s) (sbatch_b1factor.sh) from the DCE_NIK root and results slurm dirs
cd /net/beegfs/users/P101440/DCE_NIK; grep -n "output\|-o " sbatch_b1factor.sh
for f in $(ls -t b1factor_*.log b1f*.log results/xcat_physical_nomotion_nik_vs_grasp/slurm/*b1*.log 2>/dev/null | head -4); do echo "===== $f"; cat "$f"; done
grep -l "factorization residual" *.log results/*/slurm/*.log 2>/dev/null | head -3
