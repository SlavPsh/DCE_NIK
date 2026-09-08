#!/bin/bash
#SBATCH --job-name=step2-kid
#SBATCH --partition=luna-cpu-short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/rnga/vvpshenov/my-scratch/tmp/step2_kidney_%j.log
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd /scratch/rnga/vvpshenov/DCE_NIK
eval "$(micromamba shell hook --shell bash)"
for Z in 18 19 20 21; do
  echo "===== STEP 2 slice $Z ====="
  micromamba run -n torch29 python step2_kidney.py $Z
done
echo "ALL STEP2 DONE"
