#!/bin/bash
#SBATCH --job-name=step2-kid
#SBATCH --partition=defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/net/beegfs/users/P101440/tmp/step2_kidney_%j.log
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd /net/beegfs/users/P101440/DCE_NIK
eval "$(micromamba shell hook --shell bash)"
for Z in 18 19 20 21; do
  echo "===== STEP 2 slice $Z ====="
  micromamba run -n torch29 python step2_kidney.py $Z
done
echo "ALL STEP2 DONE"
