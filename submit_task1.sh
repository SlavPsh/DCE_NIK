#!/bin/bash
# Task 1 ARM B: NIK-Patlak F=0, slice 21, f25, binned t at {5,15,30,60,120} spokes/frame.
# ARM A (continuous) = existing results_batch/pk_f0_sl21.
cd /scratch/rnga/vvpshenov/DCE_NIK
KEEP=/scratch/rnga/vvpshenov/DCE_NIK/spoke_masks/keep_f25.npy
AIF=/scratch/rnga/vvpshenov/DCE_NIK/aif_slice21.npz
LOGD=/home/rnga/vvpshenov/my-scratch/tmp
for N in 5 15 30 60 120; do
  name="t1_bin${N}_sl21"; jf="$LOGD/job_${name}.sh"
  cat > "$jf" <<EOF
#!/bin/bash
#SBATCH --job-name=$name
#SBATCH --partition=luna-gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=$LOGD/${name}_%j.log
cd /scratch/rnga/vvpshenov/DCE_NIK
eval "\$(micromamba shell hook --shell bash)"
micromamba run -n torch29 python train_grasp_nik.py \\
  --model wire_ff_patlak --patlak-free 0 --aif-file $AIF --slices 21 \\
  --spoke-keep-file $KEEP --no-compile --bin-spokes $N \\
  --save-dir /scratch/rnga/vvpshenov/DCE_NIK/results_batch/pk_f0_bin${N}_sl21
echo "DONE $name"
EOF
  sbatch "$jf"
done
echo "TASK1 SUBMITTED"
