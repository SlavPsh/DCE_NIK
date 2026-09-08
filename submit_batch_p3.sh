#!/bin/bash
# P3: Patlak basis on slice 21, f25. F in {0,2,4} (F=0 pure Patlak). AIF passed the gate.
cd /scratch/rnga/vvpshenov/DCE_NIK
KEEP=/scratch/rnga/vvpshenov/DCE_NIK/spoke_masks/keep_f25.npy
AIF=/scratch/rnga/vvpshenov/DCE_NIK/aif_slice21.npz
LOGD=/home/rnga/vvpshenov/my-scratch/tmp
for F in 0 2 4; do
  name="b_pk_f${F}_sl21"; jf="$LOGD/job_${name}.sh"
  cat > "$jf" <<EOF
#!/bin/bash
#SBATCH --job-name=$name
#SBATCH --partition=luna-gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=$LOGD/batch_${name}_%j.log
cd /scratch/rnga/vvpshenov/DCE_NIK
eval "\$(micromamba shell hook --shell bash)"
micromamba run -n torch29 python train_grasp_nik.py \\
  --model wire_ff_patlak --patlak-free $F --aif-file $AIF --slices 21 \\
  --spoke-keep-file $KEEP --no-compile \\
  --save-dir /scratch/rnga/vvpshenov/DCE_NIK/results_batch/pk_f${F}_sl21
echo "DONE $name"
EOF
  sbatch "$jf"
done
echo "ALL P3 SUBMITTED"
