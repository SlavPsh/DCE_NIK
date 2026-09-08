#!/bin/bash
# Task 2: PK/Patlak F in {0,2} on sl18,19 (sl21 exists). + SPIRiT eval: full-rank sl21 + spirit {1e-2,1e-1}.
cd /scratch/rnga/vvpshenov/DCE_NIK
KEEP=/scratch/rnga/vvpshenov/DCE_NIK/spoke_masks/keep_f25.npy
LOGD=/home/rnga/vvpshenov/my-scratch/tmp
gpujob () {  # name, extra-args, save-dir
  local name="$1" extra="$2" dir="$3"; local jf="$LOGD/job_${name}.sh"
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
micromamba run -n torch29 python train_grasp_nik.py $extra --spoke-keep-file $KEEP \\
  --save-dir /scratch/rnga/vvpshenov/DCE_NIK/results_batch/$dir
echo "DONE $name"
EOF
  sbatch "$jf"
}
# Task 2 PK (sl18,19; F0,F2)
for Z in 18 19; do
  for F in 0 2; do
    gpujob "t2_pk_f${F}_sl${Z}" "--model wire_ff_patlak --patlak-free $F --aif-file /scratch/rnga/vvpshenov/DCE_NIK/aif_slice${Z}.npz --slices $Z --no-compile" "pk_f${F}_sl${Z}"
  done
done
# SPIRiT eval (sl21 full-rank + coil-consistency prior; baseline weight 0 = full_sl21f25 exists)
for W in 0.01 0.1; do
  gpujob "spirit_w${W}_sl21" "--model wire_ff_res --slices 21 --spirit-weight $W --spirit-kernel /scratch/rnga/vvpshenov/DCE_NIK/spirit_kernel_sl21.npz" "spirit_w${W}_sl21"
done
echo "TASK2+SPIRIT SUBMITTED"
