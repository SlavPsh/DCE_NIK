#!/bin/bash
# NIK-PK at f100 (ALL spokes, no keep-file) for the quality reference: F0,F2 on sl18,19,21.
cd /scratch/rnga/vvpshenov/DCE_NIK
LOGD=/home/rnga/vvpshenov/my-scratch/tmp
for Z in 18 19 21; do
  for F in 0 2; do
    name="pk_f${F}_sl${Z}_f100"; jf="$LOGD/job_${name}.sh"
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
  --model wire_ff_patlak --patlak-free $F --aif-file /scratch/rnga/vvpshenov/DCE_NIK/aif_slice${Z}.npz \\
  --slices $Z --no-compile \\
  --save-dir /scratch/rnga/vvpshenov/DCE_NIK/results_batch/pk_f${F}_sl${Z}_f100
echo "DONE $name"
EOF
    sbatch "$jf"
  done
done
echo "PK f100 SUBMITTED"
