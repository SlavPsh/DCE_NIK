#!/bin/bash
# NIK-PK at f100 (ALL spokes, no keep-file) for the quality reference: F0,F2 on sl18,19,21.
cd /net/beegfs/users/P101440/DCE_NIK
LOGD=/net/beegfs/users/P101440/tmp
for Z in 18 19 21; do
  for F in 0 2; do
    name="pk_f${F}_sl${Z}_f100"; jf="$LOGD/job_${name}.sh"
    cat > "$jf" <<EOF
#!/bin/bash
#SBATCH --job-name=$name
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=$LOGD/${name}_%j.log
cd /net/beegfs/users/P101440/DCE_NIK
eval "\$(micromamba shell hook --shell bash)"
micromamba run -n torch29 python train_grasp_nik.py \\
  --model wire_ff_patlak --patlak-free $F --aif-file /net/beegfs/users/P101440/DCE_NIK/aif_slice${Z}.npz \\
  --slices $Z --no-compile \\
  --save-dir /net/beegfs/users/P101440/DCE_NIK/results_batch/pk_f${F}_sl${Z}_f100
echo "DONE $name"
EOF
    sbatch "$jf"
  done
done
echo "PK f100 SUBMITTED"
