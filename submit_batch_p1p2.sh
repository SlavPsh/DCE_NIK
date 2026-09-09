#!/bin/bash
# submits P1 (R16+full on sl18,19,20) and P2 (R8,16,32,64,full on sl21), all at f25.
# one sbatch per run so a single failure never aborts the rest. weights + complex recon saved.
cd /net/beegfs/users/P101440/DCE_NIK
KEEP=/net/beegfs/users/P101440/DCE_NIK/spoke_masks/keep_f25.npy
LOGD=/net/beegfs/users/P101440/tmp

submit () {
  local name="$1" model="$2" rank="$3" slice="$4" dir="$5"
  local extra=""
  [ "$model" = "wire_ff_subspace" ] && extra="--rank $rank"
  local jf="$LOGD/job_${name}.sh"
  cat > "$jf" <<EOF
#!/bin/bash
#SBATCH --job-name=$name
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=$LOGD/batch_${name}_%j.log
cd /net/beegfs/users/P101440/DCE_NIK
eval "\$(micromamba shell hook --shell bash)"
micromamba run -n torch29 python train_grasp_nik.py \\
  --model $model $extra --slices $slice \\
  --spoke-keep-file $KEEP \\
  --save-dir /net/beegfs/users/P101440/DCE_NIK/results_batch/$dir
echo "DONE $name"
EOF
  sbatch "$jf"
}

# P1: R16 + full-rank on slices 18,19,20
for SL in 18 19 20; do
  submit "b_r16_sl${SL}"  wire_ff_subspace 16 $SL "nik_r16_sl${SL}"
  submit "b_full_sl${SL}" wire_ff_res      0  $SL "full_sl${SL}"
done
# P2: rank pareto on slice 21 (R8,16,32,64, full)
for R in 8 16 32 64; do
  submit "b_r${R}_sl21" wire_ff_subspace $R 21 "nik_r${R}_sl21"
done
submit "b_full_sl21f25" wire_ff_res 0 21 "full_sl21f25"
echo "ALL P1+P2 SUBMITTED"
