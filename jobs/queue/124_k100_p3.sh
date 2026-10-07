#!/bin/bash
#SBATCH -J k100p3
#SBATCH -p gpu
#SBATCH --gres=gpu:2g.24gb:1
#SBATCH -c 8
#SBATCH --mem 40G
#SBATCH -t 3:00:00
#SBATCH --array=0-15
# k100 standard (user decision 2026-10-07, FINDINGS 10): every nik arm on ALL spokes, p3 (sigma 2.5), each arm under its own protocol; slice 21 seed 0 runs
# exist from queue 119. tasks 0-7 tofts8 in-coil + prior (18 s0-2, 19 s0-2, 21 s1-2); 8-9 tofts8 out-coil + prior 18 / 19; 10-11 patlak + prior; 12-13 sub16 wd 3e-3 + prior;
# 14-15 nik-free. -> results/tofts_vs_patlak/invivo_k100{,_oc}/<arm>_sl<Z>_s<S>. the evaluation is queue 126 (waits for every output).
set -uo pipefail
D=/net/beegfs/users/P101440/DCE_NIK; cd $D
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P="micromamba run -n torch29 python -u"; RES=results/tofts_vs_patlak
echo "$(date '+%F %T') host $(hostname) job ${SLURM_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID:-none}"; nvidia-smi -L || true
i=$SLURM_ARRAY_TASK_ID
ZS=(18 18 18 19 19 19 21 21 18 19 18 19 18 19 18 19); SS=(0 1 2 0 1 2 1 2 0 0 0 0 0 0 0 0); Z=${ZS[$i]}; S=${SS[$i]}
PRIOR="--support-weight 1 --support-mask spoke_masks/support_sl${Z}_d6.npy --support-orient rot180 --coef-tv-every 32"
if   [ $i -lt 8 ];  then NM=tofts8;   OUT=$RES/invivo_k100/tofts8_sl${Z}_s$S;    MA="--model wire_ff_tofts --tofts-basis $RES/basis_sl${Z}_r8_rms1.npz $PRIOR --weight-decay 0.01"
elif [ $i -lt 10 ]; then NM=tofts8oc; OUT=$RES/invivo_k100_oc/tofts8_sl${Z}_s0; MA="--model wire_ff_tofts --tofts-basis $RES/basis_sl${Z}_r8_rms1.npz --coil-mode output $PRIOR --weight-decay 0.01"
elif [ $i -lt 12 ]; then NM=patlak;   OUT=$RES/invivo_k100/patlak_sl${Z}_s0;    MA="--model wire_ff_patlak --aif-file $D/aif_slice$Z.npz --patlak-free 0 $PRIOR --weight-decay 0.01"
elif [ $i -lt 14 ]; then NM=sub16;    OUT=$RES/invivo_k100/sub16_sl${Z}_s0;     MA="--model wire_ff_subspace --rank 16 $PRIOR --weight-decay 0.003"
else                     NM=free;     OUT=$RES/invivo_k100/free_sl${Z}_s0;      MA="--model wire_ff_res --weight-decay 0.01"
fi
[ -f $OUT/nik_slice_${Z}_cplx.npy ] && { echo "exists $OUT"; exit 0; }
mkdir -p $OUT; echo "task $i: $NM k100 slice $Z seed $S -> $OUT"
$P train_grasp_nik.py $MA --slices $Z --seed $S --ff-seed $S --steps 10000 --no-restore \
  --spoke-keep-file spoke_masks/keep_f100.npy --no-compile --resume --save-dir $OUT 2>&1 | tee $OUT/train.log
echo "$(date '+%F %T') TRAIN_DONE $NM k100 sl$Z s$S exit ${PIPESTATUS[0]}"
