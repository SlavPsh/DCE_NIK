#!/bin/bash
#SBATCH -J k100p14
#SBATCH -p gpu
#SBATCH --gres=gpu:2g.24gb:1
#SBATCH -c 8
#SBATCH --mem 64G
#SBATCH -t 4:00:00
#SBATCH --array=0-20
# k100 standard on p14 (sigma 5, 512 grid), every nik arm under its own protocol: 0-8 tofts8 in-coil + prior (3 slices x 3 seeds); 9-11 tofts8 out-coil + prior;
# 12-14 patlak + prior; 15-17 sub16 wd 3e-3 + prior; 18-20 nik-free. -> results/tofts_vs_patlak/p14/invivo_k100{,_oc}/<arm>_sl<Z>_s<S>. keep mask = every view.
set -uo pipefail
D=/net/beegfs/users/P101440/DCE_NIK; cd $D
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH DCE_DS=p14
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P="micromamba run -n torch29 python -u"; RES=results/tofts_vs_patlak; OUTR=$RES/p14; REFD=/net/beegfs/users/P101440/grasp_pro_py/results_ref_p14
echo "$(date '+%F %T') host $(hostname) job ${SLURM_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID:-none}"; nvidia-smi -L || true
[ -f spoke_masks/keep_f100_p14.npy ] || $P -c "import numpy as np, dsp; np.save('spoke_masks/keep_f100_p14.npy', np.arange(dsp.NTV, dtype=np.int64)); print('keep_f100_p14', dsp.NTV)"
i=$SLURM_ARRAY_TASK_ID; SL=(21 24 27)
if   [ $i -lt 9 ];  then j=$i;        Z=${SL[$((j/3))]}; S=$((j%3)); NM=tofts8;   OUT=$OUTR/invivo_k100/tofts8_sl${Z}_s$S;    MA="--model wire_ff_tofts --tofts-basis $RES/basis_p14_sl${Z}_r8_rms1.npz --weight-decay 0.01"
elif [ $i -lt 12 ]; then j=$((i-9));  Z=${SL[$j]}; S=0; NM=tofts8oc; OUT=$OUTR/invivo_k100_oc/tofts8_sl${Z}_s0; MA="--model wire_ff_tofts --tofts-basis $RES/basis_p14_sl${Z}_r8_rms1.npz --coil-mode output --weight-decay 0.01"
elif [ $i -lt 15 ]; then j=$((i-12)); Z=${SL[$j]}; S=0; NM=patlak;   OUT=$OUTR/invivo_k100/patlak_sl${Z}_s0;    MA="--model wire_ff_patlak --aif-file $D/aif_p14_slice$Z.npz --patlak-free 0 --weight-decay 0.01"
elif [ $i -lt 18 ]; then j=$((i-15)); Z=${SL[$j]}; S=0; NM=sub16;    OUT=$OUTR/invivo_k100/sub16_sl${Z}_s0;     MA="--model wire_ff_subspace --rank 16 --weight-decay 0.003"
else                     j=$((i-18)); Z=${SL[$j]}; S=0; NM=free;     OUT=$OUTR/invivo_k100/free_sl${Z}_s0;      MA="--model wire_ff_res --weight-decay 0.01"
fi
[ $NM != free ] && MA="$MA --support-weight 1 --support-mask spoke_masks/support_p14_sl${Z}_d6.npy --support-orient rot180 --coef-tv-every 32"
[ -f $OUT/nik_slice_${Z}_cplx.npy ] && { echo "exists $OUT"; exit 0; }
mkdir -p $OUT; echo "task $i: $NM k100 p14 slice $Z seed $S -> $OUT"
$P train_grasp_nik.py $MA --k-sigma 5 --slices $Z --seed $S --ff-seed $S --steps 10000 --no-restore --out-dir $REFD \
  --spoke-keep-file spoke_masks/keep_f100_p14.npy --no-compile --resume --save-dir $OUT 2>&1 | tee $OUT/train.log
echo "$(date '+%F %T') TRAIN_DONE $NM k100 p14 sl$Z s$S exit ${PIPESTATUS[0]}"
