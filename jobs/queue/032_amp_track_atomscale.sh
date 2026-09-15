#!/bin/bash
#SBATCH -J amptrack2
#SBATCH -p gpu
#SBATCH --gres=gpu:2g.24gb:1
#SBATCH -c 8
#SBATCH --mem 32G
#SBATCH -t 4:30:00
# fourth variant of the amplitude test (031): tofts8 atoms rescaled from unit L2 norm (rms 1/sqrt(G) ~ 0.054) to unit rms, i.e. the O(1)
# amplitude the patlak class uses, so the amplitude head needs the same coefficient scale as in nik-patlak. same span, same everything else.
# afterwards the tracker reruns over all four variants (base / wd0 / lr1e-4 / atomscale).
set -uo pipefail
D=/net/beegfs/users/P101440/DCE_NIK; cd $D
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P="micromamba run -n torch29 python -u"
RES=results/tofts_vs_patlak; AT=$RES/amp_track; Z=21; SELF=$D/jobs/queue/032_amp_track_atomscale.sh
echo "$(date '+%F %T') host $(hostname) job ${SLURM_JOB_ID:-?} stage ${STAGE:-train}"; nvidia-smi -L || true

if [ "${STAGE:-train}" = track ]; then
  $P tofts_amp_track.py --slice $Z --runs base:$D/$AT/base_sl$Z,wd0:$D/$AT/wd0_sl$Z,lr1e-4:$D/$AT/lr1e-4_sl$Z,atomscale:$D/$AT/atomscale_sl$Z; echo "TRACK exit $?"; exit
fi
sbatch --dependency=afterany:$SLURM_JOB_ID --export=ALL,STAGE=track -J amptrack2_cpu -p defq --gres="" -c 4 --mem 16G -t 1:00:00 \
  --output=$D/jobs/log/032_amp_track_cpu_%j.out --error=$D/jobs/log/032_amp_track_cpu_%j.out $SELF && echo "tracker chained afterany $SLURM_JOB_ID"
BAS=$RES/basis_sl${Z}_r8_rms1.npz
[ -f $BAS ] || $P - <<PY
import numpy as np
z = np.load("$RES/basis_sl${Z}_r8.npz", allow_pickle=True); d = {k: z[k] for k in z.files}; G = d["atoms"].shape[0]
d["atoms"] = (d["atoms"] * np.sqrt(G)).astype(np.float32); d["R_patlak"] = (d["R_patlak"] * np.sqrt(G)).astype(np.float32)   # same span, unit rms atoms
np.savez("$BAS", **d); print("atoms rms", float(np.sqrt((d["atoms"] ** 2).mean(0)).mean()), "shape", d["atoms"].shape)
PY
OUT=$AT/atomscale_sl$Z; mkdir -p $OUT; echo "variant atomscale -> $OUT (basis $BAS)"
$P train_grasp_nik.py --model wire_ff_tofts --tofts-basis $BAS --slices $Z --seed 0 --ff-seed 0 --steps 40000 \
  --spoke-keep-file spoke_masks/keep_f80match.npy --spoke-heldout-file spoke_masks/val_k80_m8.npy \
  --snapshot-every 2000 --no-restore --no-compile --resume --save-dir $OUT 2>&1 | tee $OUT/train.log
echo "$(date '+%F %T') TRAIN_DONE atomscale exit ${PIPESTATUS[0]}"
