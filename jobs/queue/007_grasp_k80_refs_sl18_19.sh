#!/bin/bash
#SBATCH -J gk80refs
#SBATCH -p defq
#SBATCH -c 8
#SBATCH --mem 32G
#SBATCH -t 12:00:00
# grasp references at the standard k80 input for slices 18/19 (rerun of 006, which ran stale code): grasp v2 n12 k80 lam0.25, grasp pro f80match K5
# then the k80 eval + figures (005's eval stage) chained afterok on a 1g slice
set -uo pipefail
D=/net/beegfs/users/P101440/DCE_NIK
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH OMP_NUM_THREADS=8
P="micromamba run -n torch29 python -u"
echo "$(date '+%F %T') host $(hostname) job ${SLURM_JOB_ID:-?}"
cd /net/beegfs/users/P101440/grasp_pro_py; git log --oneline -1
for Z in 18 19; do echo "== grasp pro f80match slice $Z $(date '+%T')"; SLICE=$Z $P cs_nikmatch.py | grep -E 'slice|SAVED|Error'; done
cd /net/beegfs/users/P101440/grasp_v2; git log --oneline -1
echo "== grasp v2 n12 k80 slices 18,19 $(date '+%T')"; SET=sweep NLINE=12 KEEP80=1 LAM_FRAC=0.25 SLICES=18,19 $P grasp_v2_real.py | grep -E 'SAVED|cached|DONE|Error'
ls -la /net/beegfs/users/P101440/grasp_v2/results_grasp_v2/gv2_slice1[89]_n12_k80.npy /net/beegfs/users/P101440/grasp_pro_py/results_spoke_cs/cs_slice1[89]_f80match.npy && \
sbatch --dependency=afterok:$SLURM_JOB_ID --export=ALL,STAGE=eval --array=0 -J pk_k80_eval2 --gres=gpu:1g.12gb:1 -c 4 --mem 32G -t 2:00:00 \
  --output=$D/jobs/log/007_grasp_k80_refs_sl18_19_eval_%j.out --error=$D/jobs/log/007_grasp_k80_refs_sl18_19_eval_%j.out $D/jobs/queue/005_pk_arms_k80.sh && echo "k80 eval + figures chained"
echo "REFS_DONE $(date '+%F %T')"
