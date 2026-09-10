#!/bin/bash
#SBATCH -J gk80refs
#SBATCH -p defq
#SBATCH -c 8
#SBATCH --mem 32G
#SBATCH -t 12:00:00
# grasp references at the standard k80 input for slices 18/19 (slice 21 exists): grasp v2 n12 k80 lam0.25, grasp pro f80match K5
set -uo pipefail
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH OMP_NUM_THREADS=8
P="micromamba run -n torch29 python -u"
echo "$(date '+%F %T') host $(hostname) job ${SLURM_JOB_ID:-?}"
cd /net/beegfs/users/P101440/grasp_pro_py
for Z in 18 19; do echo "== grasp pro f80match slice $Z $(date '+%T')"; SLICE=$Z $P cs_nikmatch.py; echo "exit $?"; done
cd /net/beegfs/users/P101440/grasp_v2
echo "== grasp v2 n12 k80 slices 18,19 $(date '+%T')"; SET=sweep NLINE=12 KEEP80=1 LAM_FRAC=0.25 SLICES=18,19 $P grasp_v2_real.py; echo "exit $?"
ls -la /net/beegfs/users/P101440/grasp_v2/results_grasp_v2/gv2_slice1[89]_n12_k80.npy /net/beegfs/users/P101440/grasp_pro_py/results_spoke_cs/cs_slice1[89]_f80match.npy
echo "REFS_DONE"
