#!/bin/bash
#SBATCH -J gv2aorta2
#SBATCH -p defq
#SBATCH -c 4
#SBATCH --mem 24G
#SBATCH -t 0:40:00
# rerun of 060 with the rewritten detector (vessel-disc matched filter + z stability + earliest ttp, tracked lumen disc):
# grasp v2 recon, then median intensity curves over that one roi for all 7 tags. verification overlay on anatomy in
# three views is part of the output. figures + csv + md in DCE_NIK/results/dce_rerun_gv2 (flow back through git).
set -uo pipefail
D=/net/beegfs/users/P101440/DCE_NIK; G=/net/beegfs/users/P101440/grasp_v2
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} PYTHONWARNINGS=ignore PYTHONDONTWRITEBYTECODE=1
export OUT=/net/beegfs/users/P101440/dce_data/orig/gv2_DCE_Rerun FIGDIR=$D/results/dce_rerun_gv2
export TAGS=nufft,lam0.02,lam0.08,lam0.25,glam0.02,glam0.08,glam0.25 REFTAG=nufft
echo "$(date '+%F %T') host $(hostname) job ${SLURM_JOB_ID:-?}"
cd $G && git log --oneline -1
for a in 1 2 3; do micromamba run -n torch29 python -u gv2_aorta.py && break; echo "attempt $a failed"; sleep 30; done
ls -la $FIGDIR/aorta*
