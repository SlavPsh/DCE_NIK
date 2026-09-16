#!/bin/bash
#SBATCH -J gv2fix
#SBATCH -p defq
#SBATCH -c 2
#SBATCH --mem 4G
#SBATCH -t 0:20:00
# 14 tasks of the 043 slice array hung at ~18:55 (no output since, every later task completed). resubmit the array over all
# 54 slices with the same driver (slices with valid outputs for every lambda exit in seconds), then the post stage.
set -uo pipefail
D=/net/beegfs/users/P101440/DCE_NIK
export PYTHONWARNINGS=ignore
echo "$(date '+%F %T') host $(hostname) job ${SLURM_JOB_ID:-?}"
A=$(sbatch --parsable --export=ALL,STAGE=recon,PYTHONWARNINGS=ignore --array=0-53%24 -J gv2rr2 -p defq -c 8 --mem 16G -t 4:00:00 \
    --output=$D/jobs/log/044_gv2_rerun_fix_recon_%A_%a.out --error=$D/jobs/log/044_gv2_rerun_fix_recon_%A_%a.out $D/jobs/queue/043_gv2_rerun.sh) || exit 1
B=$(sbatch --parsable --dependency=afterok:$A --export=ALL,STAGE=post,PYTHONWARNINGS=ignore -J gv2post2 -p defq -c 4 --mem 32G -t 2:00:00 \
    --output=$D/jobs/log/044_gv2_rerun_fix_post_%j.out --error=$D/jobs/log/044_gv2_rerun_fix_post_%j.out $D/jobs/queue/043_gv2_rerun.sh) || exit 1
echo "chained recon array $A -> post $B"; echo "FIX_DONE"
