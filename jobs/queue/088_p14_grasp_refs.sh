#!/bin/bash
#SBATCH -J p14grasp
#SBATCH -p defq
#SBATCH -c 16
#SBATCH --mem 96G
#SBATCH -t 4:00:00
#SBATCH --array=0-3
# p14 grasp references on the k80 views, parallel: tasks 0-2 grasp-pro f80match for slices 21 / 24 / 27 (cs_nikmatch, output dir now created), task 3 grasp v2 n12 k80 for all three
set -uo pipefail
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH OMP_NUM_THREADS=16 DCE_DS=p14
P="micromamba run -n torch29 python -u"; SL=(21 24 27)
if [ "$SLURM_ARRAY_TASK_ID" -lt 3 ]; then Z=${SL[$SLURM_ARRAY_TASK_ID]}; cd /net/beegfs/users/P101440/grasp_pro_py; echo "grasp pro f80match slice $Z $(date '+%T')"; SLICE=$Z $P cs_nikmatch.py | grep -E 'slice|SAVED|Error|Traceback'; echo "GP $Z exit $?"; ls -la results_spoke_cs_p14/ | awk '{print $5, $9}'
else cd /net/beegfs/users/P101440/grasp_v2; echo "grasp v2 n12 k80 slices 21,24,27 $(date '+%T')"; SET=sweep NLINE=12 KEEP80=1 LAM_FRAC=0.25 SLICES=21,24,27 $P grasp_v2_real.py | grep -E 'SAVED|cached|DONE|Error|Traceback'; echo "GV exit $?"; ls -la results_grasp_v2_p14/ | awk '{print $5, $9}'; fi
