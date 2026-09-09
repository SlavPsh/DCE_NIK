#!/bin/bash
#SBATCH --job-name=xph-report
#SBATCH --partition=defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0:50:00
#SBATCH --output=/net/beegfs/users/P101440/DCE_NIK/results/xcat_physical_nomotion_nik_vs_grasp/slurm/report_%j.log
set -e; cd /net/beegfs/users/P101440/DCE_NIK
MM=/net/beegfs/users/P101440/micromamba/bin/micromamba
echo "[report $SLURM_JOB_ID] regen phantom figures $(date)"
$MM run -n torch29 python -u xph_aggregate.py | tail -3
echo "[report $SLURM_JOB_ID] regen real-data figures"
$MM run -n torch29 python -u task_realdata_figures.py || echo "WARN real-fig regen failed"
echo "[report $SLURM_JOB_ID] build notebook"
$MM run -n torch29 python -u build_report_nb.py
echo "[report $SLURM_JOB_ID] DONE $(date)"
