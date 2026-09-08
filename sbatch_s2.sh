#!/bin/bash
#SBATCH --job-name=swcmp
#SBATCH --partition=luna-cpu-short
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/rnga/vvpshenov/DCE_NIK/results/realdata_nik_vs_cs_figures/swcmp_%j.log
cd /scratch/rnga/vvpshenov/DCE_NIK
${PY:-/scratch/rnga/vvpshenov/micromamba/envs/torch29/bin/python} -u step2_fig.py
