#!/bin/bash
#SBATCH -J neutral
#SBATCH -p luna-cpu-short
#SBATCH -c 8
#SBATCH --mem 32G
#SBATCH -t 2:00:00
#SBATCH -o /scratch/rnga/vvpshenov/DCE_NIK/neutral_%j.log
cd /scratch/rnga/vvpshenov/DCE_NIK
${PY:-/scratch/rnga/vvpshenov/micromamba/envs/torch29/bin/python} -u invivo_neutral_ruler.py
