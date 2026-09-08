#!/bin/bash
#SBATCH --job-name=acttst
#SBATCH -p luna-cpu-tiny
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -t 10
#SBATCH --output=/scratch/rnga/vvpshenov/DCE_NIK/acttest.log
cd /scratch/rnga/vvpshenov/DCE_NIK
${PY:-/scratch/rnga/vvpshenov/micromamba/envs/torch29/bin/python} -u _acttest.py
