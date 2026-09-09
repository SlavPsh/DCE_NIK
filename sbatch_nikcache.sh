#!/bin/bash
#SBATCH -J nikcache
#SBATCH -p gpu
#SBATCH --gres gpu:1g.12gb:1
#SBATCH -c 4
#SBATCH --mem 24G
#SBATCH -t 3:00:00
#SBATCH -o /net/beegfs/users/P101440/DCE_NIK/nikcache_%j.log
cd /net/beegfs/users/P101440/DCE_NIK
${PY:-/net/beegfs/users/P101440/micromamba/envs/torch29/bin/python} -u xph_nik_cache.py
echo NIKCACHE_DONE
