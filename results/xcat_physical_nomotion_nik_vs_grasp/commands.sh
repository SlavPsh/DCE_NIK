#!/usr/bin/env bash
cd /scratch/rnga/vvpshenov/DCE_NIK
MM=/home/rnga/vvpshenov/my-scratch/micromamba/bin/micromamba
# (data adapter xph_common; AIF aif_xph.npz built data-driven from coarse-recon arterial pixels)
# GRASP-Pro
sbatch sbatch_xph_grasp.sh
# NIK F0 search: Stage A width, Stage B k_sigma, final seeds (val-kNMSE selection)
for W in 256 512 768; do sbatch --export=ALL,W=$W,KS=2.5,SEED=0 sbatch_xph_train.sh; done
for KS in 1.75 3.5; do sbatch --export=ALL,W=768,KS=$KS,SEED=0 sbatch_xph_train.sh; done
for S in 1 2; do sbatch --export=ALL,W=768,KS=2.5,SEED=$S sbatch_xph_train.sh; done
# NIK image lane (subspace R5 warmstart seeds 0/1/2 + free) - MIG slices
for S in 0 1 2; do sbatch --gres=gpu:4g.40gb:1 --export=ALL,MODEL=wire_ff_subspace,RANK=5,WS=1,W=768,SEED=$S sbatch_xph_img.sh; done
sbatch --gres=gpu:2g.20gb:1 --export=ALL,MODEL=wire_ff,RANK=0,WS=0,W=768,SEED=0 sbatch_xph_img.sh
# aggregate 4-method comparison + figures + animation
$MM run -n torch29 python -u xph_aggregate.py
