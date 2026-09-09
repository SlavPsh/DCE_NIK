#!/bin/bash
# z15 capstone fleet: GRASP + NIK-F0(3 seeds) + NIK-sub5(3) + NIK-sub16(2) + NIK-free(1)
# reuse settled HPs: width 768, k_sigma 2.5. spread GRES across a100 + 4g.40gb to run concurrently.
cd /net/beegfs/users/P101440/DCE_NIK
S=sbatch
# --- GRASP-Pro (K=5) ---
$S --gres=gpu:h100:1 sbatch_xph_grasp.sh
# --- NIK-F0 rank-3 Patlak (PK lane), 3 seeds ---
for sd in 0 1 2; do
  G=$([ $sd -eq 0 ] && echo a100:1 || echo 4g.40gb:1)
  $S --gres=gpu:$G --export=ALL,W=768,KS=2.5,SEED=$sd --job-name=xph-f0s$sd sbatch_xph_train.sh
done
# --- NIK-sub5 rank-5 warmstart (image lane), 3 seeds ---
for sd in 0 1 2; do
  G=$([ $sd -eq 0 ] && echo a100:1 || echo 4g.40gb:1)
  $S --gres=gpu:$G --export=ALL,MODEL=wire_ff_subspace,RANK=5,WS=1,W=768,SEED=$sd --job-name=xph-sub5s$sd sbatch_xph_img.sh
done
# --- NIK-sub16 rank-16 warmstart, 2 seeds ---
for sd in 0 1; do
  $S --gres=gpu:4g.47gb:1 --export=ALL,MODEL=wire_ff_subspace,RANK=16,WS=1,W=768,SEED=$sd --job-name=xph-sub16s$sd sbatch_xph_img.sh
done
# --- NIK-free fully-continuous upper bound, 1 seed ---
$S --gres=gpu:h100:1 --export=ALL,MODEL=wire_ff,RANK=0,WS=0,W=768,SEED=0 --job-name=xph-free sbatch_xph_img.sh
echo "fleet submitted"
