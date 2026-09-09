# reproduction (all from /net/beegfs/users/P101440/DCE_NIK, env torch29, luna sbatch)
PY=/net/beegfs/users/P101440/micromamba/envs/torch29/bin/python

# 1 basis + rank study (cpu, 2 min)            -> results/tofts_vs_patlak/basis_{xph,xph_spgr,sl18,sl19,sl21}.npz, basis_*.log
sbatch sbatch_tofts_basis.sh
# 2 verification 3a-3e (1g.10gb, 3 min)        -> results/tofts_vs_patlak/verify.json, verify_*.log
sbatch sbatch_tofts_verify.sh
# 3 phantom runs (array 0-8, 1g.10gb, ~3.5 h each)  0-2 nomotion tofts s0-2 | 3-5 motion patlak | 6-8 motion tofts
#   nomotion patlak controls = existing results/xcat_physical_nomotion_nik_vs_grasp/checkpoints/w768_ks2.5_s{0,1,2} (xph_train.py, unchanged protocol)
sbatch sbatch_tofts_phantom.sh
#   single run by hand: XPH_SIM=nomotion $PY xph_train.py --model wire_ff_tofts --hidden-width 768 --k-sigma 2.5 --seed 0 --steps 40000 --ckpt-every 2000
#                       XPH_SIM=nomotion $PY xph_eval.py --tag w768_ks2.5_s0_tofts16
# 4 in-vivo runs (array 0-17, 2g.20gb, luna-gpu-long)  i = slice_idx*6 + model_idx*3 + seed ; slices 18,19,21 ; patlak,tofts ; seeds 0,1,2
sbatch sbatch_tofts_invivo_4g.sh   # (2g version sbatch_tofts_invivo.sh was unschedulable; same commands)
#   single run by hand (tofts, sl21, s0):
#   $PY train_grasp_nik.py --model wire_ff_tofts --tofts-basis results/tofts_vs_patlak/basis_sl21.npz --slices 21 --seed 0 --ff-seed 0 \
#      --spoke-keep-file spoke_masks/keep_f25.npy --spoke-heldout-file spoke_masks/val_f25c_m8.npy --no-compile --resume --save-dir results/tofts_vs_patlak/invivo/tofts_sl21_s0
#   patlak control: replace the model args by  --model wire_ff_patlak --aif-file aif_slice21.npz --patlak-free 0
# 5 evaluation (1g.10gb; auto-chained afterany on 3+4) -> phantom_{nomotion,motion}.{json,md}, invivo.{json,md}
sbatch sbatch_tofts_eval.sh
