# XCAT physical no-motion NIK-vs-GRASP-Pro — RESUME / progress ledger

## Dataset (FROZEN)
- NO-MOTION: /scratch/rnga/vvpshenov/XCAT-ERIC/results/simulation_results_20260816T210718.mat
  (respPeriod=[78 47 65]="N/A"). Breathing file (223443, respPeriod=5) NOT used here.
- Geometry: DCE (344 frames, 8 coils, 112=7ang x 16part, 220 RO). Aligned stack-of-stars.
  trajDCE[:,2]=partition idx (1..16, centric). Recon frame times 0.26..179.3s (0.52s).
  GroundTruth.img (901, 16, 220, 152) @0.2s = physical SPGR truth. labelGT (16,152,220), aorta=36.
  sim: TR4.66 TE1.7 FA18 relax3.5 inj12 scan180 res 1.7x1.7x6mm matrix220 16slices.
- Recon slice chosen: zi=TBD (all have aorta; kidney label13 peaks z9-13).

## Phases
[x] P1 DONE: geometry validated (static corr 0.953, sign+1), zi=10, aorta36/cortex13/medulla37, data-driven AIF peak 27.9s (aif_xph.npz)
[x] P2 DONE: train ang{0-4}(5/frame=Proj5), val{5}, test{6}; whole spokes
[~] P3: width=768 selected (val kNMSE, near-tie). Stage B ks{1.75,3.5} running (3027971/72). ks2.5 done.
[ ] P4 NIK checkpoint selection (>=20 ckpts, argmin val kNMSE)
[ ] P5 GRASP-Pro: PCA K=5 subspace + SP-TV0.0005 + T-TV0.001 via cufinufft (faithful algo; GROG->NUFFT documented)
[ ] P6 reconstruct both at same input spokes; eval at GRASP frame centres
[ ] P7 image eval (masked NRMSE + HaarPSI/SSIM), P8 curves, P9 kspace, P10 diagnostics, P11 table
[ ] figures/animations/report/sanity

## Key decisions
- eval domain: MAGNITUDE coil-combined, ONE xcat-truth-derived scale, xcat anatomical mask.
- eval times = GRASP-Pro frame centres; NIK queried there; truth sampled there.
- spatial-freq param = k_sigma (WIRE FF spatial bandwidth), current 2.5.
- GPU work = SLURM jobs (feedback_gpu_slurm_job).

## Live job flow (2-GPU account cap)
- Stage A: 3027948 w256, 3027949 w512, 3027950 w768 (ks2.5 s0) -> nik_eval_wW_ks2.5_s0.npz
- GRASP: 3027951 -> grasp_recon.npz
- NEXT after Stage A: select width=argmin(val_nmse) from the 3 nik_eval npz; submit Stage B
  (k_sigma {1.75, 3.5} at selected width, s0; 2.5 already done). Then select ks; submit final seeds 1,2.
- modules: xph_common (data/truth/roi), xph_pipeline (NIK), xph_grasp (GRASP), xph_train, xph_eval, xph_grasp_run.
- eval convention: recon rot180 -> truth; ONE global truth-derived scale; masked NRMSE + curves.

## ORIENTATION (critical, verified)
- NIK recon -> rot180 to truth (ifft2c_mri convention). Applied in xph_eval. corr 0.934.
- GRASP recon -> IDENTITY to truth (cufinufft + negated trajs). corr 0.976. xph_grasp_run FIXED (was wrongly rot180).
- each aligned by its operator's fixed convention (verified by static-image corr), applied identically to all frames.

## Stage A (val-kNMSE selection) + GRASP preliminary
- w256: val 8.342e-3, img NRMSE 0.0865, aortaCurve 0.201
- w512: val 8.367e-3, img NRMSE 0.0896, aortaCurve 0.178
- w768: PENDING
- GRASP (corrected identity): img NRMSE 0.071, aortaCurve 0.202, cortex 0.061, medulla 0.063

## TWO-LANE DESIGN (user directive: investigate image-recon NIK, not just F0 PK-builder)
- PK/coefficient lane (F0, rank-3 fixed Patlak): Stage A done (w768), Stage B running (ks1.75/3.5), then final 3 seeds.
- IMAGE lane (the fair image-quality comparators vs GRASP-Pro K=5 free PCA):
  * sub5 = wire_ff_subspace rank-5, PCA-warmstart (matches GRASP K=5; only spatial rep differs). seeds 0,1,2. jobs 3027980-82.
  * free = wire_ff fully-free continuous (upper bound). seed 0. job 3027983.
  * eval npz: img_eval_{sub5,free}_w768_s{seed}.npz  (rec_best for HaarPSI/SSIM/PSNR in aggregation)
- PRELIMINARY (150 steps, w256): sub5 imgNRMSE 0.047 << F0 0.086-0.094, and < GRASP 0.071. Confirms F0 was the wrong NIK for image quality.
- WHY F0 not F2: task said F0-only; F2 coeff maps gauge-ambiguous (Task 3). For IMAGES, free subspace is correct -> this lane.

## FOLLOW-ONS (user: "do both")
- R=16 subspace (rank axis completion): jobs 3028051/52/53 (sub16 w768 s0/1/2, 4g.40gb, 4h). aggregation auto-includes NIK-sub16.
- Real-data figures: regenerate denoising-spectra / spoke-frontier / PK-CoV / per-frame-HaarPSI as PNGs from existing real recons (results_spoke_cs, results_ref_3s). Agent inventorying recons+analysis.
- After R=16 done -> re-run xph_aggregate.py (now 5-6 methods). 
