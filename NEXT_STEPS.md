# DCE_NIK — next steps (rewritten 2026-07-18 after the render-bug finding)

## HEADLINE: the "NIK is blurry" story was a RENDER BUG, now fixed
`reconstruct_cartesian` masks |k| > support_radius at render time. It must equal the
sampled-disk radius ~1.0 (model coords = 2*traj_norm, data reaches |x|=0.997). It was set
to **0.5** (the traj_norm max -- a 2x units slip), which zeroed the OUTER HALF of sampled
k-space = the high-freq half -> ~2x blur.
- Proof (render_support_check.py, SAME model both ways): HF-ratio 7.5e-7 -> 8.5e-6,
  i.e. **8% -> 93% of CS-100** high-freq. Visually: blurry -> sharp (support_check.png).
- FIXED: train_grasp_nik --support-radius default 0.5 -> 1.0 (committed).
- This INVALIDATES a chain of earlier conclusions (all render artifacts): "NIK loses to CS
  spatially", the "held-out paradox", the 30x power-spectrum deficit, and the urgency of
  hash-grid / FF-bandwidth / SENSE as BLUR fixes. NIK was fitting high-|k| all along.

## Corrected picture (post-fix, freq/depth sweep, rank-16, slice 13)
- Perceptual vs CS-100: DISTS 0.09 -> **0.024**, HaarPSI 0.92 -> **0.96** (CS-70 = 0.018 / 0.976).
  NIK is now a near-spatial-PEER to CS; residual gap = minor streaking/grain, NOT blur.
- Temporal win STANDS (R-sweep, unaffected by render): R=5 swing 39% (=CS rank-5 cap),
  R>=6 swing 43-45% (dynamics CS's rank-5 can't represent). Sweet spot R~10-20.
- freq/depth barely move quality -> the render fix was the lever, recipe is settled:
  wire_ff_subspace rank ~16, k_sigma 1.5-2.5, w0 62-80, depth 12, dcf0, env0.75.

## In flight
- **Re-baseline at support=1.0** (results_rebase_*, waiter): full-rank + R=5/10/16/20 vs
  CS-70/CS-100, unified table (held-out/swing/nav/HF/DISTS/HaarPSI). Confirms NIK~=CS + temporal win.

## OPEN next steps (ranked, corrected)
1. **AORTA BOLUS TEST -- THE TOP ITEM.** Breaks the nav-corr circularity (NIK trains on the
   k=0 navigator, so nav-corr just grades NIK on its own training signal; the navigator is
   also too smooth to test fast dynamics). The aortic first-pass bolus (~5-10s, far faster
   than the 31s/frame binning) is a real sharp temporal feature to resolve.
   DESIGN (fairness): define the aorta ROI ONCE on a method-neutral reference (CS-100), erode
   it (interior voxels only, avoid partial-volume), apply the SAME mask to NIK and CS. Do NOT
   let NIK define its own ROI. Region across frames, not one voxel.
   SEGMENTATION (ranked): (1) early-enhancement map [(arterial frame - baseline)] + connected-
   component + circularity + eroded -- recommended; (2) temporal-signature clustering of voxel
   time-courses; (3) Hough circle on the peak-arterial frame; (4) seeded region-grow (fallback).
   METRIC on the ROI-mean curve: upslope (max dS/dt), time-to-peak, first-pass FWHM. NIK (R>5,
   continuous-t) should give steeper upslope / narrower peak than CS (rank-5 + binning).
   FIRST STEP: visualize the early-enhancement map + candidate aorta ROI on CS-100 slice 13 to
   confirm the aorta is cleanly segmentable and shows a distinct sharp rise (feasibility).
   SECOND ORGAN (later): liver parenchyma (slow portal-phase) as the counterpoint -- NIK should
   beat CS on the SHARP aorta while both match on the SLOW liver -> isolates temporal-resolution
   as the win. (Kidney cortex = optional second fast target.)
2. **Spoke-reduction frontier** -- the ACTUAL GOAL ("fewer spokes"). Everything is at 70%.
   Now reachable since spatial quality is fixed. Push below 70%, find where NIK holds and CS breaks.
3. **Aorta ROI** for dynamics -- cheap eval upgrade (sharp arterial bolus = best temporal test).
4. **Multi-slice generalization** -- everything is slice 13.
5. Residual grain (minor, DISTS 0.024 vs CS 0.018) IF worth chasing: coils-as-output / SENSE
   forward (over-determination via fixed coil maps), or the temporal low-rank we already have.
6. Hardening: end-frame DC instability (k=0 dip at t=1).

## MAJOR THREAD: pharmacokinetic (bolus-shape) temporal prior -- downstream of the aorta test
Idea: encode the KNOWN physiology (contrast curves = "sharp rise, slow fall") instead of a
generic low-rank/free temporal basis. Reuses the factorized model k=sum_r A_r(x,y)*Phi_r(t):
replace the free temporal net with PHYSICALLY-SHAPED atoms.
- FORMS: (a) Phi_r(t) = gamma-variate A*(t-t0)^alpha*exp(-(t-t0)/beta) or difference-of-sigmoids,
  learnable arrival t0 / rise alpha / fall beta -> R bolus templates at different times, mixed
  spatially by A_r; OR (b) fully parametric: predict per-voxel PK params (baseline, amplitude,
  arrival, rise, fall) and curve = baseline + amp*g(t; params).
- TIME NET NOTE: currently a SIREN (global sines) -- WRONG shape for a localized bolus transient.
  WIRE/Gabor atoms (localized bump) or explicit bolus atoms are the natural fit. (Spatial A_r is
  WIRE; temporal Phi_r is the small SIREN -- swapping Phi to WIRE/bolus atoms aligns with this.)
- WHY (for the goal): temporal DoF set by PHYSICS (~3-5 params/voxel), not an ad-hoc rank ->
  very strong temporal regularizer -> clean dynamics from FAR fewer spokes (the actual goal),
  plus interpretable PK maps (arrival time, rise rate) = what quantitative DCE wants.
- CRITICAL SEQUENCING: a bolus-shape prior UNDERMINES the aorta test as validation -- if we
  impose "sharp rise", NIK renders a sharp aorta trivially (circular; proves nothing about
  resolving it from data). So: (1) run the aorta test on the FLEXIBLE model FIRST (honest
  temporal-resolution proof); (2) THEN pursue the PK model for the fewer-spokes goal, where
  imposing the shape is a feature. OR make the prior SOFT (broad shape family + learned residual)
  so sharpness stays data-driven.
- RISK: model mismatch -- aorta(sharp AIF)/liver(slow)/kidney differ. Mitigate: flexible family
  + per-voxel params + residual term for real deviations (motion, recirculation, multiphasic).

## DE-PRIORITIZED (were blur fixes; blur was the render)
- Hash-grid encoding, higher FF bandwidth, radial |k|-warp (wire_ff_res_radial): all aimed
  at blur that was a render artifact. freq/depth sweep confirms they don't move quality much.
- SENSE / coils-as-output: still valid but only for the small residual grain, not blur.

## Built + committed (reference)
- WIRE_FF_SUBSPACE (factorized low-rank, rank knob) + warmstart_phi (PCA init).
- WIRE_FF_RES_RADIAL (|k|-warp diagnostic). Working model wire_ff_res untouched.
- Checkpointing (--resume, self-cleaning). Render fix (support_radius 1.0).
- Tooling: sweep_aggregate / freq_aggregate / rebase_aggregate / cs_heldout_loss (didn't validate).

## Constraints (user decisions)
- Stay in k-space for now (no image-domain learned priors).
- Don't touch the autoresearch orchestrator. No Co-Authored-By trailer in commits.
