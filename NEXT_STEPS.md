# DCE_NIK — next steps (rewritten 2026-07-18 after the render-bug finding)

## 2026-07-22 NUFFT REFERENCES (method-neutral) — biggest finding so far

Built finufft references for slice 13 so NEITHER method defines the target (fixes the
"HaarPSI vs CS only measures CS-likeness" flaw). nufft_reference.py / nufft_bolus.py.
Sign convention SETTLED: finufft frame is 180deg vs grasp -> use sign=-1 (corr 0.9990 vs
CS mean; sign=+1 gives 0.317). DCF = ramp |k| with floor; SENSE-combined with the SAME b1.

### A/B spatial: CS is genuinely closer to the neutral reference
  all-spokes NUFFT vs temporal mean : NIK 35.80 dB / HaarPSI .955 | CS 40.84 dB / .985
  pre-contrast NUFFT (240 sp, t<53s): NIK 33.15 dB / .888        | CS 35.37 dB / .918
So the earlier reference-bias caveat is GONE and CS still wins spatially (~5 dB, ~2 dB).

### Temporal GT in the pre-contrast window (truth = ZERO drift): NIK wins
  spurious drift  NIK +0.041 %/s (resid 2.22%)  vs  CS +0.090 %/s (resid 3.01%)
First non-circular temporal number in the project. Narrow but clean.

### C. MODEL-FREE bolus (sliding-window NUFFT) => BOTH METHODS SMEAR THE BOLUS ~7-10x
Reference self-validates: window widths 21/41/81 spokes (4.6/9.0/17.8 s) give the SAME
shape (rise ~50s, peak 62s, fall to 0.3 by 80s, recirculation bump ~92s).
  FWHM   reference 14.0s | NIK full-rank 45.1s | CS(K=5) 96.1s | NIK R=16 139.7s | R=5 130.9s
  TTP    reference 62.3s | full-rank 63.8s     | CS 68.2s      | R=16 69.3s      | R=5 77.0s
=> **THE RANK CAGE IS THE MAIN CAUSE OF THE SMEARING.** Removing it (full-rank) cuts FWHM
140s->45s and nails TTP. Verified VISUALLY (figures/bolus_rank_cage.png): full-rank really
descends after the peak and even shows the recirculation bump; R=16/CS plateau high through
75-125s. Not a noise artifact, and full-rank is SMOOTHER than R=16 here.
=> Confirms the mechanism predicted from the navigator analysis: Phi is built from the
bulk-dominated k=0 navigator, so the aorta's sharp spike is a minority variance mode and is
not representable. NIK inherits this via its PCA warm-start + a k-space loss dominated by
bulk signal (the aorta contributes ~nothing to the objective).

### CONSEQUENCE: our baseline choice was driven by a metric we KNEW was misleading
We picked the factorized low-rank model as baseline on held-out MSE (0.188 vs 0.325) and
swing. Against a neutral reference the WORST held-out model (full-rank) has BY FAR the best
bolus. This is exactly the warning in train_grasp_nik's docstring. Baseline NOT changed yet
(2026-07-22) -- pending full-rank image-quality check vs the NUFFT references.
Residual gap: full-rank is still 45s vs true 14s. Suspects: temporal encoding bandwidth
(t_freq/t_sigma) and the bulk-dominated k-space loss.
CAVEATS: slice 13, 21-voxel aorta ROI, single seed; pre-contrast reference uses 240 spokes
(below the ~302 Nyquist) so it carries some streaking.

## 2026-07-21 CONCLUSIONS (spoke frontier + metric corrections + rank semantics)

### A. "NIK ~= CS spatially" was measured on the TEMPORAL MEAN and is inflated
Same NIK, same reference, two ways of scoring:
  temporal-mean image  HaarPSI 0.964   <- what we had been quoting
  per-frame, averaged  HaarPSI 0.709   <- the relevant number for a DYNAMIC method
Reproduces across both code paths (piq + eval.image_metrics) and both CS references
(12-frame arm1 and 122-frame f100), so it is NOT a normalization/reference artifact.
The mean averages away per-frame differences. Standard spatial metric is now PER-FRAME
HaarPSI (mean + worst frame), resampled to a common frame grid.
CAVEAT: reference is CS itself -> measures CS-LIKENESS. A non-CS method cannot reach 1.0,
and per-frame CS-100 is itself noisy, so 0.709 is not "29% wrong".

### B. Spoke-reduction frontier (100/71/50/36/29 %, IDENTICAL spokes, slice 13)
Harness: shared keep-file (first m=round(14*frac) spokes per frame); NIK trains on ALL
acquired spokes with NO heldout, so NIK and CS consume the same data. CS reconstructed
from saved arrays (no twixtools), validated corr 1.000 vs cs_recon_3s.
Per-frame HaarPSI vs CS-100:  NIK 0.709/0.709/0.697/0.619/0.631
                              CS  1.000/0.922/0.857/0.810/0.785
- CS is HIGHER at every fraction. No crossover in the tested range.
- Images and metric DISAGREE: at 29% CS visibly streaks while NIK stays coherent, yet CS
  scores higher. Unresolved -- do not present either as a result.
- DO NOT quote "NIK -11% vs CS -21% degradation": CS starts at 1.000 by construction
  (self-comparison), so its drop is inflated. That comparison is invalid.
- NIK barely changes 100%->29% => output is PRIOR-DOMINATED (not using the extra spokes);
  "graceful degradation" and "over-regularized" are indistinguishable here.

### C. Rank R in NIK is SOFT; K in CS is HARD (important, changes earlier comparisons)
NIK's Phi comes from a free SIREN: warm-started from the PCA basis but then trained with
NO orthonormality/scale constraint. Factorization is only defined up to an invertible RxR
transform (A M^-1)(M Phi), so individual Phi_r are not interpretable and scale is arbitrary.
Effective rank (SVD of the recon, 99% energy):
  CS (K=5)      5   <- exact, data is PROJECTED into a fixed 5-D subspace
  NIK R=5      12   <- exceeds its nominal rank
  NIK R=16     13
  NIK full     27
=> "R" is a nominal upper bound, not a real DoF cage. When R=5 matched CS's 38% swing the
two were NOT equally constrained (NIK used ~12 effective dims). Any rank-vs-K comparison
made before this is suspect. FIX (untried): orthonormalize Phi (QR/Gram-Schmidt or a
penalty) so R means the same thing in both.

### D. GRASP-Pro facts established (for reference)
- Spatial quality is INDEPENDENT of temporal binning: unknowns are the K coefficient maps,
  not nt frames. Verified: 342 vs 122 vs 12 frames -> HaarPSI 0.992/1.000/0.990, same
  sharpness, and SVD rank identical (3 @90%, 4 @99%). Finer binning adds NO information.
- Binning is a temporal-resolution vs navigator-SNR tradeoff on a fixed 0.219 s stream
  (one k=0 sample per spoke, boxcar-averaged over nline). Also: nt x nt covariance from
  100*nc rows becomes ill-conditioned if nt grows too large.
- Phi is GLOBAL (one basis for all 27 slices) but coefficients are per-voxel/per-slice, so
  slices/voxels still differ (aorta vs liver in slice 13: correlation 0.22). The real limit
  is that all time courses must lie in the shared 5-D span.
- Navigator = k=0 sample of every spoke -> FFT along kz (zero-padded 32->100, pure
  interpolation, no new information) -> magnitude (phase DISCARDED) -> PCA over time.
  It is bulk-signal dominated, so small structures (aorta) barely contribute to the basis.
- Partial Fourier here: 22 of 32 kz partitions acquired (69%), 10 zero-filled (31%).

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

## SETTLED baseline (re-baseline at support=1.0, slice 13, 2026-07-18)
Recipe: **wire_ff_subspace rank 16**, k_sigma 2.5, w0 62, depth 12, dcf0, env0.75, 40k, 70/30.
Numbers (held / swing% / navcorr / HF-vs-CS / DISTS / HaarPSI, perceptual vs CS-100):
  full-rank 0.325 / 47.5 / .983 / .62 / .030 / .964   (highest swing but OVERFITS: worst held)
  R=5       0.199 / 38.2 / .958 / .71 / .035 / .956   (= CS rank-5 cap: swing ~= CS 37.5)
  R=10      0.189 / 43.8 / .972 / .70 / .034 / .958
  R=16      0.188 / 43.5 / .969 / .70 / .024 / .964   <-- PICK (best perceptual + balance)
  R=20      0.182 / 43.2 / .973 / .69 / .033 / .963
  CS-70       -   / 37.5 / .982 /1.03 / .018 / .976
  CS-100      -   / 37.5 / .989 /1.00 / .000 /1.000
Conclusions (all render-corrected):
- SPATIAL: NIK ~= CS-70 now (HaarPSI .96 vs .976, DISTS .024 vs .018). Small residual grain, NOT blur.
- TEMPORAL: DoF knee holds -- R=5 swing 38%(=CS cap), R>=10 swing 43-44%(>CS). Nav-corr high.
- GENERALIZATION: factorized (held .18-.20) >> full-rank (.325).
- CAVEAT: this is STATIC (temporal-mean) vs CS-100, slice 13. The temporal-resolution CLAIM
  is still UNPROVEN until the aorta bolus test (item 1 below).

## In flight
- **Re-baseline at support=1.0** (results_rebase_*, waiter): full-rank + R=5/10/16/20 vs
  CS-70/CS-100, unified table (held-out/swing/nav/HF/DISTS/HaarPSI). Confirms NIK~=CS + temporal win.

## AORTA BOLUS TEST -- DONE, and it's a NEGATIVE for the temporal claim (2026-07-18)
The non-circular temporal-resolution check: apply fixed aorta+liver ROIs (segmented on CS-100,
saved aorta_roi/liver_roi.npy) to NIK vs CS, compare the recovered bolus.
RESULT: NIK at fine temporal res (R=5 AND R=16) is AS NOISY AS CS-fine at the aorta -- it does
NOT resolve the sharp bolus more cleanly. R=5 is WORSE (noisiest). Only CS-coarse is clean (but
temporally smeared). aorta noise: NIK-R16 0.071, CS-fine 0.069 (equal); NIK-R5 0.23 (worse).
=> NIK has NO demonstrated temporal-resolution advantage as-is. The nav-corr "win" was the
circularity + smooth-navigator artifact. NIK's extra temporal DoF is largely NOISE (matches the
DoF-is-partly-noise finding). Rank constrains #patterns, NOT temporal smoothness of a voxel curve.
MECHANISM: temporal encoding (t_freq 32, t_sigma 1.5) + flexible SIREN Phi has enough temporal
bandwidth to fit frame-to-frame noise.
=> A temporal win REQUIRES an explicit temporal DENOISER: the PK/bolus-shape prior (soft, so
sharpness stays data-driven) or a temporal-smoothness regularizer / lower t bandwidth. The PK
model is now the CRITICAL PATH, not optional.
CAVEATS: one small ROI (21 vox, noisy), one slice, 2 ranks -- but the RELATIVE comparison is fair
(same ROI), NIK is simply not cleaner than CS. NIK used 70% spokes vs CS 100%.

## CURRENT HONEST STATUS
- SPATIAL: NIK ~= CS peer (render fix), better generalization. REAL, stands.
- TEMPORAL: NO demonstrated advantage yet. Needs a temporal prior (PK/smoothness) -> critical path.

## OPEN next steps (ranked, corrected)
0. **PK / temporal-smoothness prior -- NOW THE CRITICAL PATH** (see the PK thread below). It is
   what could produce a clean sharp DATA-DRIVEN bolus that CS-fine can't -> the temporal win.
   Try: (a) soft temporal-smoothness reg (penalize d2/dt2 of the curve), cheap; (b) lower t_sigma
   (less temporal bandwidth -> smoother, but E8 showed it costs swing -- find the balance);
   (c) the bolus-shape basis (gamma-variate) as a SOFT prior + residual. Re-run the aorta test after.
1. **AORTA BOLUS TEST -- DONE (negative). Re-run after the temporal prior** to see if it flips. Breaks the nav-corr circularity (NIK trains on the
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
