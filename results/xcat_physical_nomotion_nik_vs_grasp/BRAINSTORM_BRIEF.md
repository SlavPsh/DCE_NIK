# NIK vs GRASP-Pro on undersampled radial DCE-MRI: state and open problem

Self-contained briefing for brainstorming next steps. All numbers inline; no external files needed.

## Goal
Test whether NIK (a neural implicit k-space model) reconstructs undersampled radial DCE-MRI (dynamic contrast-enhanced) more faithfully than compressed-sensing baselines (GRASP-Pro, classic CS), across three things: dynamic image quality, contrast-enhancement curves, and pharmacokinetic (PK) maps. Two datasets exist and both have been run: (1) a no-motion digital phantom with full ground truth (the quantitative core, below), and (2) real breathing in-vivo data with no ground truth (consistency checks, summarized in its own section). The eventual target is the motion + no-GT regime.

## Data and setup
- Substrate: XCAT physical no-motion simulation. Concentration -> R1 -> SPGR signal is the ground-truth image (`GroundTruth.img`). One kidney-bearing axial slice (z15) with a clean descending aorta.
- Acquisition: golden-angle stack-of-stars, 2D radial per slice after z-FFT. Readout 220, 8 coils, 344 temporal frames spanning ~180 s, frame length 0.52 s. Heavy angular undersampling: 7 acquired spokes/frame per slice.
- Reference = the XCAT SPGR truth image at each frame time. This is a rare setting: full spatiotemporal ground truth for a realistic DCE acquisition.

## Methods compared (all reconstruct the same measured data)
- GRASP-Pro: CS with a temporal-PCA subspace (rank K) + spatial-TV + temporal-TV, solved by nonlinear conjugate gradient. Encoding = NUFFT-SENSE with the true coil maps + ramp density compensation. K chosen by held-out cross-validation (see below) = 12.
- CS-file: the simulator's own recon (classic GRASP: full frames + spatial/temporal TV, no PCA subspace). Reference only.
- NIK variants (the method under test):
  - NIK-free: fully continuous temporal model.
  - NIK-subR: learned rank-R temporal subspace (R = 5, 12, 16), PCA-warmstarted.
  - NIK-F0: rank-3 fixed Patlak PK basis [AIF, integral(AIF), baseline], using the analytic ground-truth arterial input.

## Spoke budget (fairness, verified)
NIK and GRASP-Pro use the IDENTICAL input: 5 in-plane spokes/frame (the same golden angles 0-4). NIK reserves angles 5-6 as held-out validation/test; GRASP does not use them either. So NIK vs GRASP is spoke-matched. CS-file used all 7 spokes/frame (the sim's full recon, 40% more data); it is a reference, not a matched competitor.

## Metrics (all vs ground truth, per-frame then averaged over all 344 frames)
- Spatial: SSIM, PSNR, HaarPSI (perceptual), NRMSE, body-masked. Computed per frame, then averaged. The spatial ranking holds within every DCE phase including first-pass, so it is not a static-frame artifact.
- Temporal: per-ROI enhancement curve NRMSE (median-of-ROI, baseline-subtracted), plus aorta first-pass peak amplitude / time-to-peak / FWHM.
- PK: signal-domain Patlak (relative), ROI medians of Ktrans-analog and vp (vascular fraction).
- Data consistency: NMSE predicting the held-out spokes (angles 5-6), truth-blind.

## Headline results (fair K=12 GRASP baseline)

NOTE: these numbers are POST a 1px render-bug fix (see the open-problem section). NIK is seed-stochastic; GRASP-K12 is deterministic. SSIM/PSNR/HaarPSI at representative seed; sub12/free are 3-seed, sub16 2-seed.

Image metrics (higher SSIM/PSNR/HaarPSI better; lower NRMSE better):
| method | SSIM | PSNR dB | HaarPSI | NRMSE |
|---|---|---|---|---|
| GRASP-K12 (deterministic) | 0.980 | 40.7 | 0.888 | 0.010 |
| NIK-sub16 | 0.927 | 34.4 | 0.749 | 0.019 |
| NIK-sub12 | 0.921 (mean 0.918+-0.023) | 33.7 | 0.738 | 0.024 |
| NIK-free | 0.914 (mean 0.899+-0.034) | 33.0 | 0.669 | 0.024 |
| NIK-sub5 | 0.913 | 32.8 | 0.694 | 0.024 |
| NIK-F0 | 0.907 | 32.3 | 0.699 | 0.027 |
| CS-file (7 spokes) | 0.880 | 33.1 | 0.625 | 0.023 |

Best NIK (sub16) beats CS-file; residual gap to GRASP-K12 = 0.05 SSIM, 6.3 dB PSNR. NIK seed spread (0.023 SSIM, 3.4 dB) is COMPARABLE to that gap.

Contrast curves (curve-NRMSE mean+-sd, lower better) and aorta first-pass (truth peak 0.772, TTP 27.4 s, FWHM 11.5 s):
| method | aorta curve | cortex | medulla | aorta peak | dPeak | TTP s | dTTP |
|---|---|---|---|---|---|---|---|
| GRASP-K12 | 0.222 | 0.088 | 0.083 | 0.767 | -0.01 | 30.0 | +2.6 |
| NIK-sub16 | 0.087+-0.011 | 0.101 | 0.107 | 0.782 | +0.01 | 28.4 | +1.0 |
| NIK-free | 0.096+-0.037 | 0.039 | 0.031 | 0.874 | +0.10 | 27.9 | +0.5 |
| NIK-F0 | 0.077+-0.035 | 0.195 | 0.233 | 0.880 | +0.11 | 27.4 | 0.0 |
| NIK-sub12 | 0.295+-0.078 | 0.198 | 0.186 | 1.054 (bad seed) | +0.28 | 27.9 | +0.5 |
| CS-file | 0.375 | 0.142 | 0.025 | 0.452 | -0.32 | 26.9 | -0.5 |

Two failure modes vs truth: NIK wins whole-curve fidelity + timing (TTP delta ~0) but OVERSHOOTS the aorta peak ~+10% (real, survives ROI erode/dilate, seed-variable); GRASP is peak-exact but 2.6 s late. PK (signal-Patlak): NIK-sub16 best on tissue Ktrans (cortex 0.253/medulla 0.291 vs truth 0.234/0.280); GRASP under-reads aorta vp; sub12 PK unreliable (seed).

Held-out k-space prediction NMSE (truth-blind data fit, lower better): NIK-sub12 ~1.5e-4, NIK-sub5 1.9e-4, NIK-free 3.8e-4, GRASP-K12 1.1e-3, NIK-F0 8.8e-3.

## What is settled
1. GRASP's rank K must be chosen fairly. Variance-threshold selection gives K=3-5 and MISSES the bolus (a 0.004%-variance PCA component). Held-out k-space cross-validation, using only measured data, picks K*=12 (sharp minimum) and captures the bolus. Fair baseline = K=12, best GRASP images.
2. A 1px even-grid render bug (NIK only) had inflated the apparent spatial gap ~4x. After fixing it, best NIK (sub16) reaches SSIM 0.93, beating CS-file (0.88); the true residual gap to GRASP is 0.05 SSIM / 6.3 dB.
3. The residual spatial gap is dominantly a SMOOTH LOW-|k| BIAS (91% of NIK error energy is low-|k|; +4.2 dB recoverable by removing a smooth bias field), traceable to the k-space normalization envelope (not coil/SENSE). Identified, not yet corrected.
4. NIK is not yet reproducible at the comparison precision: seed spread (0.023 SSIM, 3.4 dB) is comparable to the GRASP gap. GRASP-K12 is deterministic.
5. Temporal: NIK wins whole-curve fidelity + timing on all ROIs but OVERSHOOTS the arterial first-pass peak ~+10% (a real bias, survives ROI erode/dilate); GRASP is peak-exact but 2.6 s late.
6. NIK fits the measured k-space BETTER than GRASP (held-out NMSE 1.5e-4 vs 1.1e-3). The spatial gap is not a data-fit problem and not a between-spoke-filling problem (a low-|k| bias lives where sampling is densest).

## Real in-vivo data (already run; no ground truth, so consistency not accuracy)
We also have real breathing in-vivo radial DCE (one kidney slice) reconstructed by NIK and CS. There is no ground truth, so these are consistency and plausibility checks against a CS reference, not accuracy:
- Temporal denoising (real, defensible): NIK's contrast curves are much cleaner than CS. The CS curve oscillation is broadband noise, not physiology (no respiratory peak in its spectrum; the oscillation shrinks as spokes/frame increase). NIK represents the few real temporal degrees of freedom far more smoothly. No accuracy claim (no GT), but a genuine denoising advantage. This matches the phantom finding that NIK wins whole-curve temporal fidelity.
- PK inter-slice consistency (supports quantification): NIK PK maps have coefficient-of-variation across adjacent slices comparable to or better than a CS-fit reference on every tissue, and agree with the CS-fit Ktrans (correlation ~0.77-0.92).
- Spoke-fraction frontier (CS-favored, but circular): measured as HaarPSI vs the CS full-spoke recon, NIK sits flat around 0.71, below CS at every spoke fraction. This metric is cs-likeness (reference = a CS recon, not truth), not fidelity, so it is weak evidence.
- Motion: kidney contrast-curve jitter is respiratory motion (self-gating navigator ~0.33 Hz), not a binning artifact. A gated, functional cortex/medulla segmentation was built.
Implication: the eventual target is this motion + no-GT regime, so a spatial fix for NIK should ideally transfer here and not depend on the phantom's clean coils or absence of motion.

## The open problem: NIK's residual spatial gap (SSIM 0.93 vs 0.98) after a 1px bug fix
History matters here, because two earlier readings were wrong and are retracted:
- The apparent gap was SSIM 0.77 vs 0.98. It turned out ~73% of that was a **1px even-grid render bug**: NIK aligned to truth with `im[::-1,::-1]`, a 180deg rotation about (N-1)/2 not the FFT centre N/2 (RO=220, even), a pure +1px translation present in NIK only. A translation is invisible to magnitude spectra, held-out k-space NMSE, and correlation curve metrics, and degrades SSIM smoothly, so it evaded every diagnostic until a geometry cross-correlation against the zero-shift GRASP reference caught it. Fixed with an exact integer roll; SSIM 0.77 -> 0.91-0.93. RETRACTED as artifacts of the shift: the "misplaced high-frequency content" reading and the "between-spoke filling" hypothesis.
- Ruled out (still valid): post-hoc TV denoising does not help (monotonically hurts), and subspace rank is not the lever (sub12 ~ sub16).

What the residual 0.05 SSIM / 6.3 dB gap actually is:
- It is dominantly a SMOOTH LOW-|k| BIAS. Decomposing NIK error vs truth: removing a smooth spatial bias field buys +4.2 dB (33.7 -> 37.9), cutting the PSNR gap to 2.9 dB. NIK error is 91% low-|k| (GRASP 67%); SSIM (locally normalized) forgives it, PSNR/HaarPSI count it (this is why the metrics disagree).
- The bias field is NOT coil/SENSE (corr with sum|b1|^2 = -0.02) but radial/central (corr +0.36 with radius, ~47% radially symmetric) = the signature of the k-space normalization envelope (envelope_exponent=0.75). Worth ~+4.2 dB POST-HOC. A 6-point envelope sweep (0.5-1.0, 3 seeds) shows it is NOT recoverable at source: |bias|rms is minimized exactly at the current 0.75, and pushing the envelope either way is a Pareto slide (env 1.0 costs aorta curve 0.10->0.25 for a noise-level SSIM bump). So the envelope knob is a DEAD END; the +4.2 dB is post-hoc-removable only.
- Secondary: NIK leaks error into the background (radial streaks: 19-47% of error energy outside the body vs GRASP 5%).
- Blocker: seed variance (0.023 SSIM, 3.4 dB) is comparable to the gap, so the residual is not yet reproducible at the comparison precision.

## NIK technical details (for proposing fixes)
- Model: coordinate MLP mapping (kx, ky, t, coil-embedding) -> complex k-space value. Architecture = WIRE / Fourier-feature SIREN-like. Depth 12, width 768, spatial Fourier-feature bandwidth k_sigma 2.5 (k_freq 256), temporal t_freq 32 / t_sigma 1.5, coil embedding dim 8.
- Training: k-space MSE loss on the 5 training spokes, FLAT weighting (no density compensation in the loss, dcf_power = 0, no focal weighting), under a k-space normalization with envelope_exponent 0.75 (partial high-|k| whitening). 40k steps, Adam lr 1e-5, weight decay 3e-3.
- Reconstruction: query the model on a Cartesian k-space grid, zero outside |k|>1, iFFT per coil, SENSE-combine with the true coil maps. No image-domain regularization anywhere.
- Prior knowledge from related work: envelope-only whitening tends to under-fit high-|k| (soft edges); full whitening amplifies high-|k| noise (a central blob). There is a sharpness-vs-noise sweet spot in the high-|k| loss weighting.

## Candidate directions (seeds for brainstorming, not exhaustive)
- Envelope / k-space normalization: RULED OUT as a source fix (6-point sweep, above). env=0.75 already minimizes the bias; the +4.2 dB is post-hoc-removable only. A learned or explicit smooth-bias-field correction (post-recon) is the remaining route to the +4.2 dB, but it is a correction, not a principled fix, and would not transfer as-is to the no-GT in-vivo regime.
- Seed variance reduction (blocker for any paper number): why is the spread ~ the gap? ensemble/averaging, longer training, better init, or the low-|k| bias itself driving it.
- Aorta peak overshoot: a real NIK bias (+10%, survives ROI erode/dilate). Where does the model over-amplify the brief high-signal first-pass? relates to temporal capacity / loss weighting on the arterial pixels.
- Architecture / representation: hash-grid or multi-resolution features, higher spatial capacity, complex-valued networks, better coil handling.

## Retractions (do not carry these forward)
- "Misplaced high-frequency content" and "between-spoke filling" were interpretations of the 1px shift, now retracted. The residual is a low-|k| bias, not a high-|k| / unmeasured-region problem.
- All pre-fix NIK spatial numbers (SSIM ~0.77) are superseded by ~0.91-0.93.

## Key open questions for the models
After a 1px render-bug fix, NIK reaches SSIM 0.91-0.93 (beats CS, gap 0.05 to GRASP), and the residual is dominantly a smooth low-|k| bias with the k-space-normalization-envelope signature (+4.2 dB, identified, unexploited). Two questions:
1. The +4.2 dB smooth low-|k| bias is post-hoc-removable but the envelope knob does NOT fix it at source (6-point sweep: 0.75 is already optimal, pushing it is a Pareto slide). What normalization or architectural change would remove a smooth low-|k| image bias WITHOUT reweighting the contrast baseline (low |k| carries it; higher DCF power hurt dynamics in vivo)? Is a post-hoc bias-field correction defensible, or does it just launder the problem?
2. NIK's seed spread (0.023 SSIM, 3.4 dB) is comparable to its gap from GRASP. What most likely drives this non-reproducibility, and how would you reduce it to make the comparison meaningful?
Secondary: NIK overshoots the arterial first-pass peak ~+10% (real, seed-variable) while winning whole-curve fidelity; what would remove the peak bias without flattening the curve?
