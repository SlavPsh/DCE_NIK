# Physical no-motion XCAT: NIK vs GRASP-Pro / CS (slice z15)

**Substrate:** XCAT physical no-motion sim (`simulation_results_20260816T210718.mat`); concentration to R1 to SPGR truth (`GroundTruth.img`) is the accuracy reference. Slice truth z15 (kidney-bearing, clean descending aorta). Aligned stack-of-stars, z-FFT, per-slice 2D radial, golden-angle (1643 unique spoke angles). `labelGT` z-axis stored reversed vs truth (fixed: `labelGT[15-z]`).

## Methods
- **GRASP-Pro.** Temporal PCA subspace (rank K) + spatial-TV + temporal-TV, library NLCG solver (`cs_l1_nlcg_sptv`); NUFFT-SENSE operator with the true coil maps + ramp dcf, power-iteration-normalized. K=12, chosen truth-blind by held-out k-space cross-validation (rank-knob section). GROG is inapplicable (phase-less synthetic coils). **GRASP-Pro is deterministic**: one recon, one point. NIK is seed-stochastic, so this is a distribution-vs-point comparison.
- **CS-file.** The sim's own recon (`images.Recon`), classic GRASP, no PCA subspace. Reference; used all 7 spokes/frame (40% more data than NIK/GRASP).
- **NIK variants.** F0 (rank-3 fixed Patlak basis, GT AIF), sub5/sub12/sub16 (learned rank-R temporal subspace), free (continuous). NIK and GRASP-Pro are spoke-matched (5 spokes/frame, angles 0-4).
- **Render bug + guard (see PITFALLS 11).** The NIK image render aligned to truth with `im[::-1,::-1]`, a 180deg rotation about (N-1)/2 = 109.5 instead of the FFT/truth centre N/2 = 110 (RO=220, even) = a **+1.0 px shift, NIK only** (GRASP orient=id, zero shift). A pure translation is invisible to magnitude spectra, held-out k-space NMSE and correlation curve metrics, and degrades SSIM smoothly, so it survived undetected. Fixed with an exact integer roll (`np.roll(im[::-1,::-1],(1,1))`). Every recon now passes an assertion harness (`recon_asserts.check_recon`: geometry shift<0.1px vs a different-code-path reference, D4 orientation, Parseval, finite; failures raise). All render paths (image-lane, F0, GRASP NUFFT, fftc, in-vivo) carry a **measured** point-source centering verdict; audit in `CENTERING_AUDIT.md`.

All methods globally LS-scaled to truth. Contrast curves are median-of-ROI, baseline-subtracted enhancement.

## Image metrics (body-masked, vs truth). SSIM/PSNR/HaarPSI at representative seed; NRMSE mean+-sd
| method | SSIM | PSNR dB | HaarPSI | NRMSE |
|---|---|---|---|---|
| **GRASP-K12** (deterministic) | **0.980** | **40.7** | **0.888** | 0.010 |
| NIK-sub16 (2 seeds) | 0.927 | 34.4 | 0.749 | 0.019 |
| NIK-sub12 | 0.921 | 33.7 | 0.738 | 0.024 |
| NIK-free | 0.914 | 33.0 | 0.669 | 0.024 |
| NIK-sub5 | 0.913 | 32.8 | 0.694 | 0.024 |
| NIK-F0 | 0.907 | 32.3 | 0.699 | 0.027 |
| CS-file (7 spokes) | 0.880 | 33.1 | 0.625 | 0.023 |

The best NIK (sub16) reaches **SSIM 0.93, beating CS-file (0.88)**; residual gap to GRASP-K12 is 0.05 SSIM, 6.3 dB PSNR, 0.14 HaarPSI. (Before the render fix all NIK sat at 0.75-0.77; the 1px shift was ~73% of the apparent spatial gap.)

## Seed reproducibility (a finding, not a hedge)
NIK is single-seed-unstable. Over 3 seeds (sub12, free): SSIM spread 0.023 (sub12) / 0.034 (free); PSNR spread 3.35 dB (sub12); aorta-peak spread 0.24 (sub12: one seed overshoots to 1.05); aorta-curve spread 0.08-0.17. **The seed-to-seed spread (0.023 SSIM, 3.4 dB PSNR) is comparable to NIK's gap from GRASP (0.05 SSIM, 6.5 dB), so NIK's spatial performance is not yet reproducible at the precision of the comparison.** sub12 is high-variance (its representative-seed aorta is the bad overshoot); sub16 and free are steadier. What drives the spread (init, optimization, the low-|k| bias below) is open and blocks a single headline number.

## Spatial residual: smooth low-|k| envelope bias (C1 + D1)
SSIM (0.05 gap) and PSNR/HaarPSI (6.5 dB / 0.14 gap) disagree because the residual is not high-frequency structural loss. Decomposing NIK-sub12 error vs truth: global/per-frame/affine scale removal buys +0.7 dB; **removing a smooth spatial bias field (Gaussian sigma 8) buys +4.2 dB (33.7 -> 37.9)**, cutting the gap to 2.9 dB. NIK error is **91% low-|k|** (GRASP 67%); SSIM (locally normalized) forgives it, PSNR counts it. NIK also leaks error into the background (streaks: free 47%, sub12 19% of error energy outside the body, vs GRASP 5%). The bias field is **not coil/SENSE** (corr with sum|b1|^2 = -0.02) but radial/central (corr +0.36 with radius, ~47% radially symmetric) = the signature of the **k-space normalization envelope** (`envelope_exponent=0.75`). So the residual is a smooth low-|k| bias traceable to the normalization envelope, worth about +4.2 dB **post-hoc**. **A 6-point envelope sweep (0.5-1.0, 3 seeds) shows it is NOT recoverable at source: |bias|rms is minimized exactly at the current 0.75 (0.0088; all other values 0.0100-0.0111), and pushing the envelope either way is a Pareto slide (env 1.0 costs aorta curve 0.10->0.25 and cortex/medulla ~2x for a noise-level SSIM bump). So the +4.2 dB is post-hoc-removable only; the envelope knob is a dead end.**

## Contrast curves (curve-NRMSE mean+-sd, lower=better)
| ROI | NIK-F0 | NIK-sub16 | NIK-free | NIK-sub12 | GRASP-K12 | CS-file |
|---|---|---|---|---|---|---|
| aorta | 0.077+-0.035 | **0.087+-0.011** | 0.096+-0.037 | 0.295+-0.078 | 0.222 | 0.375 |
| cortex | 0.195+-0.008 | 0.101+-0.018 | **0.039+-0.024** | 0.198+-0.058 | 0.088 | 0.142 |
| medulla | 0.233+-0.009 | 0.107+-0.022 | **0.031+-0.011** | 0.186+-0.058 | 0.083 | 0.025 |

Aorta first-pass, signed vs truth (peak 0.772, TTP 27.4 s, FWHM 11.5 s):
| | peak | dPeak | TTP | dTTP | FWHM |
|---|---|---|---|---|---|
| truth | 0.772 | - | 27.4 | - | 11.5 |
| NIK-sub16 | 0.782 | +0.01 | 28.4 | +1.0 | 12.5 |
| NIK-free | 0.874 | +0.10 | 27.9 | +0.5 | 11.5 |
| NIK-F0 | 0.880 | +0.11 | 27.4 | 0.0 | 11.5 |
| GRASP-K12 | 0.767 | -0.01 | 30.0 | +2.6 | 10.4 |

**Two different failure modes measured against truth.** NIK wins whole-curve fidelity on all three ROIs (aorta 0.09 vs GRASP 0.22; cortex/medulla via free) and nails timing (TTP delta ~0). But NIK **overshoots the arterial first-pass peak by ~10%** (F0/free +0.10 to +0.11; sub12 up to +0.28, seed-variable); this survives ROI erode/dilate (16 to 108 px), so it is a real NIK bias, not partial volume. GRASP is peak-exact (-0.01) but **2.6 s late** (TTP 30.0) and slightly narrow. Note: the aorta ROI (33 px, high contrast) is sub-pixel-fragile; its curve/peak moved far more than the kidney ROIs under the 1px correction (C2: recon shift-sensitivity matched the truth-based estimate for cortex/medulla but was ~28x higher for the aorta).

## Choosing K fairly (rank knob)
Variance-threshold K selection gives K=3-5 and misses the bolus (a 0.004%-variance PC). **Held-out k-space cross-validation picks K*=12 from data alone** (`fig_kcv`), a sharp minimum: val-NMSE 2.0e-3 (K=8) -> 1.05e-3 (K=12) -> 4.5e-3 (K=16), i.e. K=12 is ~2x better than K=8 and ~4x better than K=16, so K is tightly determined. K=12 captures the bolus (peak 0.76 vs truth 0.77) and has the best GRASP images.

## PK maps (signal-domain Patlak, relative)
NIK-sub16 is the best PK variant (Ktrans cortex 0.253 / medulla 0.291 vs truth 0.234 / 0.280; vp reasonable), edging GRASP on tissue Ktrans. GRASP under-reads aorta vp (0.639 vs 0.772). sub12 PK is unreliable (representative seed overshoots: vp_aorta 1.005). CS-file worst on aorta vp (0.467). F0 under-reads kidney Ktrans (Patlak vs the phantom's Extended-Tofts).

## Bottom line
- **Spatial: GRASP-K12 leads, but the gap is small and mostly a known bias.** Best NIK (sub16) SSIM 0.93 beats CS (0.88); gap to GRASP 0.05 SSIM / 6.3 dB. The residual is dominantly a smooth low-|k| envelope-normalization bias (+4.2 dB post-hoc; a 6-point envelope sweep shows env=0.75 already minimizes it, so no source fix via this knob), not lost structure and not a between-spoke-filling problem (91% low-|k| error, where sampling is densest).
- **Reproducibility: NIK not yet at comparison precision.** Seed spread (0.023 SSIM, 3.4 dB) ~ the GRASP gap. GRASP is a deterministic point; NIK a seed distribution.
- **Temporal: NIK wins whole-curve fidelity + timing; overshoots the arterial peak (~+10%, real, seed-variable).** GRASP is peak-exact but 2.6 s late.
- **Quantification: NIK (sub16) edges GRASP on tissue PK.**

## Caveats
(1) Seed variance comparable to the spatial gap; single-seed numbers not paper-grade. (2) The +4.2 dB envelope bias is identified but recoverable only post-hoc (a 6-point envelope sweep confirms env=0.75 is optimal; the knob cannot fix it at source). (3) PK is signal-domain Patlak (relative). (4) NIK-F0 is Patlak vs the phantom's Extended-Tofts. (5) No-motion only; breathing is the next regime.

Figures: `fig1` montage, `fig2` error maps, `fig3` curves, `fig6` PK, `fig_kcv` held-out K selection, `fig_nav_spectrum`, `fig_ksweep_pareto`, `fig_2dgrid_pareto`, `fig_hikdiag` high-|k| held-out + power spectrum, `fig_bias_locate` smooth-bias field. Audit: `CENTERING_AUDIT.md`. Fair GRASP `arrays/grasp_recon.npz` = CV K=12.
