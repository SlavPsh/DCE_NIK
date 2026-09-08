# DCE_NIK running pitfalls list

Metric/analysis traps caught the hard way. Each produced a plausible-looking wrong answer, not obvious garbage.

1. **Per-ROI normalization hiding a global offset.** (the worst one: it manufactured a publishable-looking anomaly.)
   Per-ROI PEAK-normalization `(c-base)/(peak-base)` divides each curve by its OWN peak. Real cortex has a sharp first-pass peak that deflates its normalized plateau (0.62); NIK blunts that peak so its plateau reads high (0.82). Result: a fake "NIK overshoots cortex" temporal failure AND a fake "CS overestimates cortex ~12%" substrate signal that motivated the entire kidney investigation. Both EVAPORATED under raw + ONE global affine (residuals 0.04-0.08, CS cortex bias -1 to -2%).
   FIX: work on RAW curves; fit ONE global affine per method across ALL ROIs simultaneously; per-ROI fitting is exactly what hides a global offset.

2. **Per-voxel model-free NUFFT for curve analysis.** Rotating golden-angle streaks give ~80 spurious SVD components; K5 residual ~1.0 everywhere. Useless per-voxel. FIX: ROI-average (kills streaks) or use a streak-free recon.

3. **Absolute K5-residual on MAGNITUDE curves.** CS's K=5 is a COMPLEX constraint; |complex K5| is not in any real 5-dim subspace (a CS bulk curve shows residual 0.6). FIX: confound-free test = does CS reproduce the streak-free curve, not a residual projection.

4. **Realized rank from the checkpoint `rank` field.** Nominal != realized. Always SVD the COMPLEX recon (R8-64 all realize rank 5).

5. **f100 as the comparison point.** All-spoke recons are degenerate/non-discriminative. f25 is the comparison point.

6. **AIF-FWHM as a binning-degradation anchor for a Patlak model.** The AIF shape is a FIXED basis function, so the model cannot broaden the bolus regardless of binning; the metric is structurally blind. Pick a metric the model can actually move.

7. **Slice 20 unreliable.** Reference bgE 0.479 (vs ~0.09 elsewhere), empty aorta ROI. Exclude.

8. **relres normalized by curve range inflates low-enhancement ROIs.** liver (little enhancement -> small denominator) shows big relative residual (NIK-full 0.30-0.35) that is tiny in ABSOLUTE terms (5e-9, cleanest of all methods). Report absolute alongside relative for low-dynamic-range ROIs.

9. **Comparing methods at different data levels.** first PK comparison plotted NIK-PK at f25 against CS-fit derived from cs_img (f100, ALL spokes) -> CS looked dramatically cleaner/more consistent. always compare at the SAME spoke fraction (f25-vs-f25). fair comparison flipped the consistency verdict (NIK better, not worse).

10. **Background-noise metric misses lost interior structure.** air-std/body-mean called F0 vs CS "comparable", but the real problem was grainy noise + dropped anatomy INSIDE the body (F0 pure-Patlak rigidity). use a structure metric inside the body (masked SSIM, interior gradient) + eyeball the maps, not just an air-region metric. also: report the USABLE config (F2), not the extreme (F0).

11. **1-pixel even-grid rot180 render shift (a pure translation is invisible to spectrum diagnostics).** The NIK image-lane render aligned to truth with `im[::-1,::-1]`, which rotates 180deg about (N-1)/2 = 109.5, not the FFT/truth centre N/2 = 110 (RO=220, even). Net = a +1.0px diagonal shift vs truth, present in NIK only (GRASP uses orient=id, zero shift). It deflated ALL NIK spatial numbers: SSIM 0.763 -> 0.921 (sub12), ~+0.16 across variants, closing ~73% of the apparent NIK-vs-GRASP spatial gap. Fix (exact, integer, no interpolation): `np.roll(im[::-1,::-1], (1,1), axis=(0,1))` = centered 180 rotation about N/2; verified regenerated recon registers to (~0,0) matching GRASP. LESSONS: (a) a pure translation is a linear k-space phase ramp, so it preserves the magnitude power spectrum EXACTLY while destroying SSIM -> spectrum-based blur/streak diagnostics are blind to it; the STEP-1 power-spectrum "misplaced high-freq" reading was this shift. (b) The GRASP zero-shift CONTRAST is what made it detectable -- always register a known-good reference under the SAME method. (c) Registration method matters: a hand-rolled upsampled-DFT (2a) mis-measured it as 0.25px; a robust Fourier-shift + masked-NCC scan gave the true +1.0px. Rule out geometry (shift/scale/rotation/phase-ramp) BEFORE any content diagnosis.

12. **Two NRMSE normalizations in the codebase differ by ~2x (global-range vs per-frame-range).** The DELIVERABLE numbers (aggregate / `step2_regen.py`) normalize per-frame RMSE by a single GLOBAL truth range `rv = Tr[body].max()-Tr[body].min()` (=0.94), giving e.g. sub16 = 0.019. But `xph_img_eval.py`/`xph_eval.py` (and ad-hoc diagnostics) normalize by the PER-FRAME range `Tr[:,:,t].max()-min()`, giving 0.039 for the SAME recon (early low-enhancement frames have a small per-frame range -> inflated). Neither is wrong, but they are NOT comparable. When quoting an absolute spatial NRMSE, state which convention and match the deliverable's global-range `rv` to line up with report.md. This surfaced when J2e single-seed (0.040, per-frame) disagreed with the stored `img_nrmse_mean_best` (0.019, global) for an IDENTICAL rec_best -> looked like corruption, was just the denominator. `rec_best` and the stored scalar ARE consistent under global-range.

13. **Forward-projecting the NIK render must use rot(dyn) (truth frame), NOT dyn (native).** finufft nufft-adjoint of the measured radial data lands in the TRUTH frame (grid-native aligns to truth at shift (0,0), corr 0.98), while the render dyn = ifft2c_mri(network) is in the NATIVE frame that needs the 1px-fix rot to reach truth (PITFALLS 11). So finufft and ifft2c_mri differ by exactly that rot. A render low-|k| data-misfit computed as forward(dyn) gave 10.1 dB (a FRAME ARTIFACT); the correct forward(rot(dyn)) gave 22.6 dB. Both are below the raw network's 39.5 dB (real render DC loss), but the frame bug nearly tripled the apparent loss and would have justified an aggressive DC step that in fact hurts (K3a). Rule: when mixing finufft and ifft2c_mri, put both in the same frame first (rot the ifft2c image), and sanity-check with RA.measure_shift before trusting any cross-pipeline k-space comparison.
