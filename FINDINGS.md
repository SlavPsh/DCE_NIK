# DCE_NIK main findings, consolidated 2026-09-18

Standard input for every in vivo comparison: k80 (v%10<8, 1368 of 1708 views), val v%10==8, test v%10==9, slices 18 / 19 / 21 of meas_p3_dce, approved anatomical rois (both kidneys, cortex band / medulla interior, `results/realdata_nik_vs_cs_figures/rois_proposed_sl<Z>.npz`), reference = model-free 31-spoke nufft (`step2_slice<Z>.npz`), one global scale per method on the body. Phantom: XCAT no-motion, 5 of 7 spokes per frame, truth known. Every nik arm in a panel: same trainer (`train_grasp_nik.py`), same coil mode, same protocol, stated in the panel note.

## 1. The pk-basis amplitude deficit and its fix (2026-09-15 to 17)
- Symptom: nik-patlak / tofts curves at 50 to 70% of the reference kidney enhancement, static kidney 1.6 to 2x too bright. Not the basis (projection of the reference curves onto the atoms keeps 96 to 100%), not the aif, not the rois, not a gain.
- Cause, isolated by the amplitude-vs-step test (`tofts_amp_track.py`): (a) the tofts atoms were unit-norm, rms 0.054, 18x smaller than the patlak atoms, so the head had to output 18x larger coefficients against weight decay and lr 1e-5; (b) the held-out k-space early stop restored step 2000, before the dynamics converge (the held-out mse rises from step 2000 in every run while the curves improve: it is noise-fit dominated and selects the worst dynamics); (c) weight decay shrinks the dynamic coefficients.
- Fix = the in vivo pk-arm protocol now: unit-rms atoms (`basis_sl<Z>{,_r8}_rms1.npz`), 10k steps, final weights kept (`--no-restore`), wd 3e-3 (1e-2 slightly better), 3 seeds. Result, slice 21 cortex peak / washout vs reference: tofts8 0.81 / 0.97 (was 0.54 / 0.73); curve nrmse cortex 0.059, medulla 0.043 (grasp 0.075 / 0.083, grasp-pro 0.218 / 0.130). tofts8 is the best kidney-curve method on all three slices on both rulers. Excluded: lr 1e-4, envelope 0, dcf loss weighting, cosine schedule alone.
- The 31-spoke reference itself clips the first-pass peak by 7 to 13% (`mf_peak_check.py`), so peak ratios of 0.85 to 0.93 vs the reference are still below the true peak; washout ratios are unaffected.

## 2. Image quality vs curve fidelity (2026-09-17 to 18)
- After the fix tofts8 lost sharpness at the first-pass frame vs grasp (HaarPSI 0.716 vs 0.740 against the all-spoke anatomy) and gained streak texture (air energy 0.180 vs 0.134); flat-tissue noise and kidney edges equal or better. The streak texture is identical across seeds (seed average leaves air energy unchanged): deterministic aliasing, not noise fitting. Step count is not the knob: noise fit and dynamics grow together along training.
- Remedies tested on tofts8 sl21 (`iq_track_sl21_*.md`, figure 53): weight decay 1e-2 (modest gain on all four numbers); coefficient-map huber tv 0.3 (HaarPSI 0.746 > grasp, air 0.169, curves unchanged); **k-space support prior v2** (energy of the coefficient images in the air ring inside the crop over the body energy, weight 1): air energy 0.133 = grasp 0.134, HaarPSI 0.730 to 0.736, curves unchanged (cortex nrmse 0.063 to 0.074, medulla 0.044 to 0.046). Support v1 (whole-field mean) and pisco (both own-coil and cross-coil stencils) were inert.
- Candidate protocol for the production rerun: tofts8, unit-rms atoms, wd 1e-2, support prior 1, 10k steps, no restore; expected: curves at the current level, streaks at grasp level. Not yet run on 3 slices x 3 seeds.

## 3. Coil parameterization (2026-09-17 to 18)
- Inconsistency found and fixed: the in vivo NIK-sub16 had been an output-coil model from a different trainer and protocol. Rule from now on: every non-studied difference between compared arms is flagged and approved (memory feedback_flag_arm_inconsistencies).
- sub16 rerun as input-coil with the pk-arm protocol: sharpest image of all (HaarPSI 0.807, air 0.157) but oscillating curves (cortex nrmse 0.18, medulla 0.16) and a held-out k-space error 5x lower than tofts8: the learned basis reproduces temporal content of the held-out spokes that no tofts atom can (respiration and other non-kinetic dynamics, averaged out by the 6.8 s reference window). This is why the k-space ruler ranks the pk arms last and why the pk basis is the right model for kinetics and the wrong one for everything else.
- tofts8 with the output-coil head (`--coil-mode output`, shared backbone, one head per coil): equal or slightly better than input-coil on every slice (cortex peak 0.84 / 0.87 / 0.90 vs 0.81 / 0.84 / 0.88, aorta nrmse 0.056 vs 0.097, HaarPSI 0.736 vs 0.716, air 0.170 vs 0.180), same held-out error, cheaper. Figure 54.

## 4. Phantom (truth known)
- Fitted extended-kety maps: nik-free and nik-tofts within 5 to 15% of truth on the kidneys (cortex ktrans 0.61 / 0.54 vs 0.63, ve 0.87 / 0.76 vs 0.90, vp 0.12 / 0.10 vs 0.10); grasp and grasp-pro lose vp (0.04) and bias ktrans / ve by 20 to 40%. Coefficient-domain fit of nik-tofts agrees with the image-domain fit to 1 to 3% (26 s for 6138 voxels): the parameters live in the coefficients.
- Open: all nik arms over-read the aorta washout by about 7% (not partial volume, not dcf weighting, present in nik-free); the phantom tofts run still uses unit-norm atoms and should be redone with the new protocol.

## 5. Reference and ruler caveats to state on every slide
- In vivo has no truth: the reference is a low-pass, dcf-dependent nufft; grasp shares its lineage (nufft, ramp dcf, temporal tv only) and therefore tracks it; grasp-pro is a grog pathway. Agreement with the reference is not accuracy.
- The flattened held-out k-space mse must never select checkpoints or ranks. The affine / scale curve rulers hide amplitude; quote peak and washout ratios next to them.
- One subject, three slices; free breathing, no motion handling on either side.

## 6. Next steps (memory project_dce_nik_plan_after_045, updated)
1. Production rerun with the candidate protocol (wd 1e-2 + support prior 1), 3 slices x 3 seeds, input-coil and output-coil tofts8; redo the in vivo pk maps (figure 47) and the phantom tofts run.
2. First-pass peak (0.81 to 0.9 of the reference, true peak higher still): per-atom weight decay or bandwidth, test on the phantom.
3. Per-voxel span-loss map (where the tofts basis is the wrong model).
4. Oblique readout of vp / Ktrans from the coefficients (phantom test vs the nonlinear fit).
5. PISCO with the group's guidelines if they differ from the implementation here (both stencils were inert).

Files: tables `results/tofts_vs_patlak/{invivo_k80_rms1,invivo_k80_oc,iq_track_sl21_*,span_diag_sl*_rms1,mf_peak_check_sl*}.md`; figures `results/tofts_vs_patlak/figures/` and `results/realdata_nik_vs_cs_figures/figures/`; presentation folder `presentation_figs_2026-09-10/` (README index, figures 45 / 48 / 49 to 54); full chronology in RESUME.md.
