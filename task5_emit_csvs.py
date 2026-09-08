"""TASK 5: emit the tabular audit files (clean CSV quoting) from the four-subsystem code map."""
import csv, os
O = "/scratch/rnga/vvpshenov/DCE_NIK/results/task5_evaluation_code_audit"
def W(name, header, rows):
    with open(f"{O}/{name}", "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); [w.writerow(r) for r in rows]

# ---- pipeline_map.csv ----
W("pipeline_map.csv", ["Stage", "File", "Function/class", "Inputs", "Outputs", "Status", "Concern"], [
 ["XCAT anatomy/labels", "XCAT-ERIC/.../XCAT_to_MR_DCE.m", "XCAT_to_MR_DCE", "XCAT phantom, tissue LUT", "labelGT, tissue T1/T2", "correct", "labels piecewise-constant"],
 ["PK maps (truth)", "sim .mat", "results.pkLUT x labelGT", "ke,ve,vp,dt per tissue", "per-tissue PK", "partial", "per-tissue only, no voxel heterogeneity; Ktrans=ke*ve derived"],
 ["AIF", "XCAT-ERIC/.../contrast_curve_calc.m", "Cosine4AIF (Parker-style)", "Hct, cosine-4 params", "plasma AIF (mM)", "correct", "population AIF, hardcoded"],
 ["Concentration C(t)", "XCAT-ERIC/.../Cosine4AIF_ExtKety.m", "Cosine4AIF_ExtKety", "pk, AIF, t(min)", "C_t (mM)", "correct", "extended-Tofts, analytic conv"],
 ["MRI signal", "XCAT-ERIC/.../XCAT_to_MR_DCE.m:201", "SPGR steady-state", "C->R1, T1@3T, TR10ms, FA12", "S(x,t) real", "correct", "no B1, PD=1, no phase"],
 ["Coils", "XCAT-ERIC/.../Simulate_Coils.m", "Simulate_Coils", "FOV grid", "coilMaps [8,11,220,220] REAL", "correct", "real coils (no coil phase), fixed scanner coords"],
 ["Respiratory motion", "XCAT-ERIC MRXCATwERIC.mlapp", "CalculateNewGroundTruth", "20 XCAT resp-phase volumes", "phase-assigned anatomy", "implemented/disabled", "discrete 20-phase library, per-frame not per-spoke, no Jacobian, no dense field saved"],
 ["Radial k-space", "XCAT-ERIC/.../sampleKSpace.m", "sampleKSpace + NUFFT", "coil*signal, traj", "kspace.DCE [181,8,99,220]", "correct", "SNR=100 -> NO noise; noise model not proper CSCG"],
 ["Spoke timestamps", "sim .mat", "spokeTimingDCE [181,99]", "-", "per-spoke ms", "present/unused", "adapter uses per-frame time only"],
 ["--- NIK ---", "", "", "", "", "", ""],
 ["NIK data loading", "task4c_common.py:_dataset", "_dataset", "y100, kx,ky, mask", "X(2*traj),Y,T,coil", "correct", "coord=2*traj"],
 ["NIK timestamps", "xcat_adapter.py:slice_radial / task4c_common", "slice_radial", "Recon.timing", "true frame times; t=2t/Tt-1", "correct", "per-frame not per-spoke on XCAT path"],
 ["NIK normalization", "kspace_normalization.py:KSpaceNormalizer", "fit/normalize/denormalize", "train X,Y", "envelope(|k|)^0.75*gscale", "correct", "fit on train, applies to any coord"],
 ["NIK F0 basis", "train_grasp_nik.py:build_model / nik_model.py:WIRE_FF_PATLAK", "build_model", "aif_xcat.npz", "[AIF,intAIF,1] continuous", "correct", "signal-domain basis (not physical PK)"],
 ["NIK masks", "task4c_split.py", "spoke_masks.npz", "keep_f25", "train/val/test", "correct", "whole-spoke, disjoint"],
 ["NIK loss", "nik_focal_loss.py:composable_kspace_loss", "composable_kspace_loss", "pred,target reim", "complex MSE", "correct", "no dcf/focal in frozen config"],
 ["NIK grid+dynamic", "task4c_common.py:extract_pathC / nik_adapter.reconstruct_cartesian", "-", "model,coords,t,coil", "complex coil-combined signal", "correct", "continuous t query; NOT magnitude"],
 ["NIK Path C", "task4c_common.py:extract_pathC", "extract_pathC", "model amplitudes", "SENSE complex coeff maps", "correct", "canonical extraction"],
 ["--- GRASP-Pro ---", "", "", "", "", "", ""],
 ["GP data loading", "grasp_pro_py/precompute_ref.py:front_end", "front_end", "twix/kdata", "kdata_radial, traj, b1", "correct", "GROG gridding, not on-the-fly NUFFT"],
 ["GP frame def", "grasp_pro_py/precompute_ref.py", "reshape Proj=5", "spokes", "nt frames, frame_time centres", "correct", "trailing spokes discarded; Cut=15"],
 ["GP PCA basis", "grasp_pro_py/precompute_ref.py / cs_spoke_sweep.build_phi", "build_phi", "k-centre navigator", "Phi[nt,K=5]", "correct", "DISCRETE basis tied to nt grid"],
 ["GP regularization", "grasp_pro_py/cs_solver.py:cs_l1_nlcg_sptv", "cs_l1_nlcg_sptv", "E, Phi, TVweights", "subspace coeffs", "correct", "temporal TV 0.001 + spatial TV 0.0005"],
 ["GP coil combine", "grasp_pro_py/operators.py:Emat_GROG2Dksp._adj", "adjoint SENSE", "kdata, b1(Walsh)", "coil-combined", "correct", "Walsh b1 from GROG ref (DIFFERENT from sim b1)"],
 ["GP output", "grasp_pro_py/precompute_ref.py:151", "|PCA.H@recon|", "coeffs", "MAGNITUDE [bas,bas,nt]", "correct", "magnitude; frame count reruns inverse problem"],
 ["--- Evaluation ---", "", "", "", "", "", ""],
 ["Image metrics", "DCE_NIK/nik_metrics.py / masked_metrics.py", "PSNR/SSIM/NRMSE/HaarPSI/DISTS", "pred, ref", "scalars", "partial", "nik_metrics unmasked; masked_metrics correct; mixed PSNR-LS rescale in task2/consolidated"],
 ["ROI curves", "xcat_adapter.py:aorta_roi / task4_evaluate.py", "aorta_roi", "labelGT", "mean-over-ROI curves", "correct", "no motion-aware ROI"],
 ["Signal->concentration", "(none)", "-", "-", "-", "MISSING", "no S->C step anywhere"],
 ["PK fitting", "task4_csfit.py / task2_*.py:kt", "pinv(Phi)", "recon signal", "linear coeffs", "partial", "linear signal-basis fit, NOT physical PK; no fitter"],
 ["PK-map metrics", "task4c_common.py:truth_metrics", "truth_metrics", "coeff maps, truth", "R_aorta,intAIF,FP", "partial", "signal-basis coeffs only"],
 ["Held-out k-space", "task4c_common.py:kspace_nmse / task4_evaluate.heldout_nmse", "kspace_nmse", "model/theta, spokes", "NMSE + shells", "correct", "shells 0.3/0.7"],
])

# ---- reference_audit.csv ----
W("reference_audit.csv", ["Script/figure", "Current reference", "Valid role", "Correct label", "Required change"], [
 ["task2_haarpsi.py:3,38", "CS-f100", "implementation sanity check", "CS reconstruction (NOT truth)", "relabel; do not use as accuracy reference"],
 ["task2_wholepicture.py:1,3,55", "each method's own CS-f100", "data-consistency-ish", "CS reconstruction", "relabel 'gold-standard CS'"],
 ["haarpsi_spoke.py:1,49", "CS-100 full-spoke", "impl sanity check", "CS reconstruction", "relabel"],
 ["render_support_check.py:42", "CS-100", "impl sanity check", "CS reconstruction", "relabel"],
 ["compare_ranks_visual.py:22", "CS-100 static", "perceptual sanity", "CS reconstruction", "relabel"],
 ["task2_anatomy.py:38", "CS-f25", "n/a", "CS reconstruction", "relabel; not a reference"],
 ["task1a_matched_reference_audit.py", "CS-f100", "sensitivity audit", "CS reconstruction", "already acknowledged; keep as audit only"],
 ["nufft_bolus.py / task_R.py / score_vs_nufft.py / compare_all.py", "model-free NUFFT all-spoke", "descriptive best-estimate", "model-free NUFFT (honest)", "OK - keep honest label"],
 ["task4_simulate.py:35 / task4_evaluate.py:2-5", "XCAT theta_true / I_true", "ACCURACY GROUND TRUTH", "XCAT synthetic truth", "CORRECT - no change"],
 ["task4c_common.py:truth_metrics", "XCAT theta_true", "ACCURACY GROUND TRUTH", "XCAT synthetic truth", "CORRECT - post-selection only"],
 ["nik_recon.py:628 / train_multicoil_cart.py", "fully-sampled Cartesian sim image", "ACCURACY reference (synthetic)", "simulation ground truth", "CORRECT"],
 ["PLANNED XCAT experiment", "XCAT GroundTruth.img + pkLUT", "ACCURACY GROUND TRUTH", "XCAT physical truth", "USE THIS; not any recon"],
])

# ---- xcat_units_table.csv ----
W("xcat_units_table.csv", ["Quantity", "Symbol", "Unit", "Value/source", "File"], [
 ["kinetic model", "ext-Tofts", "-", "vp*cp+ve*ce", "Cosine4AIF_ExtKety.m:85"],
 ["PK time", "t", "minutes (fit)", "app seconds ->/60", "contrast_curve_calc.m:7,126"],
 ["AIF", "cp", "mM", "cosine-4 Parker (Hct 0.4)", "contrast_curve_calc.m:6"],
 ["concentration", "C", "mM", "-", "Cosine4AIF_ExtKety.m"],
 ["kep", "ke", "1/min", "pkLUT.ke", "contrast_curve_calc.m:67"],
 ["ve", "ve", "unitless", "pkLUT.ve", "-"],
 ["vp", "vp", "unitless (blood=0.6)", "pkLUT.vp", "-"],
 ["Ktrans", "ke*ve", "1/min", "DERIVED (not stored)", "-"],
 ["SPGR", "S", "a.u. (PD=1)", "sinFA(1-E1)/(1-cosFA E1)", "XCAT_to_MR_DCE.m:201"],
 ["TR", "TR", "ms", "10", "app UI"],
 ["flip angle", "FA", "deg", "12", "app UI"],
 ["baseline T1", "T1", "ms @3T", "per-tissue LUT", "XCAT_to_MR_DCE.m:95-168"],
 ["relaxivity", "r1", "1/(s*mM)", "3.5", "XCAT_to_MR_DCE.m:191"],
 ["R1 update", "R1", "1/s", "R1+r1*C", "XCAT_to_MR_DCE.m:192"],
 ["B1", "-", "-", "NOT modelled", "-"],
 ["noise", "-", "-", "SNR=100 -> none; model=magGauss*uniPhase (not CSCG)", "sampleKSpace.m:102"],
 ["phase", "-", "-", "object real; k complex from FFT only", "-"],
 ["signal scale", "-", "-", "GT normalized /max", "document.xml:741"],
])

# ---- method_input_audit.csv ----
W("method_input_audit.csv", ["Input", "NIK", "GRASP-Pro", "Identical?", "Concern/required change"], [
 ["complex k-space", "y100 subset (sha1 db2b5032e9c6)", "same kdata_radial (if fed same source)", "YES if same source", "ensure both read same k-space array"],
 ["trajectory", "kx,ky (2*traj)", "GROG grid (2*pi*traj, SIGN-1)", "same data, diff convention", "conventions equivalent; verify same raw traj"],
 ["spoke timestamps", "per-frame time (2t/Tt-1)", "frame-centre bins", "must match at frame centres", "evaluate at GRASP frame centres"],
 ["coil order", "sim b1 order", "SVD-compressed 8 vcc", "NO by default", "GP compresses coils; force same coil basis"],
 ["coil sensitivities", "sim b1 [220,220,8]", "Walsh from GROG ref (DIFFERENT)", "NO", "BLOCKING: force identical b1 for both"],
 ["image matrix / FOV", "N=220", "bas=nx/2 crop", "grid differs (crop)", "resample to common grid before metrics"],
 ["noise realization", "sim k-space (fixed)", "same k-space", "YES if same source", "OK"],
 ["input spokes", "train mask (first-m/frame)", "first-m/frame if frac<1", "YES if same keep-file", "use deterministic keep-file for NIK (not random split)"],
 ["acceleration", "f25 etc.", "spoke_frac knob", "must be set equal", "set identical frac"],
 ["kz ordering", "single-slice (stack-of-stars partition)", "same", "YES", "OK for 2D per-slice"],
 ["data scaling", "envelope-normalized internally (removed at output)", "b1/max internal", "output rescaled to truth by 1 global scale", "shared truth-derived scale at metric time"],
])

# ---- output_domain_audit.csv ----
W("output_domain_audit.csv", ["Method", "Raw output", "Coil combine", "Domain", "Normalized?", "Comparable as-is?"], [
 ["NIK dynamic", "complex k-space (denormalized)", "SENSE conj(b1)", "complex coil-combined signal", "no (1 global complex scale)", "take magnitude to match GP"],
 ["GRASP-Pro dynamic", "subspace coeffs", "adjoint SENSE (Walsh b1)", "MAGNITUDE coil-combined signal", "no (internal b1/max only)", "-"],
 ["NIK Path C coeffs", "amplitudes->IFFT->SENSE", "SENSE", "complex signal-basis coefficient maps", "no", "NIK-native; not a GP output"],
 ["GRASP-Pro coeffs", "recon_cs [bas,bas,K] (MATLAB only)", "-", "complex PCA-subspace coeffs (K=5)", "no", "different basis; not comparable to NIK F0 coeffs"],
 ["FAIRNESS RULE", "-", "-", "primary image cmp = MAGNITUDE coil-combined signal for BOTH", "one shared truth-derived scale", "convert NIK to magnitude; do NOT compare complex-NIK vs mag-GP"],
])

# ---- temporal_grid_manifest.csv ----
W("temporal_grid_manifest.csv", ["Aspect", "NIK", "GRASP-Pro", "XCAT truth", "Comparison policy"], [
 ["time representation", "continuous t (any physical time)", "discrete nt frame bins", "GroundTruth 0.1s grid + Recon frames", "evaluate all THREE at GP frame centres"],
 ["frame count effect", "query-only (no retrain)", "RERUNS inverse problem + new PCA basis", "n/a", "GP fixes the grid; NIK/truth resampled to it"],
 ["frame-centre time", "queried = GP frame_time", "mean(view_time) per bin", "same frame_time", "identical physical times"],
 ["spokes/frame", "all train spokes over continuous t", "Proj=5 (or NLINE=14)", "n/a", "document GP framing per run"],
 ["interpolation after recon", "native (continuous)", "NOT defined (discrete basis)", "linear on 0.1s grid", "do NOT densely query NIK vs coarse GP"],
])

# ---- spoke_split_audit.csv ----
W("spoke_split_audit.csv", ["Aspect", "NIK", "GRASP-Pro", "Status", "Required change"], [
 ["train spokes", "task4c train mask (angles 0,1)", "first-m/frame if frac<1 (contiguous)", "coincide only if same keep-file", "feed NIK the deterministic first-m keep-file, not random split"],
 ["validation spokes", "angles {2,4,6,8}", "n/a (GP has no val split)", "NIK-only", "GP consumes all frame spokes"],
 ["test/held-out spokes", "angles {3,5,7} untouched", "discarded remainder / none at f100", "comparable only via frac knob", "at frac<1 the dropped NLINE-m spokes are common held-out"],
 ["complete-spoke separation", "yes (whole spokes)", "yes (whole spokes/frames)", "OK", "-"],
 ["overlap", "disjoint (verified)", "n/a", "OK", "-"],
 ["shell metric", "inner<0.3/mid/outer>0.7 (r/0.5)", "not computed", "NIK-only", "apply same shells to GP forward residual if used"],
])

# ---- fairness_matrix.csv ----
W("fairness_matrix.csv", ["Difference", "NIK", "GRASP-Pro", "Class", "Action"], [
 ["temporal representation", "continuous t", "discrete PCA bins", "unavoidable method difference", "compare at GP frame centres"],
 ["temporal basis rank", "F0 rank-3 (AIF/intAIF/base)", "PCA K=5", "unavoidable method difference", "report both; do not equate coeffs"],
 ["coil handling", "sim b1, 8 coils", "Walsh from GROG, SVD 8 vcc", "removable implementation difference", "BLOCKING: use identical coil maps"],
 ["spatial grid", "N=220", "bas=110 crop", "removable implementation difference", "resample to common grid"],
 ["output domain", "complex signal", "magnitude signal", "evaluation difference", "magnitude for both"],
 ["frame times", "queryable to GP centres", "fixed bins", "evaluation difference", "match at GP centres"],
 ["normalization", "envelope-internal + 1 global scale", "b1/max internal", "evaluation difference", "one shared truth-derived scale"],
 ["data fraction / input spokes", "keep-file", "frac knob", "removable implementation difference", "set identical frac + keep-file"],
 ["regularization tuning", "loss only (no reg tune)", "TV weights fixed", "unavoidable (method-intrinsic)", "do not tune either; report defaults"],
 ["checkpoint selection", "val-NMSE selected", "NLCG fixed iters", "unavoidable method difference", "document both"],
 ["temporal interpolation", "native", "undefined", "unresolved confounder", "avoid dense-vs-coarse comparison"],
 ["PK fitting", "linear F0 coeffs (native)", "none (would need reconstruct-then-fit)", "unresolved confounder", "BLOCKING: build shared S->C->PK fit for both, or restrict claims"],
 ["masks / ROIs", "XCAT labels", "same XCAT labels (if applied)", "evaluation difference", "shared predefined masks"],
 ["reference domain", "XCAT truth", "XCAT truth", "OK if XCAT arm used", "never use a recon as truth (see reference_audit)"],
])
print("emitted 8 CSVs")
