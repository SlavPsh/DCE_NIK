# RESUME: calibration + activation/encoding sweep (2026-09-01)

## DONE (in NIK_vs_CS_combined_report.ipynb, results/xcat_physical_nomotion_nik_vs_grasp/)
- sec 4-6: expressiveness sweeps (rank/w0/t_sigma, activation siren/gabor/gaussian). siren best, w0=90 sweet spot, rank saturates.
- sec 7: fair spoke-matched grasp (cs_slice21_f80match.npy via cs_nikmatch.py; grog.py has keep= mask). grasp medulla win flips when spoke-matched.
- sec 8: single-constant actual-signal CALIBRATION (calib_curves.py). KEY REFRAME: per-curve baseline-norm exaggerated spread. calibrated (body-anchor, pre-contrast t<40s): nik input==output on aorta (0.83 vs 0.84, the ~14% gap was an ARTIFACT); grasp 0.16->0.46; cortex/medulla near-identical. checks all pass (scale region-indep CoV 1.3%, pure-scale proof). notebook 61 cells.
- text trimmed terse (no em dash, lowercase). old sec 8 retrospective deleted per user.
- DCF_U f80match time-variation checked = negligible (0 ripple, corr ~0). no fix needed.
- memory updated: project_dce_grasp_spoke_fairness, project_dce_nik_expressiveness_sweep (w/ calibration correction), project_luna_gpu_contention.

## COMPLETED - serial chain 3076223-3076229 (all done, held-out NMSE)
gaussian s-sweep (ff, w0=30, rank16): s=0.3/0.6/1.0/2.0 -> 0.642/0.652/0.640/0.641. ALL ~0.64 = gaussian fails at EVERY s AND w0. under-tuned hypothesis FULLY DEAD.
raw-t encoding (--t_enc raw, w0=30, rank16): siren-raw 0.353 (== siren-ff 0.352, no change). gabor-raw 0.480 (WORSE than gabor-ff 0.281). gaussian-raw 0.638 (bad either way).
VERDICT: dropping FF does NOT let another activation win. siren best with or without FF; gabor worse raw; gaussian broken. FF-vs-raw does not change the activation ranking. all NEGATIVE results.

## TODO on resume (optional, low priority - all results are negative)
1. renders exist: _gaus_s03/06/10/20, _raw_siren/_raw_gabor/_raw_gaus (outcoil_subspace_output_*_slice21.npy). could add a terse bar-figure (held-out NMSE: gaussian-s-sweep flat ~0.64; raw-vs-ff siren==, gabor worse) to notebook sec 6 as a "we checked, nothing changes" note. else leave - findings already negative and captured here.
2. NOTE w0: raw experiments used w0=30 (not the sec-6 w0=90). siren-ff-w0=30 baseline=0.352 is the matched ref for siren-raw. gabor-ff at w0=30 was NOT run (sec-6 gabor was w0=90=0.281) so gabor raw-vs-ff is not perfectly w0-matched, but gabor-raw 0.480 is worse than everything anyway.

## KEY SCRIPTS (DCE_NIK/)
- outcoil_real.py: real NIK trainer. args --coilmode/model/rank/phi_w0/t_sigma/act/gauss_s/t_enc/tag. GPU render (parity 4e-7). reads REF slice_21. sbatch_ocr_sw.sh (env: STEPS/RANK/PHIW0/TSIG/ACT/GAUSS_S/TENC/TAG; --mem=24G; --time up to 02:00:00).
- calib_curves.py: post-hoc calibration + checks (luna-cpu-tiny). sweep_compare.py / step2_fig.py / step3_fig.py / step7_fig.py: figures (CPU sbatch, auto-discover output*_slice21.npy). DO NOT break these.
- cs_nikmatch.py (grasp_pro_py): spoke-matched grasp. grog.py keep= mask (backward-compatible).

## INFRA (see memory project_luna_gpu_contention)
luna-01 only GPU node, rnga account MaxMemoryPerAccount cap (lab-mates run big jobs -> pend + slow /scratch reads). use --mem=24G, CPU partitions for figures, serial chains. GPU render (not numpy CPU). "hangs" = slow FS, be patient not kill.

## NEXT DIRECTIONS (user plan, 2026-09-01 - not started)
1. add classic GRASP as an additional reference (alongside/instead of GRASP-Pro).
   note: apply the fairness machinery we just built - spoke-match to NIK's exact v%10<8 set (grog.py keep= mask, cs_nikmatch.py pattern) and the single-constant calibration (calib_curves.py). classic GRASP has its own DCF/FFT convention -> same one-constant treatment.
   note: memory says the phantom "cs-file" is already classic GRASP not Pro (project_dce_phantom_grasp_fix).
2. add + fully test extended Tofts (user will provide code base).
   ASSET: /net/beegfs/users/P101440/dce_kspace_sim/ (228K, kept in the 2026-09-03 cleanup) is a VALIDATED Tofts/AIF/SPGR reference - pharmacokinetics.py (Tofts + Parker AIF), mri_signal.py (SPGR signal eq + T1 mapping = the signal<->concentration step), kspace.py, sensitivity.py, test_framework.py (validates Tofts vs analytical in <30s). use it to sanity-check the incoming ext-Tofts code AND for the signal->concentration conversion.
   note: this directly addresses the known PK mismatch - phantom GT is ext-Tofts (efflux kep>0) but F0 fits Patlak (no efflux), biasing washout/Ktrans (project_dce_nik_pk_model_mismatch).
   note: ext-Tofts wants CONCENTRATION not signal -> couples to the calibration/concentration thread (SPGR signal model, needs sequence params + baseline T1).
3. run the whole pipeline on another dataset (generalization check).
   note: things currently hard-coded to this dataset/slice that would need parameterizing: ROI masks (aif_slice21/realkid_slice21), body mask (step2_slice21), spoke split v%10<8, NLINE=14/NT=122, TA=375, pre-contrast t<40s anchor window.
