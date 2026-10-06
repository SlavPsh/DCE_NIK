# in vivo (meas_p3_dce, slices 21, k80 = 1368/1708 views (v%10<8), VAL v%10==8 for early stop, TEST v%10==9 untouched; same views for every method)

rulers: mf_* = NRMSE vs model-free NUFFT ROI curve on its 240-pt grid (affine = raw+affine fit; scale = baseline-subtracted single scale). physical bounds on aorta. *_kNMSE = complex k-space NMSE at held-out spokes (NIK only). CS rows are references, NOT truth; CS held-out blocked (magnitude-only files).

## slice 21
| metric | tofts8 mean±SD (n) | free mean±SD (n) | Δ free−tofts8 | GRASP-v2 k80 | GRASP-Pro f80match |
|---|---|---|---|---|---|
| cortex_peak_ratio | 0.846 ± 0 (1) | 0.7266 ± 0 (1) | -0.1194 | 0.7667 | 0.638 |
| cortex_washout_ratio | 0.9963 ± 0 (1) | 0.9092 ± 0 (1) | -0.08716 | 1.048 | 0.9215 |
| medulla_peak_ratio | 0.9481 ± 0 (1) | 0.928 ± 0 (1) | -0.02011 | 1.008 | 0.9193 |
| medulla_washout_ratio | 1.012 ± 0 (1) | 0.9901 ± 0 (1) | -0.02188 | 1.069 | 0.897 |
| aorta_peak_ratio | 0.9823 ± 0 (1) | 0.6945 ± 0 (1) | -0.2878 | 0.7097 | 0.2285 |
| aorta_washout_ratio | 0.9226 ± 0 (1) | 0.7123 ± 0 (1) | -0.2103 | 0.9795 | 0.7434 |
| mf_aorta_affine | 0.04527 ± 0 (1) | 0.1151 ± 0 (1) | +0.06985 | 0.1013 | 0.384 |
| mf_cortex_affine | 0.03566 ± 0 (1) | 0.05062 ± 0 (1) | +0.01496 | 0.04423 | 0.1056 |
| mf_medulla_affine | 0.02568 ± 0 (1) | 0.03886 ± 0 (1) | +0.01318 | 0.02758 | 0.07839 |
| mf_liver_affine | 0.02756 ± 0 (1) | 0.02702 ± 0 (1) | -0.0005336 | 0.02368 | 0.03818 |
| mf_aorta_scale | 0.05879 ± 0 (1) | 0.1639 ± 0 (1) | +0.1051 | 0.1419 | 0.5084 |
| mf_cortex_scale | 0.05393 ± 0 (1) | 0.07815 ± 0 (1) | +0.02422 | 0.06828 | 0.1632 |
| mf_medulla_scale | 0.03767 ± 0 (1) | 0.05744 ± 0 (1) | +0.01977 | 0.04174 | 0.1151 |
| aorta_peak_ratio_vs_mf | 0.9823 ± 0 (1) | 0.6945 ± 0 (1) | -0.2878 | 0.7097 | 0.2285 |
| aorta_fwhm_s | 15.36 ± 0 (1) | 21.5 ± 0 (1) | +6.144 | 52.22 | 98.3 |
| aorta_ttp_s | 64.73 ± 0 (1) | 66.27 ± 0 (1) | +1.536 | 64.73 | 207.6 |
| aorta_neg_frac | 0 ± 0 (1) | 0.1 ± 0 (1) | +0.1 | 0 | 0.02083 |
| aorta_rise_mono | 1 ± 0 (1) | 0.9268 ± 0 (1) | -0.07317 | 1 | 0.6466 |
| cortex_medulla_late_corr | -0.01326 ± 0 (1) | -0.1186 ± 0 (1) | -0.1054 | -0.1166 | 0.9698 |
| train_kNMSE | 0.2251 ± 0 (1) | nan ± nan (0) | +nan | nan | nan |
| val_kNMSE | 0.2455 ± 0 (1) | nan ± nan (0) | +nan | nan | nan |
| test_kNMSE | 0.25 ± 0 (1) | nan ± nan (0) | +nan | nan | nan |
| wall_s | 3860 ± 0 (1) | 2308 ± 0 (1) | -1552 | nan | nan |
| peak_gpu_mb | 1.246e+04 ± 0 (1) | 1.247e+04 ± 0 (1) | +12.81 | nan | nan |
| params | 5.531e+06 ± 0 (1) | nan ± nan (0) | +nan | nan | nan |

model-free aorta FWHM (s): 16.89584552369808

## reference peak correction

the 31-spoke model-free reference (6.8 s window) clips the first-pass peak; factor = peak at an 11-spoke window / peak at 31 spokes (`mf_peak_check.py`, per-spoke normalized, approved rois). corrected ratio = ratio vs the 31-spoke reference / factor. state this with every peak comparison.

| slice | roi | clip factor | free corrected peak ratio | tofts8 corrected peak ratio | GRASP-v2 corrected | GRASP-Pro corrected |
|---|---|---|---|---|---|---|
| 18 | cortex | 1.12 | - | - | - | - |
| 18 | medulla | 1.07 | - | - | - | - |
| 18 | aorta | 1.09 | - | - | - | - |
| 19 | cortex | 1.09 | - | - | - | - |
| 19 | medulla | 1.06 | - | - | - | - |
| 19 | aorta | 1.09 | - | - | - | - |
| 21 | cortex | 1.07 | 0.68 (raw 0.73) | 0.79 (raw 0.85) | 0.72 (raw 0.77) | 0.60 (raw 0.64) |
| 21 | medulla | 1.04 | 0.89 (raw 0.93) | 0.91 (raw 0.95) | 0.97 (raw 1.01) | 0.89 (raw 0.92) |
| 21 | aorta | 1.10 | 0.63 (raw 0.69) | 0.89 (raw 0.98) | 0.64 (raw 0.71) | 0.21 (raw 0.23) |
