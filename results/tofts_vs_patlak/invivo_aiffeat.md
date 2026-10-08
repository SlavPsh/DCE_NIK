# in vivo (meas_p3_dce, slices 21, k100 = every view in training (no held-out spokes; the val / test kNMSE columns are TRAIN-set numbers here); same views for every method)

rulers: mf_* = NRMSE vs model-free NUFFT ROI curve on its 240-pt grid (affine = raw+affine fit; scale = baseline-subtracted single scale). physical bounds on aorta. *_kNMSE = complex k-space NMSE at held-out spokes (NIK only). CS rows are references, NOT truth; CS held-out blocked (magnitude-only files).

## slice 21
| metric | free mean±SD (n) | sub16 mean±SD (n) | Δ sub16−free | GRASP-v2 all spokes |
|---|---|---|---|---|---|
| cortex_peak_ratio | 0.7411 ± 0 (1) | 0.7476 ± 0 (1) | +0.006551 | 0.7881 | nan |
| cortex_washout_ratio | 0.9056 ± 0 (1) | 0.9437 ± 0 (1) | +0.03812 | 1.053 | nan |
| medulla_peak_ratio | 0.9304 ± 0 (1) | 0.9094 ± 0 (1) | -0.021 | 1.015 | nan |
| medulla_washout_ratio | 0.9968 ± 0 (1) | 0.99 ± 0 (1) | -0.006803 | 1.072 | nan |
| aorta_peak_ratio | 0.8217 ± 0 (1) | 0.7468 ± 0 (1) | -0.07485 | 0.7702 | nan |
| aorta_washout_ratio | 0.8231 ± 0 (1) | 0.9846 ± 0 (1) | +0.1615 | 1.012 | nan |
| mf_aorta_affine | 0.0834 ± 0 (1) | 0.1507 ± 0 (1) | +0.06732 | 0.08901 | nan |
| mf_cortex_affine | 0.04302 ± 0 (1) | 0.06721 ± 0 (1) | +0.02419 | 0.03944 | nan |
| mf_medulla_affine | 0.03619 ± 0 (1) | 0.06822 ± 0 (1) | +0.03203 | 0.02517 | nan |
| mf_liver_affine | 0.02744 ± 0 (1) | 0.03587 ± 0 (1) | +0.008427 | 0.02321 | nan |
| mf_aorta_scale | 0.1117 ± 0 (1) | 0.1948 ± 0 (1) | +0.08309 | 0.1242 | nan |
| mf_cortex_scale | 0.06492 ± 0 (1) | 0.1014 ± 0 (1) | +0.03645 | 0.06079 | nan |
| mf_medulla_scale | 0.05551 ± 0 (1) | 0.1017 ± 0 (1) | +0.04615 | 0.03846 | nan |
| aorta_peak_ratio_vs_mf | 0.8217 ± 0 (1) | 0.7468 ± 0 (1) | -0.07485 | 0.7702 | nan |
| aorta_fwhm_s | 18.43 ± 0 (1) | 27.65 ± 0 (1) | +9.216 | 35.33 | nan |
| aorta_ttp_s | 64.73 ± 0 (1) | 64.73 ± 0 (1) | +0 | 64.73 | nan |
| aorta_neg_frac | 0.004167 ± 0 (1) | 0 ± 0 (1) | -0.004167 | 0 | nan |
| aorta_rise_mono | 1 ± 0 (1) | 0.875 ± 0 (1) | -0.125 | 1 | nan |
| cortex_medulla_late_corr | -0.1888 ± 0 (1) | 0.1729 ± 0 (1) | +0.3617 | -0.1108 | nan |
| train_kNMSE | nan ± nan (0) | nan ± nan (0) | +nan | nan | nan |
| val_kNMSE | nan ± nan (0) | nan ± nan (0) | +nan | nan | nan |
| test_kNMSE | nan ± nan (0) | nan ± nan (0) | +nan | nan | nan |
| wall_s | 2348 ± 0 (1) | 3971 ± 0 (1) | +1623 | nan | nan |
| peak_gpu_mb | 1.249e+04 ± 0 (1) | 1.248e+04 ± 0 (1) | -13.52 | nan | nan |
| params | nan ± nan (0) | nan ± nan (0) | +nan | nan | nan |

model-free aorta FWHM (s): 16.89584552369808

## reference peak correction

the 31-spoke model-free reference (6.8 s window) clips the first-pass peak; factor = peak at an 11-spoke window / peak at 31 spokes (`mf_peak_check.py`, per-spoke normalized, approved rois). corrected ratio = ratio vs the 31-spoke reference / factor. state this with every peak comparison.

| slice | roi | clip factor | free corrected peak ratio | sub16 corrected peak ratio | GRASP-v2 corrected | GRASP-Pro corrected |
|---|---|---|---|---|---|---|
| 18 | cortex | 1.12 | - | - | - | - |
| 18 | medulla | 1.07 | - | - | - | - |
| 18 | aorta | 1.09 | - | - | - | - |
| 19 | cortex | 1.09 | - | - | - | - |
| 19 | medulla | 1.06 | - | - | - | - |
| 19 | aorta | 1.09 | - | - | - | - |
| 21 | cortex | 1.07 | 0.69 (raw 0.74) | 0.70 (raw 0.75) | 0.74 (raw 0.79) | - |
| 21 | medulla | 1.04 | 0.90 (raw 0.93) | 0.88 (raw 0.91) | 0.98 (raw 1.02) | - |
| 21 | aorta | 1.10 | 0.74 (raw 0.82) | 0.68 (raw 0.75) | 0.70 (raw 0.77) | - |
