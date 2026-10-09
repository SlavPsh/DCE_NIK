# in vivo (meas_p3_dce, slices 18/19/21, k100 = every view in training (no held-out spokes; the val / test kNMSE columns are TRAIN-set numbers here); same views for every method)

rulers: mf_* = NRMSE vs model-free NUFFT ROI curve on its 240-pt grid (affine = raw+affine fit; scale = baseline-subtracted single scale). physical bounds on aorta. *_kNMSE = complex k-space NMSE at held-out spokes (NIK only). CS rows are references, NOT truth; CS held-out blocked (magnitude-only files).

## slice 18
| metric | sub16 mean±SD (n) |  | GRASP-v2 all spokes |
|---|---|---|---|
| cortex_peak_ratio | 0.7853 ± 0 (1) |  | 0.8153 | nan |
| cortex_washout_ratio | 0.9207 ± 0 (1) |  | 1.054 | nan |
| medulla_peak_ratio | 0.8576 ± 0 (1) |  | 1.039 | nan |
| medulla_washout_ratio | 0.9018 ± 0 (1) |  | 1.053 | nan |
| aorta_peak_ratio | 0.7085 ± 0 (1) |  | 0.7927 | nan |
| aorta_washout_ratio | 0.7722 ± 0 (1) |  | 1.019 | nan |
| mf_aorta_affine | 0.1428 ± 0 (1) |  | 0.07915 | nan |
| mf_cortex_affine | 0.05845 ± 0 (1) |  | 0.04022 | nan |
| mf_medulla_affine | 0.05851 ± 0 (1) |  | 0.02661 | nan |
| mf_liver_affine | 0.03276 ± 0 (1) |  | 0.02198 | nan |
| mf_aorta_scale | 0.1896 ± 0 (1) |  | 0.1089 | nan |
| mf_cortex_scale | 0.08998 ± 0 (1) |  | 0.06304 | nan |
| mf_medulla_scale | 0.08662 ± 0 (1) |  | 0.04044 | nan |
| aorta_peak_ratio_vs_mf | 0.7085 ± 0 (1) |  | 0.7927 | nan |
| aorta_fwhm_s | 21.5 ± 0 (1) |  | 21.5 | nan |
| aorta_ttp_s | 64.73 ± 0 (1) |  | 64.73 | nan |
| aorta_neg_frac | 0.05417 ± 0 (1) |  | 0 | nan |
| aorta_rise_mono | 1 ± 0 (1) |  | 1 | nan |
| cortex_medulla_late_corr | 0.1974 ± 0 (1) |  | -0.01859 | nan |
| train_kNMSE | nan ± nan (0) |  | nan | nan |
| val_kNMSE | nan ± nan (0) |  | nan | nan |
| test_kNMSE | nan ± nan (0) |  | nan | nan |
| wall_s | 3956 ± 0 (1) |  | nan | nan |
| peak_gpu_mb | 1.248e+04 ± 0 (1) |  | nan | nan |
| params | nan ± nan (0) |  | nan | nan |

model-free aorta FWHM (s): 15.359859566998232

## slice 19
| metric | sub16 mean±SD (n) |  | GRASP-v2 all spokes |
|---|---|---|---|
| cortex_peak_ratio | 0.6942 ± 0 (1) |  | 0.7625 | nan |
| cortex_washout_ratio | 0.9355 ± 0 (1) |  | 1.055 | nan |
| medulla_peak_ratio | 0.866 ± 0 (1) |  | 1.025 | nan |
| medulla_washout_ratio | 0.9267 ± 0 (1) |  | 1.043 | nan |
| aorta_peak_ratio | 0.7409 ± 0 (1) |  | 0.7134 | nan |
| aorta_washout_ratio | 0.7991 ± 0 (1) |  | 1.027 | nan |
| mf_aorta_affine | 0.1218 ± 0 (1) |  | 0.1115 | nan |
| mf_cortex_affine | 0.05692 ± 0 (1) |  | 0.05242 | nan |
| mf_medulla_affine | 0.06485 ± 0 (1) |  | 0.04 | nan |
| mf_liver_affine | 0.0377 ± 0 (1) |  | 0.02378 | nan |
| mf_aorta_scale | 0.1569 ± 0 (1) |  | 0.1504 | nan |
| mf_cortex_scale | 0.08996 ± 0 (1) |  | 0.08204 | nan |
| mf_medulla_scale | 0.09489 ± 0 (1) |  | 0.05853 | nan |
| aorta_peak_ratio_vs_mf | 0.7409 ± 0 (1) |  | 0.7134 | nan |
| aorta_fwhm_s | 19.97 ± 0 (1) |  | 66.05 | nan |
| aorta_ttp_s | 64.73 ± 0 (1) |  | 64.73 | nan |
| aorta_neg_frac | 0.02917 ± 0 (1) |  | 0 | nan |
| aorta_rise_mono | 1 ± 0 (1) |  | 1 | nan |
| cortex_medulla_late_corr | 0.1258 ± 0 (1) |  | 0.04944 | nan |
| train_kNMSE | nan ± nan (0) |  | nan | nan |
| val_kNMSE | nan ± nan (0) |  | nan | nan |
| test_kNMSE | nan ± nan (0) |  | nan | nan |
| wall_s | 3963 ± 0 (1) |  | nan | nan |
| peak_gpu_mb | 1.248e+04 ± 0 (1) |  | nan | nan |
| params | nan ± nan (0) |  | nan | nan |

model-free aorta FWHM (s): 15.359859566998232

## slice 21
| metric | sub16 mean±SD (n) |  | GRASP-v2 all spokes |
|---|---|---|---|
| cortex_peak_ratio | 0.7476 ± 0 (1) |  | 0.7881 | nan |
| cortex_washout_ratio | 0.9437 ± 0 (1) |  | 1.053 | nan |
| medulla_peak_ratio | 0.9094 ± 0 (1) |  | 1.015 | nan |
| medulla_washout_ratio | 0.99 ± 0 (1) |  | 1.072 | nan |
| aorta_peak_ratio | 0.7468 ± 0 (1) |  | 0.7702 | nan |
| aorta_washout_ratio | 0.9846 ± 0 (1) |  | 1.012 | nan |
| mf_aorta_affine | 0.1507 ± 0 (1) |  | 0.08901 | nan |
| mf_cortex_affine | 0.06721 ± 0 (1) |  | 0.03944 | nan |
| mf_medulla_affine | 0.06822 ± 0 (1) |  | 0.02517 | nan |
| mf_liver_affine | 0.03587 ± 0 (1) |  | 0.02321 | nan |
| mf_aorta_scale | 0.1948 ± 0 (1) |  | 0.1242 | nan |
| mf_cortex_scale | 0.1014 ± 0 (1) |  | 0.06079 | nan |
| mf_medulla_scale | 0.1017 ± 0 (1) |  | 0.03846 | nan |
| aorta_peak_ratio_vs_mf | 0.7468 ± 0 (1) |  | 0.7702 | nan |
| aorta_fwhm_s | 27.65 ± 0 (1) |  | 35.33 | nan |
| aorta_ttp_s | 64.73 ± 0 (1) |  | 64.73 | nan |
| aorta_neg_frac | 0 ± 0 (1) |  | 0 | nan |
| aorta_rise_mono | 0.875 ± 0 (1) |  | 1 | nan |
| cortex_medulla_late_corr | 0.1729 ± 0 (1) |  | -0.1108 | nan |
| train_kNMSE | nan ± nan (0) |  | nan | nan |
| val_kNMSE | nan ± nan (0) |  | nan | nan |
| test_kNMSE | nan ± nan (0) |  | nan | nan |
| wall_s | 3971 ± 0 (1) |  | nan | nan |
| peak_gpu_mb | 1.248e+04 ± 0 (1) |  | nan | nan |
| params | nan ± nan (0) |  | nan | nan |

model-free aorta FWHM (s): 16.89584552369808

## reference peak correction

the 31-spoke model-free reference (6.8 s window) clips the first-pass peak; factor = peak at an 11-spoke window / peak at 31 spokes (`mf_peak_check.py`, per-spoke normalized, approved rois). corrected ratio = ratio vs the 31-spoke reference / factor. state this with every peak comparison.

| slice | roi | clip factor | sub16 corrected peak ratio | GRASP-v2 corrected | GRASP-Pro corrected |
|---|---|---|---|---|---|
| 18 | cortex | 1.12 | 0.70 (raw 0.79) | 0.73 (raw 0.82) | - |
| 18 | medulla | 1.07 | 0.80 (raw 0.86) | 0.97 (raw 1.04) | - |
| 18 | aorta | 1.09 | 0.65 (raw 0.71) | 0.73 (raw 0.79) | - |
| 19 | cortex | 1.09 | 0.64 (raw 0.69) | 0.70 (raw 0.76) | - |
| 19 | medulla | 1.06 | 0.82 (raw 0.87) | 0.96 (raw 1.02) | - |
| 19 | aorta | 1.09 | 0.68 (raw 0.74) | 0.65 (raw 0.71) | - |
| 21 | cortex | 1.07 | 0.70 (raw 0.75) | 0.74 (raw 0.79) | - |
| 21 | medulla | 1.04 | 0.88 (raw 0.91) | 0.98 (raw 1.02) | - |
| 21 | aorta | 1.10 | 0.68 (raw 0.75) | 0.70 (raw 0.77) | - |
