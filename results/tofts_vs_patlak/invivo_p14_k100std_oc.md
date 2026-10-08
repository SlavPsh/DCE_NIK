# in vivo (meas_topqmri_p14, slices 21/24/27, k100 = every view in training (no held-out spokes; the val / test kNMSE columns are TRAIN-set numbers here); same views for every method)

rulers: mf_* = NRMSE vs model-free NUFFT ROI curve on its 240-pt grid (affine = raw+affine fit; scale = baseline-subtracted single scale). physical bounds on aorta. *_kNMSE = complex k-space NMSE at held-out spokes (NIK only). CS rows are references, NOT truth; CS held-out blocked (magnitude-only files).

## slice 21
| metric | tofts8 mean±SD (n) |  | GRASP-v2 all spokes |
|---|---|---|---|
| liver_peak_ratio | 0.9815 ± 0 (1) |  | 1.023 | nan |
| liver_washout_ratio | 1.041 ± 0 (1) |  | 1.094 | nan |
| spleen_peak_ratio | 0.9322 ± 0 (1) |  | 1.073 | nan |
| spleen_washout_ratio | 0.9947 ± 0 (1) |  | 1.123 | nan |
| aorta_peak_ratio | 0.9376 ± 0 (1) |  | 0.8699 | nan |
| aorta_washout_ratio | 1.101 ± 0 (1) |  | 1.132 | nan |
| mf_aorta_affine | 0.04348 ± 0 (1) |  | 0.06055 | nan |
| mf_liver_affine | 0.03957 ± 0 (1) |  | 0.02761 | nan |
| mf_spleen_affine | 0.03199 ± 0 (1) |  | 0.02994 | nan |
| mf_static_affine | 0.02011 ± 0 (1) |  | 0.01571 | nan |
| mf_aorta_scale | 0.06786 ± 0 (1) |  | 0.09688 | nan |
| mf_liver_scale | 0.07892 ± 0 (1) |  | 0.05506 | nan |
| mf_spleen_scale | 0.05594 ± 0 (1) |  | 0.05423 | nan |
| aorta_peak_ratio_vs_mf | 0.9376 ± 0 (1) |  | 0.8699 | nan |
| aorta_fwhm_s | 287.4 ± 0 (1) |  | 345.7 | nan |
| aorta_ttp_s | 48.3 ± 0 (1) |  | 53.36 | nan |
| aorta_neg_frac | 0 ± 0 (1) |  | 0 | nan |
| aorta_rise_mono | 1 ± 0 (1) |  | 1 | nan |
| cortex_medulla_late_corr | 0.9743 ± 0 (1) |  | 0.9778 | nan |
| train_kNMSE | 0.2526 ± 0 (1) |  | nan | nan |
| val_kNMSE | 0.2527 ± 0 (1) |  | nan | nan |
| test_kNMSE | 0.2533 ± 0 (1) |  | nan | nan |
| wall_s | 5538 ± 0 (1) |  | nan | nan |
| peak_gpu_mb | 1.287e+04 ± 0 (1) |  | nan | nan |
| params | 5.642e+06 ± 0 (1) |  | nan | nan |

model-free aorta FWHM (s): 164.61038961038963

TEST-spoke k-space NMSE per |k| annulus:
| annulus | tofts8 |
|---|---|
| 0.00-0.06 | 2.409e-01 |
| 0.06-0.12 | 4.574e-01 |
| 0.12-0.19 | 5.697e-01 |
| 0.19-0.25 | 5.552e-01 |
| 0.25-0.31 | 5.216e-01 |
| 0.31-0.38 | 5.028e-01 |
| 0.38-0.44 | 5.228e-01 |
| 0.44-0.50 | 5.271e-01 |
| 0.50-0.56 | 5.203e-01 |
| 0.56-0.62 | 5.253e-01 |
| 0.62-0.69 | 5.579e-01 |
| 0.69-0.75 | 5.517e-01 |
| 0.75-0.81 | 5.636e-01 |
| 0.81-0.88 | 5.554e-01 |
| 0.88-0.94 | 5.753e-01 |
| 0.94-1.00 | 5.900e-01 |

## slice 24
| metric | tofts8 mean±SD (n) |  | GRASP-v2 all spokes |
|---|---|---|---|
| liver_peak_ratio | 0.9551 ± 0 (1) |  | 1.014 | nan |
| liver_washout_ratio | 1.014 ± 0 (1) |  | 1.089 | nan |
| spleen_peak_ratio | 0.869 ± 0 (1) |  | 1.013 | nan |
| spleen_washout_ratio | 0.951 ± 0 (1) |  | 1.088 | nan |
| aorta_peak_ratio | 0.886 ± 0 (1) |  | 0.8305 | nan |
| aorta_washout_ratio | 0.9913 ± 0 (1) |  | 1.111 | nan |
| mf_aorta_affine | 0.04689 ± 0 (1) |  | 0.07582 | nan |
| mf_liver_affine | 0.02764 ± 0 (1) |  | 0.0234 | nan |
| mf_spleen_affine | 0.03156 ± 0 (1) |  | 0.03559 | nan |
| mf_static_affine | 0.02451 ± 0 (1) |  | 0.01865 | nan |
| mf_aorta_scale | 0.06703 ± 0 (1) |  | 0.1119 | nan |
| mf_liver_scale | 0.05674 ± 0 (1) |  | 0.04834 | nan |
| mf_spleen_scale | 0.0549 ± 0 (1) |  | 0.06395 | nan |
| aorta_peak_ratio_vs_mf | 0.886 ± 0 (1) |  | 0.8305 | nan |
| aorta_fwhm_s | 317.8 ± 0 (1) |  | 345.7 | nan |
| aorta_ttp_s | 48.3 ± 0 (1) |  | 68.56 | nan |
| aorta_neg_frac | 0 ± 0 (1) |  | 0 | nan |
| aorta_rise_mono | 1 ± 0 (1) |  | 1 | nan |
| cortex_medulla_late_corr | 0.9932 ± 0 (1) |  | 0.9872 | nan |
| train_kNMSE | 0.2396 ± 0 (1) |  | nan | nan |
| val_kNMSE | 0.2401 ± 0 (1) |  | nan | nan |
| test_kNMSE | 0.24 ± 0 (1) |  | nan | nan |
| wall_s | 5524 ± 0 (1) |  | nan | nan |
| peak_gpu_mb | 1.287e+04 ± 0 (1) |  | nan | nan |
| params | 5.642e+06 ± 0 (1) |  | nan | nan |

model-free aorta FWHM (s): 162.07792207792212

TEST-spoke k-space NMSE per |k| annulus:
| annulus | tofts8 |
|---|---|
| 0.00-0.06 | 2.287e-01 |
| 0.06-0.12 | 4.781e-01 |
| 0.12-0.19 | 5.702e-01 |
| 0.19-0.25 | 5.501e-01 |
| 0.25-0.31 | 5.098e-01 |
| 0.31-0.38 | 4.154e-01 |
| 0.38-0.44 | 4.313e-01 |
| 0.44-0.50 | 3.670e-01 |
| 0.50-0.56 | 3.426e-01 |
| 0.56-0.62 | 3.247e-01 |
| 0.62-0.69 | 2.989e-01 |
| 0.69-0.75 | 3.039e-01 |
| 0.75-0.81 | 3.299e-01 |
| 0.81-0.88 | 3.286e-01 |
| 0.88-0.94 | 3.311e-01 |
| 0.94-1.00 | 3.599e-01 |

## slice 27
| metric | tofts8 mean±SD (n) |  | GRASP-v2 all spokes |
|---|---|---|---|
| liver_peak_ratio | 0.9768 ± 0 (1) |  | 1.001 | nan |
| liver_washout_ratio | 1.029 ± 0 (1) |  | 1.073 | nan |
| spleen_peak_ratio | 0.8517 ± 0 (1) |  | 1.007 | nan |
| spleen_washout_ratio | 0.9193 ± 0 (1) |  | 1.083 | nan |
| aorta_peak_ratio | 0.9238 ± 0 (1) |  | 0.8483 | nan |
| aorta_washout_ratio | 0.9275 ± 0 (1) |  | 1.084 | nan |
| mf_aorta_affine | 0.05116 ± 0 (1) |  | 0.05616 | nan |
| mf_liver_affine | 0.02309 ± 0 (1) |  | 0.02002 | nan |
| mf_spleen_affine | 0.02975 ± 0 (1) |  | 0.03153 | nan |
| mf_static_affine | 0.02143 ± 0 (1) |  | 0.01976 | nan |
| mf_aorta_scale | 0.07554 ± 0 (1) |  | 0.07989 | nan |
| mf_liver_scale | 0.04889 ± 0 (1) |  | 0.04287 | nan |
| mf_spleen_scale | 0.053 ± 0 (1) |  | 0.05792 | nan |
| aorta_peak_ratio_vs_mf | 0.9238 ± 0 (1) |  | 0.8483 | nan |
| aorta_fwhm_s | 233 ± 0 (1) |  | 345.7 | nan |
| aorta_ttp_s | 48.3 ± 0 (1) |  | 53.36 | nan |
| aorta_neg_frac | 0 ± 0 (1) |  | 0 | nan |
| aorta_rise_mono | 1 ± 0 (1) |  | 1 | nan |
| cortex_medulla_late_corr | 0.9887 ± 0 (1) |  | 0.9933 | nan |
| train_kNMSE | 0.2164 ± 0 (1) |  | nan | nan |
| val_kNMSE | 0.2155 ± 0 (1) |  | nan | nan |
| test_kNMSE | 0.2164 ± 0 (1) |  | nan | nan |
| wall_s | 5526 ± 0 (1) |  | nan | nan |
| peak_gpu_mb | 1.287e+04 ± 0 (1) |  | nan | nan |
| params | 5.642e+06 ± 0 (1) |  | nan | nan |

model-free aorta FWHM (s): 196.26623376623377

TEST-spoke k-space NMSE per |k| annulus:
| annulus | tofts8 |
|---|---|
| 0.00-0.06 | 2.101e-01 |
| 0.06-0.12 | 4.391e-01 |
| 0.12-0.19 | 4.878e-01 |
| 0.19-0.25 | 5.207e-01 |
| 0.25-0.31 | 5.255e-01 |
| 0.31-0.38 | 5.512e-01 |
| 0.38-0.44 | 5.778e-01 |
| 0.44-0.50 | 5.866e-01 |
| 0.50-0.56 | 5.963e-01 |
| 0.56-0.62 | 6.149e-01 |
| 0.62-0.69 | 6.141e-01 |
| 0.69-0.75 | 5.935e-01 |
| 0.75-0.81 | 6.090e-01 |
| 0.81-0.88 | 6.057e-01 |
| 0.88-0.94 | 6.156e-01 |
| 0.94-1.00 | 6.298e-01 |

## reference peak correction

the 31-spoke model-free reference (6.8 s window) clips the first-pass peak; factor = peak at an 11-spoke window / peak at 31 spokes (`mf_peak_check.py`, per-spoke normalized, approved rois). corrected ratio = ratio vs the 31-spoke reference / factor. state this with every peak comparison.

| slice | roi | clip factor | tofts8 corrected peak ratio | GRASP-v2 corrected | GRASP-Pro corrected |
|---|---|---|---|---|---|
| 21 | liver | 1.06 | 0.92 (raw 0.98) | 0.96 (raw 1.02) | - |
| 21 | spleen | 1.07 | 0.87 (raw 0.93) | 1.00 (raw 1.07) | - |
| 21 | aorta | 1.09 | 0.86 (raw 0.94) | 0.80 (raw 0.87) | - |
| 24 | liver | 1.09 | 0.88 (raw 0.96) | 0.93 (raw 1.01) | - |
| 24 | spleen | 1.08 | 0.80 (raw 0.87) | 0.93 (raw 1.01) | - |
| 24 | aorta | 1.02 | 0.87 (raw 0.89) | 0.81 (raw 0.83) | - |
| 27 | liver | 1.13 | 0.87 (raw 0.98) | 0.89 (raw 1.00) | - |
| 27 | spleen | 1.09 | 0.78 (raw 0.85) | 0.93 (raw 1.01) | - |
| 27 | aorta | 1.07 | 0.86 (raw 0.92) | 0.79 (raw 0.85) | - |
