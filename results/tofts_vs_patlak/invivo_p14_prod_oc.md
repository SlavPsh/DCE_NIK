# in vivo (meas_p3_dce, slices 21/24/27, k80 = 1368/1708 views (v%10<8), VAL v%10==8 for early stop, TEST v%10==9 untouched; same views for every method)

rulers: mf_* = NRMSE vs model-free NUFFT ROI curve on its 240-pt grid (affine = raw+affine fit; scale = baseline-subtracted single scale). physical bounds on aorta. *_kNMSE = complex k-space NMSE at held-out spokes (NIK only). CS rows are references, NOT truth; CS held-out blocked (magnitude-only files).

## slice 21
| metric | tofts8 mean±SD (n) |  | GRASP-v2 k80 | GRASP-Pro f80match |
|---|---|---|---|
| liver_peak_ratio | 1.004 ± 0 (1) |  | 1.005 | 0.7193 |
| liver_washout_ratio | 1.071 ± 0 (1) |  | 1.073 | 0.8801 |
| spleen_peak_ratio | 0.9228 ± 0 (1) |  | 1.069 | 1.051 |
| spleen_washout_ratio | 0.9513 ± 0 (1) |  | 1.128 | 1.067 |
| aorta_peak_ratio | 0.9248 ± 0 (1) |  | 0.8454 | 0.792 |
| aorta_washout_ratio | 1.116 ± 0 (1) |  | 1.125 | 0.9252 |
| mf_aorta_affine | 0.04922 ± 0 (1) |  | 0.06276 | 0.1551 |
| mf_liver_affine | 0.0397 ± 0 (1) |  | 0.02897 | 0.06528 |
| mf_spleen_affine | 0.03 ± 0 (1) |  | 0.03147 | 0.05841 |
| mf_static_affine | 0.01935 ± 0 (1) |  | 0.01614 | 0.03601 |
| mf_aorta_scale | 0.07611 ± 0 (1) |  | 0.1004 | 0.246 |
| mf_liver_scale | 0.07929 ± 0 (1) |  | 0.05826 | 0.1315 |
| mf_spleen_scale | 0.05136 ± 0 (1) |  | 0.05689 | 0.1023 |
| aorta_peak_ratio_vs_mf | 0.9248 ± 0 (1) |  | 0.8454 | 0.792 |
| aorta_fwhm_s | 286.2 ± 0 (1) |  | 345.7 | 234.3 |
| aorta_ttp_s | 53.36 ± 0 (1) |  | 53.36 | 62.23 |
| aorta_neg_frac | 0 ± 0 (1) |  | 0 | 0.006579 |
| aorta_rise_mono | 1 ± 0 (1) |  | 1 | 0.8511 |
| cortex_medulla_late_corr | 0.9755 ± 0 (1) |  | 0.9812 | 0.3957 |
| train_kNMSE | 0.2541 ± 0 (1) |  | nan | nan |
| val_kNMSE | 0.2696 ± 0 (1) |  | nan | nan |
| test_kNMSE | 0.2731 ± 0 (1) |  | nan | nan |
| wall_s | 924.5 ± 0 (1) |  | nan | nan |
| peak_gpu_mb | 5165 ± 0 (1) |  | nan | nan |
| params | 5.642e+06 ± 0 (1) |  | nan | nan |

model-free aorta FWHM (s): 164.61038961038963

TEST-spoke k-space NMSE per |k| annulus:
| annulus | tofts8 |
|---|---|
| 0.00-0.06 | 2.551e-01 |
| 0.06-0.12 | 5.437e-01 |
| 0.12-0.19 | 7.169e-01 |
| 0.19-0.25 | 7.504e-01 |
| 0.25-0.31 | 7.356e-01 |
| 0.31-0.38 | 7.295e-01 |
| 0.38-0.44 | 7.696e-01 |
| 0.44-0.50 | 7.995e-01 |
| 0.50-0.56 | 8.033e-01 |
| 0.56-0.62 | 8.262e-01 |
| 0.62-0.69 | 8.729e-01 |
| 0.69-0.75 | 8.628e-01 |
| 0.75-0.81 | 8.991e-01 |
| 0.81-0.88 | 8.812e-01 |
| 0.88-0.94 | 9.359e-01 |
| 0.94-1.00 | 9.540e-01 |

## slice 24
| metric | tofts8 mean±SD (n) |  | GRASP-v2 k80 | GRASP-Pro f80match |
|---|---|---|---|
| liver_peak_ratio | 0.9501 ± 0 (1) |  | 0.9963 | 0.6594 |
| liver_washout_ratio | 1.012 ± 0 (1) |  | 1.071 | 0.7786 |
| spleen_peak_ratio | 0.8471 ± 0 (1) |  | 1.006 | 0.5909 |
| spleen_washout_ratio | 0.9097 ± 0 (1) |  | 1.082 | 0.6307 |
| aorta_peak_ratio | 0.9396 ± 0 (1) |  | 0.8087 | 0.4435 |
| aorta_washout_ratio | 1.077 ± 0 (1) |  | 1.093 | 0.4443 |
| mf_aorta_affine | 0.04824 ± 0 (1) |  | 0.08018 | 0.1783 |
| mf_liver_affine | 0.02787 ± 0 (1) |  | 0.02434 | 0.06781 |
| mf_spleen_affine | 0.02981 ± 0 (1) |  | 0.03766 | 0.0698 |
| mf_static_affine | 0.02469 ± 0 (1) |  | 0.01905 | 0.05442 |
| mf_aorta_scale | 0.06916 ± 0 (1) |  | 0.1182 | 0.2871 |
| mf_liver_scale | 0.0572 ± 0 (1) |  | 0.05001 | 0.1391 |
| mf_spleen_scale | 0.05131 ± 0 (1) |  | 0.06732 | 0.1199 |
| aorta_peak_ratio_vs_mf | 0.9396 ± 0 (1) |  | 0.8087 | 0.4435 |
| aorta_fwhm_s | 317.8 ± 0 (1) |  | 345.7 | 70.91 |
| aorta_ttp_s | 48.3 ± 0 (1) |  | 69.82 | 59.69 |
| aorta_neg_frac | 0 ± 0 (1) |  | 0 | 0.009868 |
| aorta_rise_mono | 1 ± 0 (1) |  | 1 | 0.8 |
| cortex_medulla_late_corr | 0.9942 ± 0 (1) |  | 0.9912 | -0.02402 |
| train_kNMSE | 0.2427 ± 0 (1) |  | nan | nan |
| val_kNMSE | 0.2519 ± 0 (1) |  | nan | nan |
| test_kNMSE | 0.2538 ± 0 (1) |  | nan | nan |
| wall_s | 925.2 ± 0 (1) |  | nan | nan |
| peak_gpu_mb | 5165 ± 0 (1) |  | nan | nan |
| params | 5.642e+06 ± 0 (1) |  | nan | nan |

model-free aorta FWHM (s): 162.07792207792212

TEST-spoke k-space NMSE per |k| annulus:
| annulus | tofts8 |
|---|---|
| 0.00-0.06 | 2.379e-01 |
| 0.06-0.12 | 5.302e-01 |
| 0.12-0.19 | 6.649e-01 |
| 0.19-0.25 | 6.540e-01 |
| 0.25-0.31 | 6.822e-01 |
| 0.31-0.38 | 6.351e-01 |
| 0.38-0.44 | 7.513e-01 |
| 0.44-0.50 | 6.513e-01 |
| 0.50-0.56 | 6.931e-01 |
| 0.56-0.62 | 7.359e-01 |
| 0.62-0.69 | 6.374e-01 |
| 0.69-0.75 | 6.962e-01 |
| 0.75-0.81 | 7.508e-01 |
| 0.81-0.88 | 7.399e-01 |
| 0.88-0.94 | 7.017e-01 |
| 0.94-1.00 | 7.209e-01 |

## slice 27
| metric | tofts8 mean±SD (n) |  | GRASP-v2 k80 | GRASP-Pro f80match |
|---|---|---|---|
| liver_peak_ratio | 1.003 ± 0 (1) |  | 0.9778 | 0.7799 |
| liver_washout_ratio | 1.057 ± 0 (1) |  | 1.047 | 0.8155 |
| spleen_peak_ratio | 0.8416 ± 0 (1) |  | 1.005 | 0.5465 |
| spleen_washout_ratio | 0.8941 ± 0 (1) |  | 1.08 | 0.7082 |
| aorta_peak_ratio | 0.9196 ± 0 (1) |  | 0.8151 | 0.4317 |
| aorta_washout_ratio | 1.065 ± 0 (1) |  | 1.06 | 0.475 |
| mf_aorta_affine | 0.04753 ± 0 (1) |  | 0.0594 | 0.1828 |
| mf_liver_affine | 0.02384 ± 0 (1) |  | 0.02041 | 0.05849 |
| mf_spleen_affine | 0.02823 ± 0 (1) |  | 0.03264 | 0.09962 |
| mf_static_affine | 0.02133 ± 0 (1) |  | 0.02001 | 0.04029 |
| mf_aorta_scale | 0.06593 ± 0 (1) |  | 0.08333 | 0.2832 |
| mf_liver_scale | 0.05051 ± 0 (1) |  | 0.04324 | 0.1236 |
| mf_spleen_scale | 0.0499 ± 0 (1) |  | 0.06003 | 0.1746 |
| aorta_peak_ratio_vs_mf | 0.9196 ± 0 (1) |  | 0.8151 | 0.4317 |
| aorta_fwhm_s | 341.9 ± 0 (1) |  | 345.7 | 46.85 |
| aorta_ttp_s | 48.3 ± 0 (1) |  | 68.56 | 59.69 |
| aorta_neg_frac | 0 ± 0 (1) |  | 0 | 0.01316 |
| aorta_rise_mono | 1 ± 0 (1) |  | 1 | 0.8 |
| cortex_medulla_late_corr | 0.9871 ± 0 (1) |  | 0.9975 | 0.2277 |
| train_kNMSE | 0.2164 ± 0 (1) |  | nan | nan |
| val_kNMSE | 0.2236 ± 0 (1) |  | nan | nan |
| test_kNMSE | 0.2264 ± 0 (1) |  | nan | nan |
| wall_s | 924.6 ± 0 (1) |  | nan | nan |
| peak_gpu_mb | 5165 ± 0 (1) |  | nan | nan |
| params | 5.642e+06 ± 0 (1) |  | nan | nan |

model-free aorta FWHM (s): 196.26623376623377

TEST-spoke k-space NMSE per |k| annulus:
| annulus | tofts8 |
|---|---|
| 0.00-0.06 | 2.183e-01 |
| 0.06-0.12 | 5.008e-01 |
| 0.12-0.19 | 5.643e-01 |
| 0.19-0.25 | 6.278e-01 |
| 0.25-0.31 | 6.631e-01 |
| 0.31-0.38 | 7.262e-01 |
| 0.38-0.44 | 7.906e-01 |
| 0.44-0.50 | 8.192e-01 |
| 0.50-0.56 | 8.606e-01 |
| 0.56-0.62 | 9.084e-01 |
| 0.62-0.69 | 9.313e-01 |
| 0.69-0.75 | 9.459e-01 |
| 0.75-0.81 | 9.744e-01 |
| 0.81-0.88 | 9.806e-01 |
| 0.88-0.94 | 9.975e-01 |
| 0.94-1.00 | 1.017e+00 |

## reference peak correction

the 31-spoke model-free reference (6.8 s window) clips the first-pass peak; factor = peak at an 11-spoke window / peak at 31 spokes (`mf_peak_check.py`, per-spoke normalized, approved rois). corrected ratio = ratio vs the 31-spoke reference / factor. state this with every peak comparison.

| slice | roi | clip factor | tofts8 corrected peak ratio | GRASP-v2 corrected | GRASP-Pro corrected |
|---|---|---|---|---|---|
| 21 | liver | 1.06 | 0.94 (raw 1.00) | 0.94 (raw 1.01) | 0.68 (raw 0.72) |
| 21 | spleen | 1.07 | 0.86 (raw 0.92) | 1.00 (raw 1.07) | 0.98 (raw 1.05) |
| 21 | aorta | 1.09 | 0.85 (raw 0.92) | 0.77 (raw 0.85) | 0.73 (raw 0.79) |
| 24 | liver | 1.09 | 0.87 (raw 0.95) | 0.91 (raw 1.00) | 0.61 (raw 0.66) |
| 24 | spleen | 1.08 | 0.78 (raw 0.85) | 0.93 (raw 1.01) | 0.54 (raw 0.59) |
| 24 | aorta | 1.02 | 0.92 (raw 0.94) | 0.79 (raw 0.81) | 0.43 (raw 0.44) |
| 27 | liver | 1.13 | 0.89 (raw 1.00) | 0.87 (raw 0.98) | 0.69 (raw 0.78) |
| 27 | spleen | 1.09 | 0.77 (raw 0.84) | 0.93 (raw 1.01) | 0.50 (raw 0.55) |
| 27 | aorta | 1.07 | 0.86 (raw 0.92) | 0.76 (raw 0.82) | 0.40 (raw 0.43) |
