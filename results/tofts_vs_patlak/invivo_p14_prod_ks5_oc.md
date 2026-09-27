# in vivo (meas_topqmri_p14, slices 21/24/27, k80 = 1368/1708 views (v%10<8), VAL v%10==8 for early stop, TEST v%10==9 untouched; same views for every method)

rulers: mf_* = NRMSE vs model-free NUFFT ROI curve on its 240-pt grid (affine = raw+affine fit; scale = baseline-subtracted single scale). physical bounds on aorta. *_kNMSE = complex k-space NMSE at held-out spokes (NIK only). CS rows are references, NOT truth; CS held-out blocked (magnitude-only files).

## slice 21
| metric | tofts8 mean±SD (n) |  | GRASP-v2 k80 | GRASP-Pro f80match |
|---|---|---|---|
| liver_peak_ratio | 1 ± 0 (1) |  | 1.005 | 0.7193 |
| liver_washout_ratio | 1.066 ± 0 (1) |  | 1.073 | 0.8801 |
| spleen_peak_ratio | 0.9167 ± 0 (1) |  | 1.069 | 1.051 |
| spleen_washout_ratio | 0.9495 ± 0 (1) |  | 1.128 | 1.067 |
| aorta_peak_ratio | 0.8584 ± 0 (1) |  | 0.8454 | 0.792 |
| aorta_washout_ratio | 1.084 ± 0 (1) |  | 1.125 | 0.9252 |
| mf_aorta_affine | 0.05646 ± 0 (1) |  | 0.06276 | 0.1551 |
| mf_liver_affine | 0.03986 ± 0 (1) |  | 0.02897 | 0.06528 |
| mf_spleen_affine | 0.03048 ± 0 (1) |  | 0.03147 | 0.05841 |
| mf_static_affine | 0.01918 ± 0 (1) |  | 0.01614 | 0.03601 |
| mf_aorta_scale | 0.08689 ± 0 (1) |  | 0.1004 | 0.246 |
| mf_liver_scale | 0.07966 ± 0 (1) |  | 0.05826 | 0.1315 |
| mf_spleen_scale | 0.05248 ± 0 (1) |  | 0.05689 | 0.1023 |
| aorta_peak_ratio_vs_mf | 0.8584 ± 0 (1) |  | 0.8454 | 0.792 |
| aorta_fwhm_s | 341.9 ± 0 (1) |  | 345.7 | 234.3 |
| aorta_ttp_s | 54.63 ± 0 (1) |  | 53.36 | 62.23 |
| aorta_neg_frac | 0 ± 0 (1) |  | 0 | 0.006579 |
| aorta_rise_mono | 1 ± 0 (1) |  | 1 | 0.8511 |
| cortex_medulla_late_corr | 0.9775 ± 0 (1) |  | 0.9812 | 0.3957 |
| train_kNMSE | 0.2504 ± 0 (1) |  | nan | nan |
| val_kNMSE | 0.2695 ± 0 (1) |  | nan | nan |
| test_kNMSE | 0.2733 ± 0 (1) |  | nan | nan |
| wall_s | 5522 ± 0 (1) |  | nan | nan |
| peak_gpu_mb | 1.283e+04 ± 0 (1) |  | nan | nan |
| params | 5.642e+06 ± 0 (1) |  | nan | nan |

model-free aorta FWHM (s): 164.61038961038963

TEST-spoke k-space NMSE per |k| annulus:
| annulus | tofts8 |
|---|---|
| 0.00-0.06 | 2.537e-01 |
| 0.06-0.12 | 5.552e-01 |
| 0.12-0.19 | 7.762e-01 |
| 0.19-0.25 | 8.210e-01 |
| 0.25-0.31 | 7.852e-01 |
| 0.31-0.38 | 7.859e-01 |
| 0.38-0.44 | 8.285e-01 |
| 0.44-0.50 | 8.478e-01 |
| 0.50-0.56 | 8.406e-01 |
| 0.56-0.62 | 8.599e-01 |
| 0.62-0.69 | 9.310e-01 |
| 0.69-0.75 | 9.324e-01 |
| 0.75-0.81 | 9.693e-01 |
| 0.81-0.88 | 9.409e-01 |
| 0.88-0.94 | 9.942e-01 |
| 0.94-1.00 | 1.020e+00 |

## slice 24
| metric | tofts8 mean±SD (n) |  | GRASP-v2 k80 | GRASP-Pro f80match |
|---|---|---|---|
| liver_peak_ratio | 0.969 ± 0 (1) |  | 0.9963 | 0.6594 |
| liver_washout_ratio | 1.02 ± 0 (1) |  | 1.071 | 0.7786 |
| spleen_peak_ratio | 0.8565 ± 0 (1) |  | 1.006 | 0.5909 |
| spleen_washout_ratio | 0.9038 ± 0 (1) |  | 1.082 | 0.6307 |
| aorta_peak_ratio | 0.8566 ± 0 (1) |  | 0.8087 | 0.4435 |
| aorta_washout_ratio | 1.025 ± 0 (1) |  | 1.093 | 0.4443 |
| mf_aorta_affine | 0.05633 ± 0 (1) |  | 0.08018 | 0.1783 |
| mf_liver_affine | 0.02776 ± 0 (1) |  | 0.02434 | 0.06781 |
| mf_spleen_affine | 0.0288 ± 0 (1) |  | 0.03766 | 0.0698 |
| mf_static_affine | 0.02393 ± 0 (1) |  | 0.01905 | 0.05442 |
| mf_aorta_scale | 0.08044 ± 0 (1) |  | 0.1182 | 0.2871 |
| mf_liver_scale | 0.057 ± 0 (1) |  | 0.05001 | 0.1391 |
| mf_spleen_scale | 0.04949 ± 0 (1) |  | 0.06732 | 0.1199 |
| aorta_peak_ratio_vs_mf | 0.8566 ± 0 (1) |  | 0.8087 | 0.4435 |
| aorta_fwhm_s | 339.4 ± 0 (1) |  | 345.7 | 70.91 |
| aorta_ttp_s | 48.3 ± 0 (1) |  | 69.82 | 59.69 |
| aorta_neg_frac | 0 ± 0 (1) |  | 0 | 0.009868 |
| aorta_rise_mono | 1 ± 0 (1) |  | 1 | 0.8 |
| cortex_medulla_late_corr | 0.9937 ± 0 (1) |  | 0.9912 | -0.02402 |
| train_kNMSE | 0.2381 ± 0 (1) |  | nan | nan |
| val_kNMSE | 0.2547 ± 0 (1) |  | nan | nan |
| test_kNMSE | 0.2562 ± 0 (1) |  | nan | nan |
| wall_s | 5522 ± 0 (1) |  | nan | nan |
| peak_gpu_mb | 1.283e+04 ± 0 (1) |  | nan | nan |
| params | 5.642e+06 ± 0 (1) |  | nan | nan |

model-free aorta FWHM (s): 162.07792207792212

TEST-spoke k-space NMSE per |k| annulus:
| annulus | tofts8 |
|---|---|
| 0.00-0.06 | 2.388e-01 |
| 0.06-0.12 | 5.504e-01 |
| 0.12-0.19 | 7.392e-01 |
| 0.19-0.25 | 7.260e-01 |
| 0.25-0.31 | 7.358e-01 |
| 0.31-0.38 | 6.686e-01 |
| 0.38-0.44 | 7.984e-01 |
| 0.44-0.50 | 6.987e-01 |
| 0.50-0.56 | 6.969e-01 |
| 0.56-0.62 | 7.249e-01 |
| 0.62-0.69 | 6.634e-01 |
| 0.69-0.75 | 6.979e-01 |
| 0.75-0.81 | 7.200e-01 |
| 0.81-0.88 | 6.943e-01 |
| 0.88-0.94 | 6.886e-01 |
| 0.94-1.00 | 7.024e-01 |

## slice 27
| metric | tofts8 mean±SD (n) |  | GRASP-v2 k80 | GRASP-Pro f80match |
|---|---|---|---|
| liver_peak_ratio | 0.9962 ± 0 (1) |  | 0.9778 | 0.7799 |
| liver_washout_ratio | 1.054 ± 0 (1) |  | 1.047 | 0.8155 |
| spleen_peak_ratio | 0.8459 ± 0 (1) |  | 1.005 | 0.5465 |
| spleen_washout_ratio | 0.8965 ± 0 (1) |  | 1.08 | 0.7082 |
| aorta_peak_ratio | 0.8377 ± 0 (1) |  | 0.8151 | 0.4317 |
| aorta_washout_ratio | 0.9811 ± 0 (1) |  | 1.06 | 0.475 |
| mf_aorta_affine | 0.04889 ± 0 (1) |  | 0.0594 | 0.1828 |
| mf_liver_affine | 0.02401 ± 0 (1) |  | 0.02041 | 0.05849 |
| mf_spleen_affine | 0.0291 ± 0 (1) |  | 0.03264 | 0.09962 |
| mf_static_affine | 0.0213 ± 0 (1) |  | 0.02001 | 0.04029 |
| mf_aorta_scale | 0.06771 ± 0 (1) |  | 0.08333 | 0.2832 |
| mf_liver_scale | 0.05093 ± 0 (1) |  | 0.04324 | 0.1236 |
| mf_spleen_scale | 0.05147 ± 0 (1) |  | 0.06003 | 0.1746 |
| aorta_peak_ratio_vs_mf | 0.8377 ± 0 (1) |  | 0.8151 | 0.4317 |
| aorta_fwhm_s | 341.9 ± 0 (1) |  | 345.7 | 46.85 |
| aorta_ttp_s | 48.3 ± 0 (1) |  | 68.56 | 59.69 |
| aorta_neg_frac | 0 ± 0 (1) |  | 0 | 0.01316 |
| aorta_rise_mono | 1 ± 0 (1) |  | 1 | 0.8 |
| cortex_medulla_late_corr | 0.9908 ± 0 (1) |  | 0.9975 | 0.2277 |
| train_kNMSE | 0.215 ± 0 (1) |  | nan | nan |
| val_kNMSE | 0.2237 ± 0 (1) |  | nan | nan |
| test_kNMSE | 0.2259 ± 0 (1) |  | nan | nan |
| wall_s | 5520 ± 0 (1) |  | nan | nan |
| peak_gpu_mb | 1.283e+04 ± 0 (1) |  | nan | nan |
| params | 5.642e+06 ± 0 (1) |  | nan | nan |

model-free aorta FWHM (s): 196.26623376623377

TEST-spoke k-space NMSE per |k| annulus:
| annulus | tofts8 |
|---|---|
| 0.00-0.06 | 2.174e-01 |
| 0.06-0.12 | 5.017e-01 |
| 0.12-0.19 | 5.789e-01 |
| 0.19-0.25 | 6.659e-01 |
| 0.25-0.31 | 7.094e-01 |
| 0.31-0.38 | 7.610e-01 |
| 0.38-0.44 | 8.306e-01 |
| 0.44-0.50 | 8.659e-01 |
| 0.50-0.56 | 9.030e-01 |
| 0.56-0.62 | 9.519e-01 |
| 0.62-0.69 | 9.829e-01 |
| 0.69-0.75 | 9.925e-01 |
| 0.75-0.81 | 1.023e+00 |
| 0.81-0.88 | 1.036e+00 |
| 0.88-0.94 | 1.067e+00 |
| 0.94-1.00 | 1.091e+00 |

## reference peak correction

the 31-spoke model-free reference (6.8 s window) clips the first-pass peak; factor = peak at an 11-spoke window / peak at 31 spokes (`mf_peak_check.py`, per-spoke normalized, approved rois). corrected ratio = ratio vs the 31-spoke reference / factor. state this with every peak comparison.

| slice | roi | clip factor | tofts8 corrected peak ratio | GRASP-v2 corrected | GRASP-Pro corrected |
|---|---|---|---|---|---|
| 21 | liver | 1.06 | 0.94 (raw 1.00) | 0.94 (raw 1.01) | 0.68 (raw 0.72) |
| 21 | spleen | 1.07 | 0.86 (raw 0.92) | 1.00 (raw 1.07) | 0.98 (raw 1.05) |
| 21 | aorta | 1.09 | 0.79 (raw 0.86) | 0.77 (raw 0.85) | 0.73 (raw 0.79) |
| 24 | liver | 1.09 | 0.89 (raw 0.97) | 0.91 (raw 1.00) | 0.61 (raw 0.66) |
| 24 | spleen | 1.08 | 0.79 (raw 0.86) | 0.93 (raw 1.01) | 0.54 (raw 0.59) |
| 24 | aorta | 1.02 | 0.84 (raw 0.86) | 0.79 (raw 0.81) | 0.43 (raw 0.44) |
| 27 | liver | 1.13 | 0.88 (raw 1.00) | 0.87 (raw 0.98) | 0.69 (raw 0.78) |
| 27 | spleen | 1.09 | 0.78 (raw 0.85) | 0.93 (raw 1.01) | 0.50 (raw 0.55) |
| 27 | aorta | 1.07 | 0.78 (raw 0.84) | 0.76 (raw 0.82) | 0.40 (raw 0.43) |
