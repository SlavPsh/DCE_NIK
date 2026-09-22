# in vivo (meas_p3_dce, slices 18/19/21, k80 = 1368/1708 views (v%10<8), VAL v%10==8 for early stop, TEST v%10==9 untouched; same views for every method)

rulers: mf_* = NRMSE vs model-free NUFFT ROI curve on its 240-pt grid (affine = raw+affine fit; scale = baseline-subtracted single scale). physical bounds on aorta. *_kNMSE = complex k-space NMSE at held-out spokes (NIK only). CS rows are references, NOT truth; CS held-out blocked (magnitude-only files).

## slice 18
| metric | tofts8 mean±SD (n) |  | GRASP-v2 k80 | GRASP-Pro f80match |
|---|---|---|---|
| cortex_peak_ratio | 0.87 ± 0.0056 (3) |  | 0.7884 | 0.9924 |
| cortex_washout_ratio | 0.9675 ± 0.0028 (3) |  | 1.046 | 1.012 |
| medulla_peak_ratio | 0.9276 ± 0.0056 (3) |  | 1.025 | 0.958 |
| medulla_washout_ratio | 0.9316 ± 0.0088 (3) |  | 1.046 | 0.933 |
| aorta_peak_ratio | 1.041 ± 0.0085 (3) |  | 0.7434 | 0.6819 |
| aorta_washout_ratio | 1.068 ± 0.04 (3) |  | 1.008 | 0.8691 |
| mf_aorta_affine | 0.04417 ± 0.0038 (3) |  | 0.09051 | 0.2972 |
| mf_cortex_affine | 0.03895 ± 0.00072 (3) |  | 0.04518 | 0.05357 |
| mf_medulla_affine | 0.02608 ± 0.0016 (3) |  | 0.0288 | 0.05778 |
| mf_liver_affine | 0.02151 ± 7.6e-05 (3) |  | 0.02232 | 0.03045 |
| mf_aorta_scale | 0.05804 ± 0.005 (3) |  | 0.1248 | 0.3905 |
| mf_cortex_scale | 0.06005 ± 0.0011 (3) |  | 0.07049 | 0.08244 |
| mf_medulla_scale | 0.03861 ± 0.0023 (3) |  | 0.04372 | 0.08555 |
| aorta_peak_ratio_vs_mf | 1.041 ± 0.0085 (3) |  | 0.7434 | 0.6819 |
| aorta_fwhm_s | 15.36 ± 0 (3) |  | 47.62 | 18.43 |
| aorta_ttp_s | 64.22 ± 0.72 (3) |  | 63.19 | 69.34 |
| aorta_neg_frac | 0 ± 0 (3) |  | 0 | 0 |
| aorta_rise_mono | 1 ± 0 (3) |  | 1 | 0.8605 |
| cortex_medulla_late_corr | 0.04203 ± 0.048 (3) |  | -0.01639 | 0.4411 |
| train_kNMSE | 0.2492 ± 0.00069 (3) |  | nan | nan |
| val_kNMSE | 0.2734 ± 0.00088 (3) |  | nan | nan |
| test_kNMSE | 0.2776 ± 0.00077 (3) |  | nan | nan |
| wall_s | 3859 ± 19 (3) |  | nan | nan |
| peak_gpu_mb | 1.246e+04 ± 0 (3) |  | nan | nan |
| params | 5.642e+06 ± 0 (3) |  | nan | nan |

model-free aorta FWHM (s): 15.359859566998232

TEST-spoke k-space NMSE per |k| annulus:
| annulus | tofts8 |
|---|---|
| 0.00-0.06 | 2.488e-01 |
| 0.06-0.12 | 5.471e-01 |
| 0.12-0.19 | 7.217e-01 |
| 0.19-0.25 | 8.514e-01 |
| 0.25-0.31 | 8.758e-01 |
| 0.31-0.38 | 7.841e-01 |
| 0.38-0.44 | 6.002e-01 |
| 0.44-0.50 | 6.902e-01 |
| 0.50-0.56 | 7.519e-01 |
| 0.56-0.62 | 6.186e-01 |
| 0.62-0.69 | 6.789e-01 |
| 0.69-0.75 | 7.361e-01 |
| 0.75-0.81 | 7.073e-01 |
| 0.81-0.88 | 7.875e-01 |
| 0.88-0.94 | 7.817e-01 |
| 0.94-1.00 | 8.522e-01 |

## slice 19
| metric | tofts8 mean±SD (n) |  | GRASP-v2 k80 | GRASP-Pro f80match |
|---|---|---|---|
| cortex_peak_ratio | 0.8718 ± 0.013 (3) |  | 0.7471 | 0.853 |
| cortex_washout_ratio | 0.9602 ± 0.003 (3) |  | 1.046 | 1.056 |
| medulla_peak_ratio | 0.905 ± 0.013 (3) |  | 1.015 | 1.062 |
| medulla_washout_ratio | 0.9006 ± 0.0058 (3) |  | 1.038 | 0.9018 |
| aorta_peak_ratio | 1.088 ± 0.018 (3) |  | 0.6472 | 0.3791 |
| aorta_washout_ratio | 1.109 ± 0.052 (3) |  | 1.003 | 1.042 |
| mf_aorta_affine | 0.04253 ± 0.0032 (3) |  | 0.1323 | 0.3489 |
| mf_cortex_affine | 0.03702 ± 0.0017 (3) |  | 0.0576 | 0.06224 |
| mf_medulla_affine | 0.03091 ± 0.00031 (3) |  | 0.0423 | 0.07265 |
| mf_liver_affine | 0.0236 ± 0.00018 (3) |  | 0.02405 | 0.03211 |
| mf_aorta_scale | 0.05523 ± 0.0059 (3) |  | 0.1761 | 0.4434 |
| mf_cortex_scale | 0.05625 ± 0.0026 (3) |  | 0.08963 | 0.09467 |
| mf_medulla_scale | 0.04493 ± 0.00043 (3) |  | 0.0619 | 0.1055 |
| aorta_peak_ratio_vs_mf | 1.088 ± 0.018 (3) |  | 0.6472 | 0.3791 |
| aorta_fwhm_s | 15.36 ± 0 (3) |  | 92.16 | 56.83 |
| aorta_ttp_s | 63.71 ± 0.72 (3) |  | 64.73 | 69.34 |
| aorta_neg_frac | 0 ± 0 (3) |  | 0 | 0.004167 |
| aorta_rise_mono | 1 ± 0 (3) |  | 1 | 0.8837 |
| cortex_medulla_late_corr | 0.3375 ± 0.087 (3) |  | 0.05177 | 0.9518 |
| train_kNMSE | 0.2534 ± 0.003 (3) |  | nan | nan |
| val_kNMSE | 0.273 ± 0.00065 (3) |  | nan | nan |
| test_kNMSE | 0.2777 ± 0.00019 (3) |  | nan | nan |
| wall_s | 3845 ± 27 (3) |  | nan | nan |
| peak_gpu_mb | 1.246e+04 ± 0 (3) |  | nan | nan |
| params | 5.642e+06 ± 0 (3) |  | nan | nan |

model-free aorta FWHM (s): 15.359859566998232

TEST-spoke k-space NMSE per |k| annulus:
| annulus | tofts8 |
|---|---|
| 0.00-0.06 | 2.434e-01 |
| 0.06-0.12 | 5.065e-01 |
| 0.12-0.19 | 6.438e-01 |
| 0.19-0.25 | 9.078e-01 |
| 0.25-0.31 | 9.052e-01 |
| 0.31-0.38 | 5.735e-01 |
| 0.38-0.44 | 3.599e-01 |
| 0.44-0.50 | 6.713e-01 |
| 0.50-0.56 | 6.955e-01 |
| 0.56-0.62 | 5.003e-01 |
| 0.62-0.69 | 6.386e-01 |
| 0.69-0.75 | 6.494e-01 |
| 0.75-0.81 | 6.105e-01 |
| 0.81-0.88 | 6.253e-01 |
| 0.88-0.94 | 6.191e-01 |
| 0.94-1.00 | 5.998e-01 |

## slice 21
| metric | tofts8 mean±SD (n) |  | GRASP-v2 k80 | GRASP-Pro f80match |
|---|---|---|---|
| cortex_peak_ratio | 0.8406 ± 0.009 (3) |  | 0.7667 | 0.638 |
| cortex_washout_ratio | 0.966 ± 0.0024 (3) |  | 1.048 | 0.9215 |
| medulla_peak_ratio | 0.9117 ± 0.0067 (3) |  | 1.008 | 0.9193 |
| medulla_washout_ratio | 0.965 ± 0.00072 (3) |  | 1.069 | 0.897 |
| aorta_peak_ratio | 1.044 ± 0.03 (3) |  | 0.7097 | 0.2285 |
| aorta_washout_ratio | 1.119 ± 0.041 (3) |  | 0.9795 | 0.7434 |
| mf_aorta_affine | 0.04781 ± 0.001 (3) |  | 0.1013 | 0.384 |
| mf_cortex_affine | 0.03488 ± 0.0012 (3) |  | 0.04423 | 0.1056 |
| mf_medulla_affine | 0.03121 ± 0.00061 (3) |  | 0.02758 | 0.07839 |
| mf_liver_affine | 0.0236 ± 0.0003 (3) |  | 0.02368 | 0.03818 |
| mf_aorta_scale | 0.06579 ± 0.0031 (3) |  | 0.1419 | 0.5084 |
| mf_cortex_scale | 0.0528 ± 0.0018 (3) |  | 0.06828 | 0.1632 |
| mf_medulla_scale | 0.0458 ± 0.0009 (3) |  | 0.04174 | 0.1151 |
| aorta_peak_ratio_vs_mf | 1.044 ± 0.03 (3) |  | 0.7097 | 0.2285 |
| aorta_fwhm_s | 16.38 ± 0.72 (3) |  | 52.22 | 98.3 |
| aorta_ttp_s | 64.22 ± 0.72 (3) |  | 64.73 | 207.6 |
| aorta_neg_frac | 0 ± 0 (3) |  | 0 | 0.02083 |
| aorta_rise_mono | 1 ± 0 (3) |  | 1 | 0.6466 |
| cortex_medulla_late_corr | 0.07944 ± 0.096 (3) |  | -0.1166 | 0.9698 |
| train_kNMSE | 0.2265 ± 0.00099 (3) |  | nan | nan |
| val_kNMSE | 0.2482 ± 0.0011 (3) |  | nan | nan |
| test_kNMSE | 0.252 ± 0.001 (3) |  | nan | nan |
| wall_s | 3822 ± 3.9 (3) |  | nan | nan |
| peak_gpu_mb | 1.246e+04 ± 0 (3) |  | nan | nan |
| params | 5.642e+06 ± 0 (3) |  | nan | nan |

model-free aorta FWHM (s): 16.89584552369808

TEST-spoke k-space NMSE per |k| annulus:
| annulus | tofts8 |
|---|---|
| 0.00-0.06 | 2.318e-01 |
| 0.06-0.12 | 4.621e-01 |
| 0.12-0.19 | 5.594e-01 |
| 0.19-0.25 | 7.707e-01 |
| 0.25-0.31 | 8.685e-01 |
| 0.31-0.38 | 9.252e-01 |
| 0.38-0.44 | 9.068e-01 |
| 0.44-0.50 | 1.059e+00 |
| 0.50-0.56 | 1.167e+00 |
| 0.56-0.62 | 1.184e+00 |
| 0.62-0.69 | 1.254e+00 |
| 0.69-0.75 | 1.299e+00 |
| 0.75-0.81 | 1.299e+00 |
| 0.81-0.88 | 1.320e+00 |
| 0.88-0.94 | 1.323e+00 |
| 0.94-1.00 | 1.355e+00 |

## reference peak correction

the 31-spoke model-free reference (6.8 s window) clips the first-pass peak; factor = peak at an 11-spoke window / peak at 31 spokes (`mf_peak_check.py`, per-spoke normalized, approved rois). corrected ratio = ratio vs the 31-spoke reference / factor. state this with every peak comparison.

| slice | roi | clip factor | tofts8 corrected peak ratio | GRASP-v2 corrected | GRASP-Pro corrected |
|---|---|---|---|---|---|
| 18 | cortex | 1.12 | 0.77 (raw 0.87) | 0.70 (raw 0.79) | 0.88 (raw 0.99) |
| 18 | medulla | 1.07 | 0.87 (raw 0.93) | 0.96 (raw 1.02) | 0.90 (raw 0.96) |
| 18 | aorta | 1.09 | 0.95 (raw 1.04) | 0.68 (raw 0.74) | 0.62 (raw 0.68) |
| 19 | cortex | 1.09 | 0.80 (raw 0.87) | 0.68 (raw 0.75) | 0.78 (raw 0.85) |
| 19 | medulla | 1.06 | 0.85 (raw 0.90) | 0.96 (raw 1.02) | 1.00 (raw 1.06) |
| 19 | aorta | 1.09 | 1.00 (raw 1.09) | 0.59 (raw 0.65) | 0.35 (raw 0.38) |
| 21 | cortex | 1.07 | 0.78 (raw 0.84) | 0.72 (raw 0.77) | 0.60 (raw 0.64) |
| 21 | medulla | 1.04 | 0.88 (raw 0.91) | 0.97 (raw 1.01) | 0.89 (raw 0.92) |
| 21 | aorta | 1.10 | 0.95 (raw 1.04) | 0.64 (raw 0.71) | 0.21 (raw 0.23) |
