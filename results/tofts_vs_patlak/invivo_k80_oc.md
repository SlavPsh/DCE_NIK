# in vivo (meas_p3_dce, slices 18/19/21, k80 = 1368/1708 views (v%10<8), VAL v%10==8 for early stop, TEST v%10==9 untouched; same views for every method)

rulers: mf_* = NRMSE vs model-free NUFFT ROI curve on its 240-pt grid (affine = raw+affine fit; scale = baseline-subtracted single scale). physical bounds on aorta. *_kNMSE = complex k-space NMSE at held-out spokes (NIK only). CS rows are references, NOT truth; CS held-out blocked (magnitude-only files).

## slice 18
| metric | tofts8 mean±SD (n) |  | GRASP-v2 k80 | GRASP-Pro f80match |
|---|---|---|---|
| cortex_peak_ratio | 0.8742 ± 0.01 (3) |  | 0.7884 | 0.9924 |
| cortex_washout_ratio | 0.978 ± 0.0035 (3) |  | 1.046 | 1.012 |
| medulla_peak_ratio | 0.9339 ± 0.0052 (3) |  | 1.025 | 0.958 |
| medulla_washout_ratio | 0.9474 ± 0.0059 (3) |  | 1.046 | 0.933 |
| aorta_peak_ratio | 1.028 ± 0.028 (3) |  | 0.7434 | 0.6819 |
| aorta_washout_ratio | 1.061 ± 0.042 (3) |  | 1.008 | 0.8691 |
| mf_aorta_affine | 0.0438 ± 0.0029 (3) |  | 0.09051 | 0.2972 |
| mf_cortex_affine | 0.03911 ± 0.00098 (3) |  | 0.04518 | 0.05357 |
| mf_medulla_affine | 0.02406 ± 0.0021 (3) |  | 0.0288 | 0.05778 |
| mf_liver_affine | 0.02136 ± 8.1e-05 (3) |  | 0.02232 | 0.03045 |
| mf_aorta_scale | 0.0581 ± 0.0049 (3) |  | 0.1248 | 0.3905 |
| mf_cortex_scale | 0.0601 ± 0.0016 (3) |  | 0.07049 | 0.08244 |
| mf_medulla_scale | 0.03568 ± 0.0032 (3) |  | 0.04372 | 0.08555 |
| aorta_peak_ratio_vs_mf | 1.028 ± 0.028 (3) |  | 0.7434 | 0.6819 |
| aorta_fwhm_s | 15.36 ± 0 (3) |  | 47.62 | 18.43 |
| aorta_ttp_s | 63.71 ± 0.72 (3) |  | 63.19 | 69.34 |
| aorta_neg_frac | 0 ± 0 (3) |  | 0 | 0 |
| aorta_rise_mono | 1 ± 0 (3) |  | 1 | 0.8605 |
| cortex_medulla_late_corr | -0.2674 ± 0.073 (3) |  | -0.01639 | 0.4411 |
| train_kNMSE | 0.2482 ± 8.9e-05 (3) |  | nan | nan |
| val_kNMSE | 0.2819 ± 0.0014 (3) |  | nan | nan |
| test_kNMSE | 0.2856 ± 0.0003 (3) |  | nan | nan |
| wall_s | 2329 ± 1.6 (3) |  | nan | nan |
| peak_gpu_mb | 1.246e+04 ± 0 (3) |  | nan | nan |
| params | 5.642e+06 ± 0 (3) |  | nan | nan |

model-free aorta FWHM (s): 15.359859566998232

TEST-spoke k-space NMSE per |k| annulus:
| annulus | tofts8 |
|---|---|
| 0.00-0.06 | 2.539e-01 |
| 0.06-0.12 | 5.883e-01 |
| 0.12-0.19 | 8.098e-01 |
| 0.19-0.25 | 8.576e-01 |
| 0.25-0.31 | 8.675e-01 |
| 0.31-0.38 | 8.016e-01 |
| 0.38-0.44 | 6.692e-01 |
| 0.44-0.50 | 7.153e-01 |
| 0.50-0.56 | 7.699e-01 |
| 0.56-0.62 | 6.686e-01 |
| 0.62-0.69 | 7.235e-01 |
| 0.69-0.75 | 7.692e-01 |
| 0.75-0.81 | 7.765e-01 |
| 0.81-0.88 | 8.543e-01 |
| 0.88-0.94 | 8.581e-01 |
| 0.94-1.00 | 9.208e-01 |

## slice 19
| metric | tofts8 mean±SD (n) |  | GRASP-v2 k80 | GRASP-Pro f80match |
|---|---|---|---|
| cortex_peak_ratio | 0.9012 ± 0.005 (3) |  | 0.7471 | 0.853 |
| cortex_washout_ratio | 0.9805 ± 0.003 (3) |  | 1.046 | 1.056 |
| medulla_peak_ratio | 0.9363 ± 0.012 (3) |  | 1.015 | 1.062 |
| medulla_washout_ratio | 0.9293 ± 0.00072 (3) |  | 1.038 | 0.9018 |
| aorta_peak_ratio | 1.012 ± 0.014 (3) |  | 0.6472 | 0.3791 |
| aorta_washout_ratio | 1.058 ± 0.04 (3) |  | 1.003 | 1.042 |
| mf_aorta_affine | 0.04681 ± 0.0029 (3) |  | 0.1323 | 0.3489 |
| mf_cortex_affine | 0.03514 ± 0.00029 (3) |  | 0.0576 | 0.06224 |
| mf_medulla_affine | 0.02684 ± 0.0013 (3) |  | 0.0423 | 0.07265 |
| mf_liver_affine | 0.02307 ± 0.00017 (3) |  | 0.02405 | 0.03211 |
| mf_aorta_scale | 0.0606 ± 0.0058 (3) |  | 0.1761 | 0.4434 |
| mf_cortex_scale | 0.05334 ± 0.00043 (3) |  | 0.08963 | 0.09467 |
| mf_medulla_scale | 0.03907 ± 0.002 (3) |  | 0.0619 | 0.1055 |
| aorta_peak_ratio_vs_mf | 1.012 ± 0.014 (3) |  | 0.6472 | 0.3791 |
| aorta_fwhm_s | 15.36 ± 0 (3) |  | 92.16 | 56.83 |
| aorta_ttp_s | 64.22 ± 0.72 (3) |  | 64.73 | 69.34 |
| aorta_neg_frac | 0 ± 0 (3) |  | 0 | 0.004167 |
| aorta_rise_mono | 1 ± 0 (3) |  | 1 | 0.8837 |
| cortex_medulla_late_corr | 0.09667 ± 0.063 (3) |  | 0.05177 | 0.9518 |
| train_kNMSE | 0.2403 ± 0.00018 (3) |  | nan | nan |
| val_kNMSE | 0.2821 ± 0.00078 (3) |  | nan | nan |
| test_kNMSE | 0.2847 ± 0.00074 (3) |  | nan | nan |
| wall_s | 2662 ± 83 (3) |  | nan | nan |
| peak_gpu_mb | 1.246e+04 ± 0 (3) |  | nan | nan |
| params | 5.642e+06 ± 0 (3) |  | nan | nan |

model-free aorta FWHM (s): 15.359859566998232

TEST-spoke k-space NMSE per |k| annulus:
| annulus | tofts8 |
|---|---|
| 0.00-0.06 | 2.484e-01 |
| 0.06-0.12 | 5.592e-01 |
| 0.12-0.19 | 7.635e-01 |
| 0.19-0.25 | 8.408e-01 |
| 0.25-0.31 | 8.008e-01 |
| 0.31-0.38 | 6.161e-01 |
| 0.38-0.44 | 3.891e-01 |
| 0.44-0.50 | 5.397e-01 |
| 0.50-0.56 | 6.100e-01 |
| 0.56-0.62 | 4.971e-01 |
| 0.62-0.69 | 6.318e-01 |
| 0.69-0.75 | 6.408e-01 |
| 0.75-0.81 | 6.268e-01 |
| 0.81-0.88 | 6.424e-01 |
| 0.88-0.94 | 6.458e-01 |
| 0.94-1.00 | 6.503e-01 |

## slice 21
| metric | tofts8 mean±SD (n) |  | GRASP-v2 k80 | GRASP-Pro f80match |
|---|---|---|---|
| cortex_peak_ratio | 0.8378 ± 0.0059 (3) |  | 0.7667 | 0.638 |
| cortex_washout_ratio | 0.9734 ± 0.0067 (3) |  | 1.048 | 0.9215 |
| medulla_peak_ratio | 0.9016 ± 0.0097 (3) |  | 1.008 | 0.9193 |
| medulla_washout_ratio | 0.9651 ± 0.006 (3) |  | 1.069 | 0.897 |
| aorta_peak_ratio | 1.042 ± 0.024 (3) |  | 0.7097 | 0.2285 |
| aorta_washout_ratio | 1.131 ± 0.08 (3) |  | 0.9795 | 0.7434 |
| mf_aorta_affine | 0.05116 ± 0.0055 (3) |  | 0.1013 | 0.384 |
| mf_cortex_affine | 0.03494 ± 3.2e-05 (3) |  | 0.04423 | 0.1056 |
| mf_medulla_affine | 0.02964 ± 0.00038 (3) |  | 0.02758 | 0.07839 |
| mf_liver_affine | 0.02304 ± 0.00046 (3) |  | 0.02368 | 0.03818 |
| mf_aorta_scale | 0.07082 ± 0.012 (3) |  | 0.1419 | 0.5084 |
| mf_cortex_scale | 0.05274 ± 6.6e-05 (3) |  | 0.06828 | 0.1632 |
| mf_medulla_scale | 0.04365 ± 0.0006 (3) |  | 0.04174 | 0.1151 |
| aorta_peak_ratio_vs_mf | 1.042 ± 0.024 (3) |  | 0.7097 | 0.2285 |
| aorta_fwhm_s | 16.38 ± 0.72 (3) |  | 52.22 | 98.3 |
| aorta_ttp_s | 64.73 ± 0 (3) |  | 64.73 | 207.6 |
| aorta_neg_frac | 0 ± 0 (3) |  | 0 | 0.02083 |
| aorta_rise_mono | 1 ± 0 (3) |  | 1 | 0.6466 |
| cortex_medulla_late_corr | -0.08465 ± 0.12 (3) |  | -0.1166 | 0.9698 |
| train_kNMSE | 0.2292 ± 0.00033 (3) |  | nan | nan |
| val_kNMSE | 0.2542 ± 0.0013 (3) |  | nan | nan |
| test_kNMSE | 0.2582 ± 0.0005 (3) |  | nan | nan |
| wall_s | 2641 ± 84 (3) |  | nan | nan |
| peak_gpu_mb | 1.246e+04 ± 0 (3) |  | nan | nan |
| params | 5.642e+06 ± 0 (3) |  | nan | nan |

model-free aorta FWHM (s): 16.89584552369808

TEST-spoke k-space NMSE per |k| annulus:
| annulus | tofts8 |
|---|---|
| 0.00-0.06 | 2.362e-01 |
| 0.06-0.12 | 4.879e-01 |
| 0.12-0.19 | 5.922e-01 |
| 0.19-0.25 | 8.248e-01 |
| 0.25-0.31 | 9.521e-01 |
| 0.31-0.38 | 1.012e+00 |
| 0.38-0.44 | 9.898e-01 |
| 0.44-0.50 | 1.122e+00 |
| 0.50-0.56 | 1.226e+00 |
| 0.56-0.62 | 1.240e+00 |
| 0.62-0.69 | 1.305e+00 |
| 0.69-0.75 | 1.371e+00 |
| 0.75-0.81 | 1.396e+00 |
| 0.81-0.88 | 1.412e+00 |
| 0.88-0.94 | 1.417e+00 |
| 0.94-1.00 | 1.439e+00 |

## reference peak correction

the 31-spoke model-free reference (6.8 s window) clips the first-pass peak; factor = peak at an 11-spoke window / peak at 31 spokes (`mf_peak_check.py`, per-spoke normalized, approved rois). corrected ratio = ratio vs the 31-spoke reference / factor. state this with every peak comparison.

| slice | roi | clip factor | tofts8 corrected peak ratio | GRASP-v2 corrected | GRASP-Pro corrected |
|---|---|---|---|---|---|
| 18 | cortex | 1.12 | 0.78 (raw 0.87) | 0.70 (raw 0.79) | 0.88 (raw 0.99) |
| 18 | medulla | 1.07 | 0.87 (raw 0.93) | 0.96 (raw 1.02) | 0.90 (raw 0.96) |
| 18 | aorta | 1.09 | 0.94 (raw 1.03) | 0.68 (raw 0.74) | 0.62 (raw 0.68) |
| 19 | cortex | 1.09 | 0.82 (raw 0.90) | 0.68 (raw 0.75) | 0.78 (raw 0.85) |
| 19 | medulla | 1.06 | 0.88 (raw 0.94) | 0.96 (raw 1.02) | 1.00 (raw 1.06) |
| 19 | aorta | 1.09 | 0.93 (raw 1.01) | 0.59 (raw 0.65) | 0.35 (raw 0.38) |
| 21 | cortex | 1.07 | 0.78 (raw 0.84) | 0.72 (raw 0.77) | 0.60 (raw 0.64) |
| 21 | medulla | 1.04 | 0.87 (raw 0.90) | 0.97 (raw 1.01) | 0.89 (raw 0.92) |
| 21 | aorta | 1.10 | 0.94 (raw 1.04) | 0.64 (raw 0.71) | 0.21 (raw 0.23) |
