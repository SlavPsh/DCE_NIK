# in vivo (meas_topqmri_p14, slices 21/24/27, k80 = 1368/1708 views (v%10<8), VAL v%10==8 for early stop, TEST v%10==9 untouched; same views for every method)

rulers: mf_* = NRMSE vs model-free NUFFT ROI curve on its 240-pt grid (affine = raw+affine fit; scale = baseline-subtracted single scale). physical bounds on aorta. *_kNMSE = complex k-space NMSE at held-out spokes (NIK only). CS rows are references, NOT truth; CS held-out blocked (magnitude-only files).

## slice 21
| metric | tofts8 mean±SD (n) |  | GRASP-v2 k80 | GRASP-Pro f80match |
|---|---|---|---|
| liver_peak_ratio | 0.9923 ± 0.0092 (3) |  | 1.005 | 0.7193 |
| liver_washout_ratio | 1.064 ± 0.005 (3) |  | 1.073 | 0.8801 |
| spleen_peak_ratio | 0.9073 ± 0.008 (3) |  | 1.069 | 1.051 |
| spleen_washout_ratio | 0.9473 ± 0.0085 (3) |  | 1.128 | 1.067 |
| aorta_peak_ratio | 0.8824 ± 0.017 (3) |  | 0.8454 | 0.792 |
| aorta_washout_ratio | 1.079 ± 0.0066 (3) |  | 1.125 | 0.9252 |
| mf_aorta_affine | 0.05298 ± 0.0025 (3) |  | 0.06276 | 0.1551 |
| mf_liver_affine | 0.03962 ± 0.00017 (3) |  | 0.02897 | 0.06528 |
| mf_spleen_affine | 0.03039 ± 0.0003 (3) |  | 0.03147 | 0.05841 |
| mf_static_affine | 0.01974 ± 0.00041 (3) |  | 0.01614 | 0.03601 |
| mf_aorta_scale | 0.08186 ± 0.0036 (3) |  | 0.1004 | 0.246 |
| mf_liver_scale | 0.07909 ± 0.00041 (3) |  | 0.05826 | 0.1315 |
| mf_spleen_scale | 0.05279 ± 0.00082 (3) |  | 0.05689 | 0.1023 |
| aorta_peak_ratio_vs_mf | 0.8824 ± 0.017 (3) |  | 0.8454 | 0.792 |
| aorta_fwhm_s | 342.3 ± 0.6 (3) |  | 345.7 | 234.3 |
| aorta_ttp_s | 53.78 ± 1.2 (3) |  | 53.36 | 62.23 |
| aorta_neg_frac | 0 ± 0 (3) |  | 0 | 0.006579 |
| aorta_rise_mono | 1 ± 0 (3) |  | 1 | 0.8511 |
| cortex_medulla_late_corr | 0.9753 ± 0.0016 (3) |  | 0.9812 | 0.3957 |
| train_kNMSE | 0.2498 ± 0.0005 (3) |  | nan | nan |
| val_kNMSE | 0.2689 ± 0.00049 (3) |  | nan | nan |
| test_kNMSE | 0.2714 ± 0.0013 (3) |  | nan | nan |
| wall_s | 5529 ± 5.4 (3) |  | nan | nan |
| peak_gpu_mb | 1.283e+04 ± 0 (3) |  | nan | nan |
| params | 5.642e+06 ± 0 (3) |  | nan | nan |

model-free aorta FWHM (s): 164.61038961038963

TEST-spoke k-space NMSE per |k| annulus:
| annulus | tofts8 |
|---|---|
| 0.00-0.06 | 2.520e-01 |
| 0.06-0.12 | 5.528e-01 |
| 0.12-0.19 | 7.668e-01 |
| 0.19-0.25 | 8.164e-01 |
| 0.25-0.31 | 7.806e-01 |
| 0.31-0.38 | 7.761e-01 |
| 0.38-0.44 | 8.216e-01 |
| 0.44-0.50 | 8.371e-01 |
| 0.50-0.56 | 8.349e-01 |
| 0.56-0.62 | 8.567e-01 |
| 0.62-0.69 | 9.254e-01 |
| 0.69-0.75 | 9.229e-01 |
| 0.75-0.81 | 9.583e-01 |
| 0.81-0.88 | 9.322e-01 |
| 0.88-0.94 | 9.805e-01 |
| 0.94-1.00 | 1.008e+00 |

## slice 24
| metric | tofts8 mean±SD (n) |  | GRASP-v2 k80 | GRASP-Pro f80match |
|---|---|---|---|
| liver_peak_ratio | 0.9638 ± 0.0073 (3) |  | 0.9963 | 0.6594 |
| liver_washout_ratio | 1.02 ± 0.0052 (3) |  | 1.071 | 0.7786 |
| spleen_peak_ratio | 0.8536 ± 0.0055 (3) |  | 1.006 | 0.5909 |
| spleen_washout_ratio | 0.9134 ± 0.0076 (3) |  | 1.082 | 0.6307 |
| aorta_peak_ratio | 0.8748 ± 0.013 (3) |  | 0.8087 | 0.4435 |
| aorta_washout_ratio | 1.01 ± 0.022 (3) |  | 1.093 | 0.4443 |
| mf_aorta_affine | 0.05256 ± 0.0038 (3) |  | 0.08018 | 0.1783 |
| mf_liver_affine | 0.02784 ± 0.00017 (3) |  | 0.02434 | 0.06781 |
| mf_spleen_affine | 0.02945 ± 0.00071 (3) |  | 0.03766 | 0.0698 |
| mf_static_affine | 0.02307 ± 0.00063 (3) |  | 0.01905 | 0.05442 |
| mf_aorta_scale | 0.07505 ± 0.0055 (3) |  | 0.1182 | 0.2871 |
| mf_liver_scale | 0.05712 ± 0.00034 (3) |  | 0.05001 | 0.1391 |
| mf_spleen_scale | 0.05082 ± 0.0012 (3) |  | 0.06732 | 0.1199 |
| aorta_peak_ratio_vs_mf | 0.8748 ± 0.013 (3) |  | 0.8087 | 0.4435 |
| aorta_fwhm_s | 312.3 ± 26 (3) |  | 345.7 | 70.91 |
| aorta_ttp_s | 49.14 ± 1.2 (3) |  | 69.82 | 59.69 |
| aorta_neg_frac | 0 ± 0 (3) |  | 0 | 0.009868 |
| aorta_rise_mono | 1 ± 0 (3) |  | 1 | 0.8 |
| cortex_medulla_late_corr | 0.9934 ± 0.00022 (3) |  | 0.9912 | -0.02402 |
| train_kNMSE | 0.2375 ± 0.00056 (3) |  | nan | nan |
| val_kNMSE | 0.2543 ± 0.00032 (3) |  | nan | nan |
| test_kNMSE | 0.2564 ± 0.00032 (3) |  | nan | nan |
| wall_s | 5529 ± 5.9 (3) |  | nan | nan |
| peak_gpu_mb | 1.283e+04 ± 0 (3) |  | nan | nan |
| params | 5.642e+06 ± 0 (3) |  | nan | nan |

model-free aorta FWHM (s): 162.07792207792212

TEST-spoke k-space NMSE per |k| annulus:
| annulus | tofts8 |
|---|---|
| 0.00-0.06 | 2.386e-01 |
| 0.06-0.12 | 5.588e-01 |
| 0.12-0.19 | 7.449e-01 |
| 0.19-0.25 | 7.297e-01 |
| 0.25-0.31 | 7.475e-01 |
| 0.31-0.38 | 6.877e-01 |
| 0.38-0.44 | 8.156e-01 |
| 0.44-0.50 | 6.999e-01 |
| 0.50-0.56 | 7.188e-01 |
| 0.56-0.62 | 7.547e-01 |
| 0.62-0.69 | 6.720e-01 |
| 0.69-0.75 | 7.165e-01 |
| 0.75-0.81 | 7.542e-01 |
| 0.81-0.88 | 7.434e-01 |
| 0.88-0.94 | 7.136e-01 |
| 0.94-1.00 | 7.136e-01 |

## slice 27
| metric | tofts8 mean±SD (n) |  | GRASP-v2 k80 | GRASP-Pro f80match |
|---|---|---|---|
| liver_peak_ratio | 0.9919 ± 0.013 (3) |  | 0.9778 | 0.7799 |
| liver_washout_ratio | 1.046 ± 0.0079 (3) |  | 1.047 | 0.8155 |
| spleen_peak_ratio | 0.8409 ± 0.0083 (3) |  | 1.005 | 0.5465 |
| spleen_washout_ratio | 0.8958 ± 0.012 (3) |  | 1.08 | 0.7082 |
| aorta_peak_ratio | 0.8707 ± 0.024 (3) |  | 0.8151 | 0.4317 |
| aorta_washout_ratio | 1.03 ± 0.061 (3) |  | 1.06 | 0.475 |
| mf_aorta_affine | 0.05176 ± 0.0049 (3) |  | 0.0594 | 0.1828 |
| mf_liver_affine | 0.02372 ± 0.00027 (3) |  | 0.02041 | 0.05849 |
| mf_spleen_affine | 0.02935 ± 0.00088 (3) |  | 0.03264 | 0.09962 |
| mf_static_affine | 0.02128 ± 0.00022 (3) |  | 0.02001 | 0.04029 |
| mf_aorta_scale | 0.07173 ± 0.0067 (3) |  | 0.08333 | 0.2832 |
| mf_liver_scale | 0.0502 ± 0.00062 (3) |  | 0.04324 | 0.1236 |
| mf_spleen_scale | 0.0521 ± 0.0016 (3) |  | 0.06003 | 0.1746 |
| aorta_peak_ratio_vs_mf | 0.8707 ± 0.024 (3) |  | 0.8151 | 0.4317 |
| aorta_fwhm_s | 321.2 ± 32 (3) |  | 345.7 | 46.85 |
| aorta_ttp_s | 51.67 ± 3.9 (3) |  | 68.56 | 59.69 |
| aorta_neg_frac | 0 ± 0 (3) |  | 0 | 0.01316 |
| aorta_rise_mono | 0.9922 ± 0.011 (3) |  | 1 | 0.8 |
| cortex_medulla_late_corr | 0.9892 ± 0.0014 (3) |  | 0.9975 | 0.2277 |
| train_kNMSE | 0.2149 ± 9.5e-05 (3) |  | nan | nan |
| val_kNMSE | 0.223 ± 0.00051 (3) |  | nan | nan |
| test_kNMSE | 0.226 ± 0.0002 (3) |  | nan | nan |
| wall_s | 5567 ± 33 (3) |  | nan | nan |
| peak_gpu_mb | 1.283e+04 ± 0 (3) |  | nan | nan |
| params | 5.642e+06 ± 0 (3) |  | nan | nan |

model-free aorta FWHM (s): 196.26623376623377

TEST-spoke k-space NMSE per |k| annulus:
| annulus | tofts8 |
|---|---|
| 0.00-0.06 | 2.176e-01 |
| 0.06-0.12 | 5.001e-01 |
| 0.12-0.19 | 5.776e-01 |
| 0.19-0.25 | 6.620e-01 |
| 0.25-0.31 | 7.048e-01 |
| 0.31-0.38 | 7.619e-01 |
| 0.38-0.44 | 8.264e-01 |
| 0.44-0.50 | 8.615e-01 |
| 0.50-0.56 | 9.018e-01 |
| 0.56-0.62 | 9.505e-01 |
| 0.62-0.69 | 9.799e-01 |
| 0.69-0.75 | 9.910e-01 |
| 0.75-0.81 | 1.023e+00 |
| 0.81-0.88 | 1.036e+00 |
| 0.88-0.94 | 1.061e+00 |
| 0.94-1.00 | 1.091e+00 |

## reference peak correction

the 31-spoke model-free reference (6.8 s window) clips the first-pass peak; factor = peak at an 11-spoke window / peak at 31 spokes (`mf_peak_check.py`, per-spoke normalized, approved rois). corrected ratio = ratio vs the 31-spoke reference / factor. state this with every peak comparison.

| slice | roi | clip factor | tofts8 corrected peak ratio | GRASP-v2 corrected | GRASP-Pro corrected |
|---|---|---|---|---|---|
| 21 | liver | 1.06 | 0.93 (raw 0.99) | 0.94 (raw 1.01) | 0.68 (raw 0.72) |
| 21 | spleen | 1.07 | 0.85 (raw 0.91) | 1.00 (raw 1.07) | 0.98 (raw 1.05) |
| 21 | aorta | 1.09 | 0.81 (raw 0.88) | 0.77 (raw 0.85) | 0.73 (raw 0.79) |
| 24 | liver | 1.09 | 0.88 (raw 0.96) | 0.91 (raw 1.00) | 0.61 (raw 0.66) |
| 24 | spleen | 1.08 | 0.79 (raw 0.85) | 0.93 (raw 1.01) | 0.54 (raw 0.59) |
| 24 | aorta | 1.02 | 0.86 (raw 0.87) | 0.79 (raw 0.81) | 0.43 (raw 0.44) |
| 27 | liver | 1.13 | 0.88 (raw 0.99) | 0.87 (raw 0.98) | 0.69 (raw 0.78) |
| 27 | spleen | 1.09 | 0.77 (raw 0.84) | 0.93 (raw 1.01) | 0.50 (raw 0.55) |
| 27 | aorta | 1.07 | 0.81 (raw 0.87) | 0.76 (raw 0.82) | 0.40 (raw 0.43) |
