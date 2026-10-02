# in vivo (meas_topqmri_p14, slices 21/24/27, k80 = 1368/1708 views (v%10<8), VAL v%10==8 for early stop, TEST v%10==9 untouched; same views for every method)

rulers: mf_* = NRMSE vs model-free NUFFT ROI curve on its 240-pt grid (affine = raw+affine fit; scale = baseline-subtracted single scale). physical bounds on aorta. *_kNMSE = complex k-space NMSE at held-out spokes (NIK only). CS rows are references, NOT truth; CS held-out blocked (magnitude-only files).

## slice 21
| metric | sub16 mean±SD (n) |  | GRASP-v2 k80 | GRASP-Pro f80match |
|---|---|---|---|
| liver_peak_ratio | 0.7996 ± 0 (1) |  | 1.005 | 0.7193 |
| liver_washout_ratio | 0.8126 ± 0 (1) |  | 1.073 | 0.8801 |
| spleen_peak_ratio | 0.648 ± 0 (1) |  | 1.069 | 1.051 |
| spleen_washout_ratio | 0.7547 ± 0 (1) |  | 1.128 | 1.067 |
| aorta_peak_ratio | 0.5536 ± 0 (1) |  | 0.8454 | 0.792 |
| aorta_washout_ratio | 0.6737 ± 0 (1) |  | 1.125 | 0.9252 |
| mf_aorta_affine | 0.1876 ± 0 (1) |  | 0.06276 | 0.1551 |
| mf_liver_affine | 0.04912 ± 0 (1) |  | 0.02897 | 0.06528 |
| mf_spleen_affine | 0.1146 ± 0 (1) |  | 0.03147 | 0.05841 |
| mf_static_affine | 0.05209 ± 0 (1) |  | 0.01614 | 0.03601 |
| mf_aorta_scale | 0.3207 ± 0 (1) |  | 0.1004 | 0.246 |
| mf_liver_scale | 0.0979 ± 0 (1) |  | 0.05826 | 0.1315 |
| mf_spleen_scale | 0.1971 ± 0 (1) |  | 0.05689 | 0.1023 |
| aorta_peak_ratio_vs_mf | 0.5536 ± 0 (1) |  | 0.8454 | 0.792 |
| aorta_fwhm_s | 84.84 ± 0 (1) |  | 345.7 | 234.3 |
| aorta_ttp_s | 85.02 ± 0 (1) |  | 53.36 | 62.23 |
| aorta_neg_frac | 0.02961 ± 0 (1) |  | 0 | 0.006579 |
| aorta_rise_mono | 0.6154 ± 0 (1) |  | 1 | 0.8511 |
| cortex_medulla_late_corr | 0.9427 ± 0 (1) |  | 0.9812 | 0.3957 |
| train_kNMSE | 0.05604 ± 0 (1) |  | nan | nan |
| val_kNMSE | 0.06303 ± 0 (1) |  | nan | nan |
| test_kNMSE | 0.06155 ± 0 (1) |  | nan | nan |
| wall_s | 5760 ± 0 (1) |  | nan | nan |
| peak_gpu_mb | 1.284e+04 ± 0 (1) |  | nan | nan |
| params | 5.558e+06 ± 0 (1) |  | nan | nan |

model-free aorta FWHM (s): 164.61038961038963

TEST-spoke k-space NMSE per |k| annulus:
| annulus | sub16 |
|---|---|
| 0.00-0.06 | 5.556e-02 |
| 0.06-0.12 | 1.251e-01 |
| 0.12-0.19 | 1.652e-01 |
| 0.19-0.25 | 2.058e-01 |
| 0.25-0.31 | 2.765e-01 |
| 0.31-0.38 | 3.431e-01 |
| 0.38-0.44 | 4.096e-01 |
| 0.44-0.50 | 4.761e-01 |
| 0.50-0.56 | 5.684e-01 |
| 0.56-0.62 | 6.112e-01 |
| 0.62-0.69 | 6.517e-01 |
| 0.69-0.75 | 7.283e-01 |
| 0.75-0.81 | 7.733e-01 |
| 0.81-0.88 | 7.902e-01 |
| 0.88-0.94 | 8.516e-01 |
| 0.94-1.00 | 9.253e-01 |

## slice 24
| metric | sub16 mean±SD (n) |  | GRASP-v2 k80 | GRASP-Pro f80match |
|---|---|---|---|
| liver_peak_ratio | 0.8508 ± 0 (1) |  | 0.9963 | 0.6594 |
| liver_washout_ratio | 0.8813 ± 0 (1) |  | 1.071 | 0.7786 |
| spleen_peak_ratio | 0.6137 ± 0 (1) |  | 1.006 | 0.5909 |
| spleen_washout_ratio | 0.7867 ± 0 (1) |  | 1.082 | 0.6307 |
| aorta_peak_ratio | 0.4724 ± 0 (1) |  | 0.8087 | 0.4435 |
| aorta_washout_ratio | 0.5534 ± 0 (1) |  | 1.093 | 0.4443 |
| mf_aorta_affine | 0.215 ± 0 (1) |  | 0.08018 | 0.1783 |
| mf_liver_affine | 0.04997 ± 0 (1) |  | 0.02434 | 0.06781 |
| mf_spleen_affine | 0.1195 ± 0 (1) |  | 0.03766 | 0.0698 |
| mf_static_affine | 0.05803 ± 0 (1) |  | 0.01905 | 0.05442 |
| mf_aorta_scale | 0.3777 ± 0 (1) |  | 0.1182 | 0.2871 |
| mf_liver_scale | 0.1035 ± 0 (1) |  | 0.05001 | 0.1391 |
| mf_spleen_scale | 0.2073 ± 0 (1) |  | 0.06732 | 0.1199 |
| aorta_peak_ratio_vs_mf | 0.4724 ± 0 (1) |  | 0.8087 | 0.4435 |
| aorta_fwhm_s | 30.39 ± 0 (1) |  | 345.7 | 70.91 |
| aorta_ttp_s | 95.15 ± 0 (1) |  | 69.82 | 59.69 |
| aorta_neg_frac | 0.05921 ± 0 (1) |  | 0 | 0.009868 |
| aorta_rise_mono | 0.6986 ± 0 (1) |  | 1 | 0.8 |
| cortex_medulla_late_corr | 0.7438 ± 0 (1) |  | 0.9912 | -0.02402 |
| train_kNMSE | 0.05652 ± 0 (1) |  | nan | nan |
| val_kNMSE | 0.06157 ± 0 (1) |  | nan | nan |
| test_kNMSE | 0.06183 ± 0 (1) |  | nan | nan |
| wall_s | 5753 ± 0 (1) |  | nan | nan |
| peak_gpu_mb | 1.284e+04 ± 0 (1) |  | nan | nan |
| params | 5.558e+06 ± 0 (1) |  | nan | nan |

model-free aorta FWHM (s): 162.07792207792212

TEST-spoke k-space NMSE per |k| annulus:
| annulus | sub16 |
|---|---|
| 0.00-0.06 | 5.687e-02 |
| 0.06-0.12 | 1.123e-01 |
| 0.12-0.19 | 1.872e-01 |
| 0.19-0.25 | 2.305e-01 |
| 0.25-0.31 | 2.415e-01 |
| 0.31-0.38 | 2.170e-01 |
| 0.38-0.44 | 2.478e-01 |
| 0.44-0.50 | 2.823e-01 |
| 0.50-0.56 | 2.964e-01 |
| 0.56-0.62 | 3.095e-01 |
| 0.62-0.69 | 3.246e-01 |
| 0.69-0.75 | 3.494e-01 |
| 0.75-0.81 | 3.994e-01 |
| 0.81-0.88 | 4.365e-01 |
| 0.88-0.94 | 4.757e-01 |
| 0.94-1.00 | 5.309e-01 |

## slice 27
| metric | sub16 mean±SD (n) |  | GRASP-v2 k80 | GRASP-Pro f80match |
|---|---|---|---|
| liver_peak_ratio | 0.8475 ± 0 (1) |  | 0.9778 | 0.7799 |
| liver_washout_ratio | 0.9196 ± 0 (1) |  | 1.047 | 0.8155 |
| spleen_peak_ratio | 0.6466 ± 0 (1) |  | 1.005 | 0.5465 |
| spleen_washout_ratio | 0.8013 ± 0 (1) |  | 1.08 | 0.7082 |
| aorta_peak_ratio | 0.4793 ± 0 (1) |  | 0.8151 | 0.4317 |
| aorta_washout_ratio | 0.6767 ± 0 (1) |  | 1.06 | 0.475 |
| mf_aorta_affine | 0.2032 ± 0 (1) |  | 0.0594 | 0.1828 |
| mf_liver_affine | 0.05458 ± 0 (1) |  | 0.02041 | 0.05849 |
| mf_spleen_affine | 0.1158 ± 0 (1) |  | 0.03264 | 0.09962 |
| mf_static_affine | 0.03664 ± 0 (1) |  | 0.02001 | 0.04029 |
| mf_aorta_scale | 0.3206 ± 0 (1) |  | 0.08333 | 0.2832 |
| mf_liver_scale | 0.1153 ± 0 (1) |  | 0.04324 | 0.1236 |
| mf_spleen_scale | 0.2096 ± 0 (1) |  | 0.06003 | 0.1746 |
| aorta_peak_ratio_vs_mf | 0.4793 ± 0 (1) |  | 0.8151 | 0.4317 |
| aorta_fwhm_s | 229.2 ± 0 (1) |  | 345.7 | 46.85 |
| aorta_ttp_s | 157.2 ± 0 (1) |  | 68.56 | 59.69 |
| aorta_neg_frac | 0.04605 ± 0 (1) |  | 0 | 0.01316 |
| aorta_rise_mono | 0.7049 ± 0 (1) |  | 1 | 0.8 |
| cortex_medulla_late_corr | 0.9272 ± 0 (1) |  | 0.9975 | 0.2277 |
| train_kNMSE | 0.05616 ± 0 (1) |  | nan | nan |
| val_kNMSE | 0.0602 ± 0 (1) |  | nan | nan |
| test_kNMSE | 0.06005 ± 0 (1) |  | nan | nan |
| wall_s | 5805 ± 0 (1) |  | nan | nan |
| peak_gpu_mb | 1.284e+04 ± 0 (1) |  | nan | nan |
| params | 5.558e+06 ± 0 (1) |  | nan | nan |

model-free aorta FWHM (s): 196.26623376623377

TEST-spoke k-space NMSE per |k| annulus:
| annulus | sub16 |
|---|---|
| 0.00-0.06 | 5.649e-02 |
| 0.06-0.12 | 1.288e-01 |
| 0.12-0.19 | 1.819e-01 |
| 0.19-0.25 | 2.750e-01 |
| 0.25-0.31 | 3.368e-01 |
| 0.31-0.38 | 4.206e-01 |
| 0.38-0.44 | 5.204e-01 |
| 0.44-0.50 | 6.032e-01 |
| 0.50-0.56 | 6.888e-01 |
| 0.56-0.62 | 8.072e-01 |
| 0.62-0.69 | 8.806e-01 |
| 0.69-0.75 | 9.548e-01 |
| 0.75-0.81 | 1.020e+00 |
| 0.81-0.88 | 1.050e+00 |
| 0.88-0.94 | 1.115e+00 |
| 0.94-1.00 | 1.153e+00 |

## reference peak correction

the 31-spoke model-free reference (6.8 s window) clips the first-pass peak; factor = peak at an 11-spoke window / peak at 31 spokes (`mf_peak_check.py`, per-spoke normalized, approved rois). corrected ratio = ratio vs the 31-spoke reference / factor. state this with every peak comparison.

| slice | roi | clip factor | sub16 corrected peak ratio | GRASP-v2 corrected | GRASP-Pro corrected |
|---|---|---|---|---|---|
| 21 | liver | 1.06 | 0.75 (raw 0.80) | 0.94 (raw 1.01) | 0.68 (raw 0.72) |
| 21 | spleen | 1.07 | 0.61 (raw 0.65) | 1.00 (raw 1.07) | 0.98 (raw 1.05) |
| 21 | aorta | 1.09 | 0.51 (raw 0.55) | 0.77 (raw 0.85) | 0.73 (raw 0.79) |
| 24 | liver | 1.09 | 0.78 (raw 0.85) | 0.91 (raw 1.00) | 0.61 (raw 0.66) |
| 24 | spleen | 1.08 | 0.57 (raw 0.61) | 0.93 (raw 1.01) | 0.54 (raw 0.59) |
| 24 | aorta | 1.02 | 0.46 (raw 0.47) | 0.79 (raw 0.81) | 0.43 (raw 0.44) |
| 27 | liver | 1.13 | 0.75 (raw 0.85) | 0.87 (raw 0.98) | 0.69 (raw 0.78) |
| 27 | spleen | 1.09 | 0.60 (raw 0.65) | 0.93 (raw 1.01) | 0.50 (raw 0.55) |
| 27 | aorta | 1.07 | 0.45 (raw 0.48) | 0.76 (raw 0.82) | 0.40 (raw 0.43) |
