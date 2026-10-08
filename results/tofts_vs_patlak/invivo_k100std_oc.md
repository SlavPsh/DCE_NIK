# in vivo (meas_p3_dce, slices 18/19/21, k100 = every view in training (no held-out spokes; the val / test kNMSE columns are TRAIN-set numbers here); same views for every method)

rulers: mf_* = NRMSE vs model-free NUFFT ROI curve on its 240-pt grid (affine = raw+affine fit; scale = baseline-subtracted single scale). physical bounds on aorta. *_kNMSE = complex k-space NMSE at held-out spokes (NIK only). CS rows are references, NOT truth; CS held-out blocked (magnitude-only files).

## slice 18
| metric | tofts8 mean±SD (n) |  | GRASP-v2 all spokes |
|---|---|---|---|
| cortex_peak_ratio | 0.875 ± 0 (1) |  | 0.8153 | nan |
| cortex_washout_ratio | 0.966 ± 0 (1) |  | 1.054 | nan |
| medulla_peak_ratio | 0.932 ± 0 (1) |  | 1.039 | nan |
| medulla_washout_ratio | 0.9304 ± 0 (1) |  | 1.053 | nan |
| aorta_peak_ratio | 1.013 ± 0 (1) |  | 0.7927 | nan |
| aorta_washout_ratio | 1.078 ± 0 (1) |  | 1.019 | nan |
| mf_aorta_affine | 0.04097 ± 0 (1) |  | 0.07915 | nan |
| mf_cortex_affine | 0.03605 ± 0 (1) |  | 0.04022 | nan |
| mf_medulla_affine | 0.02692 ± 0 (1) |  | 0.02661 | nan |
| mf_liver_affine | 0.02152 ± 0 (1) |  | 0.02198 | nan |
| mf_aorta_scale | 0.05423 ± 0 (1) |  | 0.1089 | nan |
| mf_cortex_scale | 0.0556 ± 0 (1) |  | 0.06304 | nan |
| mf_medulla_scale | 0.03985 ± 0 (1) |  | 0.04044 | nan |
| aorta_peak_ratio_vs_mf | 1.013 ± 0 (1) |  | 0.7927 | nan |
| aorta_fwhm_s | 16.9 ± 0 (1) |  | 21.5 | nan |
| aorta_ttp_s | 63.19 ± 0 (1) |  | 64.73 | nan |
| aorta_neg_frac | 0 ± 0 (1) |  | 0 | nan |
| aorta_rise_mono | 1 ± 0 (1) |  | 1 | nan |
| cortex_medulla_late_corr | 0.2331 ± 0 (1) |  | -0.01859 | nan |
| train_kNMSE | 0.2517 ± 0 (1) |  | nan | nan |
| val_kNMSE | 0.2523 ± 0 (1) |  | nan | nan |
| test_kNMSE | 0.2547 ± 0 (1) |  | nan | nan |
| wall_s | 3875 ± 0 (1) |  | nan | nan |
| peak_gpu_mb | 1.248e+04 ± 0 (1) |  | nan | nan |
| params | 5.642e+06 ± 0 (1) |  | nan | nan |

model-free aorta FWHM (s): 15.359859566998232

TEST-spoke k-space NMSE per |k| annulus:
| annulus | tofts8 |
|---|---|
| 0.00-0.06 | 2.359e-01 |
| 0.06-0.12 | 4.751e-01 |
| 0.12-0.19 | 5.263e-01 |
| 0.19-0.25 | 5.735e-01 |
| 0.25-0.31 | 5.567e-01 |
| 0.31-0.38 | 4.683e-01 |
| 0.38-0.44 | 3.421e-01 |
| 0.44-0.50 | 3.628e-01 |
| 0.50-0.56 | 4.154e-01 |
| 0.56-0.62 | 3.245e-01 |
| 0.62-0.69 | 3.545e-01 |
| 0.69-0.75 | 3.838e-01 |
| 0.75-0.81 | 3.438e-01 |
| 0.81-0.88 | 3.719e-01 |
| 0.88-0.94 | 3.628e-01 |
| 0.94-1.00 | 4.101e-01 |

## slice 19
| metric | tofts8 mean±SD (n) |  | GRASP-v2 all spokes |
|---|---|---|---|
| cortex_peak_ratio | 0.8986 ± 0 (1) |  | 0.7625 | nan |
| cortex_washout_ratio | 0.9731 ± 0 (1) |  | 1.055 | nan |
| medulla_peak_ratio | 0.9129 ± 0 (1) |  | 1.025 | nan |
| medulla_washout_ratio | 0.9141 ± 0 (1) |  | 1.043 | nan |
| aorta_peak_ratio | 1.058 ± 0 (1) |  | 0.7134 | nan |
| aorta_washout_ratio | 0.9984 ± 0 (1) |  | 1.027 | nan |
| mf_aorta_affine | 0.04151 ± 0 (1) |  | 0.1115 | nan |
| mf_cortex_affine | 0.03475 ± 0 (1) |  | 0.05242 | nan |
| mf_medulla_affine | 0.0291 ± 0 (1) |  | 0.04 | nan |
| mf_liver_affine | 0.02302 ± 0 (1) |  | 0.02378 | nan |
| mf_aorta_scale | 0.05367 ± 0 (1) |  | 0.1504 | nan |
| mf_cortex_scale | 0.05279 ± 0 (1) |  | 0.08204 | nan |
| mf_medulla_scale | 0.04241 ± 0 (1) |  | 0.05853 | nan |
| aorta_peak_ratio_vs_mf | 1.058 ± 0 (1) |  | 0.7134 | nan |
| aorta_fwhm_s | 15.36 ± 0 (1) |  | 66.05 | nan |
| aorta_ttp_s | 64.73 ± 0 (1) |  | 64.73 | nan |
| aorta_neg_frac | 0 ± 0 (1) |  | 0 | nan |
| aorta_rise_mono | 1 ± 0 (1) |  | 1 | nan |
| cortex_medulla_late_corr | 0.2938 ± 0 (1) |  | 0.04944 | nan |
| train_kNMSE | 0.2575 ± 0 (1) |  | nan | nan |
| val_kNMSE | 0.2582 ± 0 (1) |  | nan | nan |
| test_kNMSE | 0.2598 ± 0 (1) |  | nan | nan |
| wall_s | 3933 ± 0 (1) |  | nan | nan |
| peak_gpu_mb | 1.248e+04 ± 0 (1) |  | nan | nan |
| params | 5.642e+06 ± 0 (1) |  | nan | nan |

model-free aorta FWHM (s): 15.359859566998232

TEST-spoke k-space NMSE per |k| annulus:
| annulus | tofts8 |
|---|---|
| 0.00-0.06 | 2.339e-01 |
| 0.06-0.12 | 4.637e-01 |
| 0.12-0.19 | 5.300e-01 |
| 0.19-0.25 | 7.101e-01 |
| 0.25-0.31 | 6.919e-01 |
| 0.31-0.38 | 4.313e-01 |
| 0.38-0.44 | 2.643e-01 |
| 0.44-0.50 | 5.274e-01 |
| 0.50-0.56 | 5.918e-01 |
| 0.56-0.62 | 4.037e-01 |
| 0.62-0.69 | 5.021e-01 |
| 0.69-0.75 | 4.992e-01 |
| 0.75-0.81 | 4.935e-01 |
| 0.81-0.88 | 4.749e-01 |
| 0.88-0.94 | 4.682e-01 |
| 0.94-1.00 | 4.610e-01 |

## slice 21
| metric | tofts8 mean±SD (n) |  | GRASP-v2 all spokes |
|---|---|---|---|
| cortex_peak_ratio | 0.8718 ± 0 (1) |  | 0.7881 | nan |
| cortex_washout_ratio | 0.9479 ± 0 (1) |  | 1.053 | nan |
| medulla_peak_ratio | 0.9012 ± 0 (1) |  | 1.015 | nan |
| medulla_washout_ratio | 0.9524 ± 0 (1) |  | 1.072 | nan |
| aorta_peak_ratio | 1.054 ± 0 (1) |  | 0.7702 | nan |
| aorta_washout_ratio | 1.063 ± 0 (1) |  | 1.012 | nan |
| mf_aorta_affine | 0.049 ± 0 (1) |  | 0.08901 | nan |
| mf_cortex_affine | 0.02814 ± 0 (1) |  | 0.03944 | nan |
| mf_medulla_affine | 0.03349 ± 0 (1) |  | 0.02517 | nan |
| mf_liver_affine | 0.0233 ± 0 (1) |  | 0.02321 | nan |
| mf_aorta_scale | 0.06321 ± 0 (1) |  | 0.1242 | nan |
| mf_cortex_scale | 0.04278 ± 0 (1) |  | 0.06079 | nan |
| mf_medulla_scale | 0.04917 ± 0 (1) |  | 0.03846 | nan |
| aorta_peak_ratio_vs_mf | 1.054 ± 0 (1) |  | 0.7702 | nan |
| aorta_fwhm_s | 16.9 ± 0 (1) |  | 35.33 | nan |
| aorta_ttp_s | 64.73 ± 0 (1) |  | 64.73 | nan |
| aorta_neg_frac | 0 ± 0 (1) |  | 0 | nan |
| aorta_rise_mono | 1 ± 0 (1) |  | 1 | nan |
| cortex_medulla_late_corr | 0.1639 ± 0 (1) |  | -0.1108 | nan |
| train_kNMSE | 0.2281 ± 0 (1) |  | nan | nan |
| val_kNMSE | 0.2281 ± 0 (1) |  | nan | nan |
| test_kNMSE | 0.2299 ± 0 (1) |  | nan | nan |
| wall_s | 3827 ± 0 (1) |  | nan | nan |
| peak_gpu_mb | 1.248e+04 ± 0 (1) |  | nan | nan |
| params | 5.642e+06 ± 0 (1) |  | nan | nan |

model-free aorta FWHM (s): 16.89584552369808

TEST-spoke k-space NMSE per |k| annulus:
| annulus | tofts8 |
|---|---|
| 0.00-0.06 | 2.191e-01 |
| 0.06-0.12 | 3.915e-01 |
| 0.12-0.19 | 3.931e-01 |
| 0.19-0.25 | 4.649e-01 |
| 0.25-0.31 | 4.889e-01 |
| 0.31-0.38 | 4.439e-01 |
| 0.38-0.44 | 3.734e-01 |
| 0.44-0.50 | 4.231e-01 |
| 0.50-0.56 | 4.186e-01 |
| 0.56-0.62 | 3.629e-01 |
| 0.62-0.69 | 3.642e-01 |
| 0.69-0.75 | 3.479e-01 |
| 0.75-0.81 | 3.486e-01 |
| 0.81-0.88 | 3.356e-01 |
| 0.88-0.94 | 3.436e-01 |
| 0.94-1.00 | 3.323e-01 |

## reference peak correction

the 31-spoke model-free reference (6.8 s window) clips the first-pass peak; factor = peak at an 11-spoke window / peak at 31 spokes (`mf_peak_check.py`, per-spoke normalized, approved rois). corrected ratio = ratio vs the 31-spoke reference / factor. state this with every peak comparison.

| slice | roi | clip factor | tofts8 corrected peak ratio | GRASP-v2 corrected | GRASP-Pro corrected |
|---|---|---|---|---|---|
| 18 | cortex | 1.12 | 0.78 (raw 0.88) | 0.73 (raw 0.82) | - |
| 18 | medulla | 1.07 | 0.87 (raw 0.93) | 0.97 (raw 1.04) | - |
| 18 | aorta | 1.09 | 0.93 (raw 1.01) | 0.73 (raw 0.79) | - |
| 19 | cortex | 1.09 | 0.82 (raw 0.90) | 0.70 (raw 0.76) | - |
| 19 | medulla | 1.06 | 0.86 (raw 0.91) | 0.96 (raw 1.02) | - |
| 19 | aorta | 1.09 | 0.97 (raw 1.06) | 0.65 (raw 0.71) | - |
| 21 | cortex | 1.07 | 0.81 (raw 0.87) | 0.74 (raw 0.79) | - |
| 21 | medulla | 1.04 | 0.87 (raw 0.90) | 0.98 (raw 1.02) | - |
| 21 | aorta | 1.10 | 0.95 (raw 1.05) | 0.70 (raw 0.77) | - |
