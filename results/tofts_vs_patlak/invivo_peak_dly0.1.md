# in vivo (meas_p3_dce, slices 21, k100 = every view in training (no held-out spokes; the val / test kNMSE columns are TRAIN-set numbers here); same views for every method)

rulers: mf_* = NRMSE vs model-free NUFFT ROI curve on its 240-pt grid (affine = raw+affine fit; scale = baseline-subtracted single scale). physical bounds on aorta. *_kNMSE = complex k-space NMSE at held-out spokes (NIK only). CS rows are references, NOT truth; CS held-out blocked (magnitude-only files).

## slice 21
| metric | tofts8 mean±SD (n) |  | GRASP-v2 all spokes |
|---|---|---|---|
| cortex_peak_ratio | 0.8678 ± 0.0073 (3) |  | 0.7881 | nan |
| cortex_washout_ratio | 0.9693 ± 0.0082 (3) |  | 1.053 | nan |
| medulla_peak_ratio | 0.9352 ± 0.012 (3) |  | 1.015 | nan |
| medulla_washout_ratio | 0.9921 ± 0.0094 (3) |  | 1.072 | nan |
| aorta_peak_ratio | 1.055 ± 0.02 (3) |  | 0.7702 | nan |
| aorta_washout_ratio | 1.084 ± 0.079 (3) |  | 1.012 | nan |
| mf_aorta_affine | 0.04858 ± 0.0031 (3) |  | 0.08901 | nan |
| mf_cortex_affine | 0.02896 ± 0.00016 (3) |  | 0.03944 | nan |
| mf_medulla_affine | 0.02855 ± 0.0012 (3) |  | 0.02517 | nan |
| mf_liver_affine | 0.02537 ± 0.0012 (3) |  | 0.02321 | nan |
| mf_aorta_scale | 0.06434 ± 0.0057 (3) |  | 0.1242 | nan |
| mf_cortex_scale | 0.04389 ± 0.00025 (3) |  | 0.06079 | nan |
| mf_medulla_scale | 0.04188 ± 0.0017 (3) |  | 0.03846 | nan |
| aorta_peak_ratio_vs_mf | 1.055 ± 0.02 (3) |  | 0.7702 | nan |
| aorta_fwhm_s | 16.9 ± 0 (3) |  | 35.33 | nan |
| aorta_ttp_s | 64.73 ± 0 (3) |  | 64.73 | nan |
| aorta_neg_frac | 0 ± 0 (3) |  | 0 | nan |
| aorta_rise_mono | 1 ± 0 (3) |  | 1 | nan |
| cortex_medulla_late_corr | 0.08682 ± 0.043 (3) |  | -0.1108 | nan |
| train_kNMSE | 0.2273 ± 0.00042 (3) |  | nan | nan |
| val_kNMSE | 0.226 ± 0.00025 (3) |  | nan | nan |
| test_kNMSE | 0.2288 ± 0.00082 (3) |  | nan | nan |
| wall_s | 3861 ± 0.88 (3) |  | nan | nan |
| peak_gpu_mb | 1.248e+04 ± 0 (3) |  | nan | nan |
| params | 5.531e+06 ± 0 (3) |  | nan | nan |

model-free aorta FWHM (s): 16.89584552369808

TEST-spoke k-space NMSE per |k| annulus:
| annulus | tofts8 |
|---|---|
| 0.00-0.06 | 2.172e-01 |
| 0.06-0.12 | 3.930e-01 |
| 0.12-0.19 | 4.010e-01 |
| 0.19-0.25 | 4.966e-01 |
| 0.25-0.31 | 5.226e-01 |
| 0.31-0.38 | 5.111e-01 |
| 0.38-0.44 | 4.338e-01 |
| 0.44-0.50 | 4.716e-01 |
| 0.50-0.56 | 4.721e-01 |
| 0.56-0.62 | 4.097e-01 |
| 0.62-0.69 | 4.190e-01 |
| 0.69-0.75 | 4.029e-01 |
| 0.75-0.81 | 4.056e-01 |
| 0.81-0.88 | 3.806e-01 |
| 0.88-0.94 | 3.969e-01 |
| 0.94-1.00 | 3.793e-01 |

## reference peak correction

the 31-spoke model-free reference (6.8 s window) clips the first-pass peak; factor = peak at an 11-spoke window / peak at 31 spokes (`mf_peak_check.py`, per-spoke normalized, approved rois). corrected ratio = ratio vs the 31-spoke reference / factor. state this with every peak comparison.

| slice | roi | clip factor | tofts8 corrected peak ratio | GRASP-v2 corrected | GRASP-Pro corrected |
|---|---|---|---|---|---|
| 18 | cortex | 1.12 | - | - | - |
| 18 | medulla | 1.07 | - | - | - |
| 18 | aorta | 1.09 | - | - | - |
| 19 | cortex | 1.09 | - | - | - |
| 19 | medulla | 1.06 | - | - | - |
| 19 | aorta | 1.09 | - | - | - |
| 21 | cortex | 1.07 | 0.81 (raw 0.87) | 0.74 (raw 0.79) | - |
| 21 | medulla | 1.04 | 0.90 (raw 0.94) | 0.98 (raw 1.02) | - |
| 21 | aorta | 1.10 | 0.96 (raw 1.06) | 0.70 (raw 0.77) | - |
