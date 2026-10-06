# in vivo (meas_p3_dce, slices 21, k80 = 1368/1708 views (v%10<8), VAL v%10==8 for early stop, TEST v%10==9 untouched; same views for every method)

rulers: mf_* = NRMSE vs model-free NUFFT ROI curve on its 240-pt grid (affine = raw+affine fit; scale = baseline-subtracted single scale). physical bounds on aorta. *_kNMSE = complex k-space NMSE at held-out spokes (NIK only). CS rows are references, NOT truth; CS held-out blocked (magnitude-only files).

## slice 21
| metric | tofts5 mean±SD (n) |  | GRASP-v2 k80 | GRASP-Pro f80match |
|---|---|---|---|
| cortex_peak_ratio | 0.9644 ± 0 (1) |  | 0.7667 | 0.638 |
| cortex_washout_ratio | 1.018 ± 0 (1) |  | 1.048 | 0.9215 |
| medulla_peak_ratio | 0.9274 ± 0 (1) |  | 1.008 | 0.9193 |
| medulla_washout_ratio | 1.022 ± 0 (1) |  | 1.069 | 0.897 |
| aorta_peak_ratio | 1.051 ± 0 (1) |  | 0.7097 | 0.2285 |
| aorta_washout_ratio | 1.083 ± 0 (1) |  | 0.9795 | 0.7434 |
| mf_aorta_affine | 0.04135 ± 0 (1) |  | 0.1013 | 0.384 |
| mf_cortex_affine | 0.04762 ± 0 (1) |  | 0.04423 | 0.1056 |
| mf_medulla_affine | 0.07646 ± 0 (1) |  | 0.02758 | 0.07839 |
| mf_liver_affine | 0.02707 ± 0 (1) |  | 0.02368 | 0.03818 |
| mf_aorta_scale | 0.05446 ± 0 (1) |  | 0.1419 | 0.5084 |
| mf_cortex_scale | 0.07181 ± 0 (1) |  | 0.06828 | 0.1632 |
| mf_medulla_scale | 0.1122 ± 0 (1) |  | 0.04174 | 0.1151 |
| aorta_peak_ratio_vs_mf | 1.051 ± 0 (1) |  | 0.7097 | 0.2285 |
| aorta_fwhm_s | 16.9 ± 0 (1) |  | 52.22 | 98.3 |
| aorta_ttp_s | 63.19 ± 0 (1) |  | 64.73 | 207.6 |
| aorta_neg_frac | 0 ± 0 (1) |  | 0 | 0.02083 |
| aorta_rise_mono | 1 ± 0 (1) |  | 1 | 0.6466 |
| cortex_medulla_late_corr | -0.8706 ± 0 (1) |  | -0.1166 | 0.9698 |
| train_kNMSE | 0.2291 ± 0 (1) |  | nan | nan |
| val_kNMSE | 0.2406 ± 0 (1) |  | nan | nan |
| test_kNMSE | 0.2451 ± 0 (1) |  | nan | nan |
| wall_s | 3875 ± 0 (1) |  | nan | nan |
| peak_gpu_mb | 1.246e+04 ± 0 (1) |  | nan | nan |
| params | 5.525e+06 ± 0 (1) |  | nan | nan |

model-free aorta FWHM (s): 16.89584552369808

TEST-spoke k-space NMSE per |k| annulus:
| annulus | tofts5 |
|---|---|
| 0.00-0.06 | 2.256e-01 |
| 0.06-0.12 | 4.502e-01 |
| 0.12-0.19 | 5.311e-01 |
| 0.19-0.25 | 7.031e-01 |
| 0.25-0.31 | 8.379e-01 |
| 0.31-0.38 | 8.942e-01 |
| 0.38-0.44 | 9.436e-01 |
| 0.44-0.50 | 1.095e+00 |
| 0.50-0.56 | 1.196e+00 |
| 0.56-0.62 | 1.196e+00 |
| 0.62-0.69 | 1.293e+00 |
| 0.69-0.75 | 1.360e+00 |
| 0.75-0.81 | 1.342e+00 |
| 0.81-0.88 | 1.362e+00 |
| 0.88-0.94 | 1.365e+00 |
| 0.94-1.00 | 1.362e+00 |

## reference peak correction

the 31-spoke model-free reference (6.8 s window) clips the first-pass peak; factor = peak at an 11-spoke window / peak at 31 spokes (`mf_peak_check.py`, per-spoke normalized, approved rois). corrected ratio = ratio vs the 31-spoke reference / factor. state this with every peak comparison.

| slice | roi | clip factor | tofts5 corrected peak ratio | GRASP-v2 corrected | GRASP-Pro corrected |
|---|---|---|---|---|---|
| 18 | cortex | 1.12 | - | - | - |
| 18 | medulla | 1.07 | - | - | - |
| 18 | aorta | 1.09 | - | - | - |
| 19 | cortex | 1.09 | - | - | - |
| 19 | medulla | 1.06 | - | - | - |
| 19 | aorta | 1.09 | - | - | - |
| 21 | cortex | 1.07 | 0.90 (raw 0.96) | 0.72 (raw 0.77) | 0.60 (raw 0.64) |
| 21 | medulla | 1.04 | 0.89 (raw 0.93) | 0.97 (raw 1.01) | 0.89 (raw 0.92) |
| 21 | aorta | 1.10 | 0.95 (raw 1.05) | 0.64 (raw 0.71) | 0.21 (raw 0.23) |
