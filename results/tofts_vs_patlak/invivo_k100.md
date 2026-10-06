# in vivo (meas_p3_dce, slices 21, k100 = every view in training (no held-out spokes; the val / test kNMSE columns are TRAIN-set numbers here); same views for every method)

rulers: mf_* = NRMSE vs model-free NUFFT ROI curve on its 240-pt grid (affine = raw+affine fit; scale = baseline-subtracted single scale). physical bounds on aorta. *_kNMSE = complex k-space NMSE at held-out spokes (NIK only). CS rows are references, NOT truth; CS held-out blocked (magnitude-only files).

## slice 21
| metric | tofts8 mean±SD (n) | Patlak mean±SD (n) | sub16 mean±SD (n) | free mean±SD (n) | Δ Patlak−tofts8 | Δ sub16−tofts8 | Δ free−tofts8 | GRASP-v2 all spokes | GRASP-Pro all spokes |
|---|---|---|---|---|---|---|---|---|---|
| cortex_peak_ratio | 0.8906 ± 0 (1) | 0.9513 ± 0 (1) | 0.739 ± 0 (1) | 0.7276 ± 0 (1) | +0.06064 | -0.1516 | -0.163 | 0.7881 | 0.648 |
| cortex_washout_ratio | 0.9747 ± 0 (1) | 1.007 ± 0 (1) | 0.8322 ± 0 (1) | 0.8662 ± 0 (1) | +0.03193 | -0.1425 | -0.1085 | 1.053 | 0.9643 |
| medulla_peak_ratio | 0.9374 ± 0 (1) | 0.8289 ± 0 (1) | 0.9041 ± 0 (1) | 0.8888 ± 0 (1) | -0.1085 | -0.03331 | -0.0486 | 1.015 | 0.9392 |
| medulla_washout_ratio | 0.9918 ± 0 (1) | 1.086 ± 0 (1) | 0.8402 ± 0 (1) | 0.9495 ± 0 (1) | +0.09403 | -0.1516 | -0.04232 | 1.072 | 0.9635 |
| aorta_peak_ratio | 1.04 ± 0 (1) | 1.01 ± 0 (1) | 0.593 ± 0 (1) | 0.7707 ± 0 (1) | -0.03002 | -0.4474 | -0.2696 | 0.7702 | 0.2427 |
| aorta_washout_ratio | 1.061 ± 0 (1) | 1.058 ± 0 (1) | 0.9056 ± 0 (1) | 0.8564 ± 0 (1) | -0.003076 | -0.1559 | -0.2051 | 1.012 | 0.7249 |
| mf_aorta_affine | 0.04772 ± 0 (1) | 0.04235 ± 0 (1) | 0.2526 ± 0 (1) | 0.1066 ± 0 (1) | -0.005371 | +0.2049 | +0.05884 | 0.08901 | 0.3692 |
| mf_cortex_affine | 0.02822 ± 0 (1) | 0.1492 ± 0 (1) | 0.05951 ± 0 (1) | 0.04408 ± 0 (1) | +0.121 | +0.03128 | +0.01586 | 0.03944 | 0.09906 |
| mf_medulla_affine | 0.02961 ± 0 (1) | 0.1252 ± 0 (1) | 0.07689 ± 0 (1) | 0.03737 ± 0 (1) | +0.09562 | +0.04728 | +0.007753 | 0.02517 | 0.06704 |
| mf_liver_affine | 0.02485 ± 0 (1) | 0.02814 ± 0 (1) | 0.03265 ± 0 (1) | 0.026 ± 0 (1) | +0.003285 | +0.007796 | +0.00115 | 0.02321 | 0.03736 |
| mf_aorta_scale | 0.062 ± 0 (1) | 0.05627 ± 0 (1) | 0.3301 ± 0 (1) | 0.1373 ± 0 (1) | -0.005737 | +0.2681 | +0.07528 | 0.1242 | 0.4861 |
| mf_cortex_scale | 0.04294 ± 0 (1) | 0.2343 ± 0 (1) | 0.09624 ± 0 (1) | 0.06725 ± 0 (1) | +0.1913 | +0.0533 | +0.02431 | 0.06079 | 0.1525 |
| mf_medulla_scale | 0.04351 ± 0 (1) | 0.1918 ± 0 (1) | 0.1151 ± 0 (1) | 0.05483 ± 0 (1) | +0.1483 | +0.07154 | +0.01132 | 0.03846 | 0.09852 |
| aorta_peak_ratio_vs_mf | 1.04 ± 0 (1) | 1.01 ± 0 (1) | 0.593 ± 0 (1) | 0.7707 ± 0 (1) | -0.03002 | -0.4474 | -0.2696 | 0.7702 | 0.2443 |
| aorta_fwhm_s | 16.9 ± 0 (1) | 16.9 ± 0 (1) | 23.04 ± 0 (1) | 19.97 ± 0 (1) | +0 | +6.144 | +3.072 | 35.33 | 70.66 |
| aorta_ttp_s | 64.73 ± 0 (1) | 64.73 ± 0 (1) | 64.73 ± 0 (1) | 64.73 ± 0 (1) | +0 | +0 | +0 | 64.73 | 247.5 |
| aorta_neg_frac | 0 ± 0 (1) | 0 ± 0 (1) | 0.025 ± 0 (1) | 0.02083 ± 0 (1) | +0 | +0.025 | +0.02083 | 0 | 0.0125 |
| aorta_rise_mono | 1 ± 0 (1) | 1 ± 0 (1) | 0.7 ± 0 (1) | 1 ± 0 (1) | +0 | -0.3 | +0 | 1 | 0.6164 |
| cortex_medulla_late_corr | 0.1018 ± 0 (1) | 0.9927 ± 0 (1) | 0.02282 ± 0 (1) | -0.1318 ± 0 (1) | +0.8908 | -0.079 | -0.2337 | -0.1108 | 0.876 |
| train_kNMSE | 0.2274 ± 0 (1) | 0.2357 ± 0 (1) | 0.0335 ± 0 (1) | nan ± nan (0) | +0.008248 | -0.1939 | +nan | nan | nan |
| val_kNMSE | 0.2269 ± 0 (1) | 0.2347 ± 0 (1) | 0.03334 ± 0 (1) | nan ± nan (0) | +0.007887 | -0.1935 | +nan | nan | nan |
| test_kNMSE | 0.2293 ± 0 (1) | 0.2374 ± 0 (1) | 0.03368 ± 0 (1) | nan ± nan (0) | +0.008121 | -0.1956 | +nan | nan | nan |
| wall_s | 3919 ± 0 (1) | 3871 ± 0 (1) | 3952 ± 0 (1) | 2346 ± 0 (1) | -48.2 | +33.43 | -1573 | nan | nan |
| peak_gpu_mb | 1.248e+04 ± 0 (1) | 1.248e+04 ± 0 (1) | 1.248e+04 ± 0 (1) | 1.249e+04 ± 0 (1) | -0.1255 | +0.2964 | +12.81 | nan | nan |
| params | 5.531e+06 ± 0 (1) | 5.521e+06 ± 0 (1) | 5.558e+06 ± 0 (1) | nan ± nan (0) | -1.025e+04 | +2.68e+04 | +nan | nan | nan |

model-free aorta FWHM (s): 16.89584552369808

## reference peak correction

the 31-spoke model-free reference (6.8 s window) clips the first-pass peak; factor = peak at an 11-spoke window / peak at 31 spokes (`mf_peak_check.py`, per-spoke normalized, approved rois). corrected ratio = ratio vs the 31-spoke reference / factor. state this with every peak comparison.

| slice | roi | clip factor | free corrected peak ratio | patlak corrected peak ratio | sub16 corrected peak ratio | tofts8 corrected peak ratio | GRASP-v2 corrected | GRASP-Pro corrected |
|---|---|---|---|---|---|---|---|---|
| 18 | cortex | 1.12 | - | - | - | - | - | - |
| 18 | medulla | 1.07 | - | - | - | - | - | - |
| 18 | aorta | 1.09 | - | - | - | - | - | - |
| 19 | cortex | 1.09 | - | - | - | - | - | - |
| 19 | medulla | 1.06 | - | - | - | - | - | - |
| 19 | aorta | 1.09 | - | - | - | - | - | - |
| 21 | cortex | 1.07 | 0.68 (raw 0.73) | 0.89 (raw 0.95) | 0.69 (raw 0.74) | 0.83 (raw 0.89) | 0.74 (raw 0.79) | 0.61 (raw 0.65) |
| 21 | medulla | 1.04 | 0.86 (raw 0.89) | 0.80 (raw 0.83) | 0.87 (raw 0.90) | 0.90 (raw 0.94) | 0.98 (raw 1.02) | 0.91 (raw 0.94) |
| 21 | aorta | 1.10 | 0.70 (raw 0.77) | 0.92 (raw 1.01) | 0.54 (raw 0.59) | 0.94 (raw 1.04) | 0.70 (raw 0.77) | 0.22 (raw 0.24) |
