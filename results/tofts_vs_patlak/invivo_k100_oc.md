# in vivo (meas_p3_dce, slices 21, k100 = every view in training (no held-out spokes; the val / test kNMSE columns are TRAIN-set numbers here); same views for every method)

rulers: mf_* = NRMSE vs model-free NUFFT ROI curve on its 240-pt grid (affine = raw+affine fit; scale = baseline-subtracted single scale). physical bounds on aorta. *_kNMSE = complex k-space NMSE at held-out spokes (NIK only). CS rows are references, NOT truth; CS held-out blocked (magnitude-only files).

## slice 21
| metric | tofts8 mean±SD (n) |  | GRASP-v2 all spokes | GRASP-Pro all spokes |
|---|---|---|---|
| cortex_peak_ratio | 0.8718 ± 0 (1) |  | 0.7881 | 0.648 |
| cortex_washout_ratio | 0.9479 ± 0 (1) |  | 1.053 | 0.9643 |
| medulla_peak_ratio | 0.9012 ± 0 (1) |  | 1.015 | 0.9392 |
| medulla_washout_ratio | 0.9524 ± 0 (1) |  | 1.072 | 0.9635 |
| aorta_peak_ratio | 1.054 ± 0 (1) |  | 0.7702 | 0.2427 |
| aorta_washout_ratio | 1.063 ± 0 (1) |  | 1.012 | 0.7249 |
| mf_aorta_affine | 0.049 ± 0 (1) |  | 0.08901 | 0.3692 |
| mf_cortex_affine | 0.02814 ± 0 (1) |  | 0.03944 | 0.09906 |
| mf_medulla_affine | 0.03349 ± 0 (1) |  | 0.02517 | 0.06704 |
| mf_liver_affine | 0.0233 ± 0 (1) |  | 0.02321 | 0.03736 |
| mf_aorta_scale | 0.06321 ± 0 (1) |  | 0.1242 | 0.4861 |
| mf_cortex_scale | 0.04278 ± 0 (1) |  | 0.06079 | 0.1525 |
| mf_medulla_scale | 0.04917 ± 0 (1) |  | 0.03846 | 0.09852 |
| aorta_peak_ratio_vs_mf | 1.054 ± 0 (1) |  | 0.7702 | 0.2443 |
| aorta_fwhm_s | 16.9 ± 0 (1) |  | 35.33 | 70.66 |
| aorta_ttp_s | 64.73 ± 0 (1) |  | 64.73 | 247.5 |
| aorta_neg_frac | 0 ± 0 (1) |  | 0 | 0.0125 |
| aorta_rise_mono | 1 ± 0 (1) |  | 1 | 0.6164 |
| cortex_medulla_late_corr | 0.1639 ± 0 (1) |  | -0.1108 | 0.876 |
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
| 0.31-0.38 | 4.440e-01 |
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
| 18 | cortex | 1.12 | - | - | - |
| 18 | medulla | 1.07 | - | - | - |
| 18 | aorta | 1.09 | - | - | - |
| 19 | cortex | 1.09 | - | - | - |
| 19 | medulla | 1.06 | - | - | - |
| 19 | aorta | 1.09 | - | - | - |
| 21 | cortex | 1.07 | 0.81 (raw 0.87) | 0.74 (raw 0.79) | 0.61 (raw 0.65) |
| 21 | medulla | 1.04 | 0.87 (raw 0.90) | 0.98 (raw 1.02) | 0.91 (raw 0.94) |
| 21 | aorta | 1.10 | 0.95 (raw 1.05) | 0.70 (raw 0.77) | 0.22 (raw 0.24) |
