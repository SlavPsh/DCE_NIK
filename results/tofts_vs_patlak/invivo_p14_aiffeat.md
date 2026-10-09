# in vivo (meas_topqmri_p14, slices 21/24/27, k100 = every view in training (no held-out spokes; the val / test kNMSE columns are TRAIN-set numbers here); same views for every method)

rulers: mf_* = NRMSE vs model-free NUFFT ROI curve on its 240-pt grid (affine = raw+affine fit; scale = baseline-subtracted single scale). physical bounds on aorta. *_kNMSE = complex k-space NMSE at held-out spokes (NIK only). CS rows are references, NOT truth; CS held-out blocked (magnitude-only files).

## slice 21
| metric | sub16 mean±SD (n) |  | GRASP-v2 all spokes |
|---|---|---|---|
| liver_peak_ratio | 0.8765 ± 0 (1) |  | 1.023 | nan |
| liver_washout_ratio | 0.9775 ± 0 (1) |  | 1.094 | nan |
| spleen_peak_ratio | 0.764 ± 0 (1) |  | 1.073 | nan |
| spleen_washout_ratio | 0.8828 ± 0 (1) |  | 1.123 | nan |
| aorta_peak_ratio | 0.6986 ± 0 (1) |  | 0.8699 | nan |
| aorta_washout_ratio | 0.8711 ± 0 (1) |  | 1.132 | nan |
| mf_aorta_affine | 0.07218 ± 0 (1) |  | 0.06055 | nan |
| mf_liver_affine | 0.0516 ± 0 (1) |  | 0.02761 | nan |
| mf_spleen_affine | 0.08125 ± 0 (1) |  | 0.02994 | nan |
| mf_static_affine | 0.02986 ± 0 (1) |  | 0.01571 | nan |
| mf_aorta_scale | 0.1101 ± 0 (1) |  | 0.09688 | nan |
| mf_liver_scale | 0.1057 ± 0 (1) |  | 0.05506 | nan |
| mf_spleen_scale | 0.1393 ± 0 (1) |  | 0.05423 | nan |
| aorta_peak_ratio_vs_mf | 0.6986 ± 0 (1) |  | 0.8699 | nan |
| aorta_fwhm_s | 303.9 ± 0 (1) |  | 345.7 | nan |
| aorta_ttp_s | 49.56 ± 0 (1) |  | 53.36 | nan |
| aorta_neg_frac | 0.01974 ± 0 (1) |  | 0 | nan |
| aorta_rise_mono | 0.8919 ± 0 (1) |  | 1 | nan |
| cortex_medulla_late_corr | 0.8951 ± 0 (1) |  | 0.9778 | nan |
| train_kNMSE | nan ± nan (0) |  | nan | nan |
| val_kNMSE | nan ± nan (0) |  | nan | nan |
| test_kNMSE | nan ± nan (0) |  | nan | nan |
| wall_s | 5785 ± 0 (1) |  | nan | nan |
| peak_gpu_mb | 1.287e+04 ± 0 (1) |  | nan | nan |
| params | nan ± nan (0) |  | nan | nan |

model-free aorta FWHM (s): 164.61038961038963

## slice 24
| metric | sub16 mean±SD (n) |  | GRASP-v2 all spokes |
|---|---|---|---|
| liver_peak_ratio | 0.8189 ± 0 (1) |  | 1.014 | nan |
| liver_washout_ratio | 0.9598 ± 0 (1) |  | 1.089 | nan |
| spleen_peak_ratio | 0.7203 ± 0 (1) |  | 1.013 | nan |
| spleen_washout_ratio | 0.8922 ± 0 (1) |  | 1.088 | nan |
| aorta_peak_ratio | 0.6426 ± 0 (1) |  | 0.8305 | nan |
| aorta_washout_ratio | 0.8794 ± 0 (1) |  | 1.111 | nan |
| mf_aorta_affine | 0.08906 ± 0 (1) |  | 0.07582 | nan |
| mf_liver_affine | 0.04749 ± 0 (1) |  | 0.0234 | nan |
| mf_spleen_affine | 0.08002 ± 0 (1) |  | 0.03559 | nan |
| mf_static_affine | 0.02935 ± 0 (1) |  | 0.01865 | nan |
| mf_aorta_scale | 0.1273 ± 0 (1) |  | 0.1119 | nan |
| mf_liver_scale | 0.1026 ± 0 (1) |  | 0.04834 | nan |
| mf_spleen_scale | 0.14 ± 0 (1) |  | 0.06395 | nan |
| aorta_peak_ratio_vs_mf | 0.6426 ± 0 (1) |  | 0.8305 | nan |
| aorta_fwhm_s | 344.4 ± 0 (1) |  | 345.7 | nan |
| aorta_ttp_s | 73.62 ± 0 (1) |  | 68.56 | nan |
| aorta_neg_frac | 0 ± 0 (1) |  | 0 | nan |
| aorta_rise_mono | 0.875 ± 0 (1) |  | 1 | nan |
| cortex_medulla_late_corr | 0.7942 ± 0 (1) |  | 0.9872 | nan |
| train_kNMSE | nan ± nan (0) |  | nan | nan |
| val_kNMSE | nan ± nan (0) |  | nan | nan |
| test_kNMSE | nan ± nan (0) |  | nan | nan |
| wall_s | 5798 ± 0 (1) |  | nan | nan |
| peak_gpu_mb | 1.287e+04 ± 0 (1) |  | nan | nan |
| params | nan ± nan (0) |  | nan | nan |

model-free aorta FWHM (s): 162.07792207792212

## slice 27
| metric | sub16 mean±SD (n) |  | GRASP-v2 all spokes |
|---|---|---|---|
| liver_peak_ratio | 0.8441 ± 0 (1) |  | 1.001 | nan |
| liver_washout_ratio | 0.971 ± 0 (1) |  | 1.073 | nan |
| spleen_peak_ratio | 0.7611 ± 0 (1) |  | 1.007 | nan |
| spleen_washout_ratio | 0.9029 ± 0 (1) |  | 1.083 | nan |
| aorta_peak_ratio | 0.6283 ± 0 (1) |  | 0.8483 | nan |
| aorta_washout_ratio | 0.8071 ± 0 (1) |  | 1.084 | nan |
| mf_aorta_affine | 0.09496 ± 0 (1) |  | 0.05616 | nan |
| mf_liver_affine | 0.04689 ± 0 (1) |  | 0.02002 | nan |
| mf_spleen_affine | 0.07935 ± 0 (1) |  | 0.03153 | nan |
| mf_static_affine | 0.02485 ± 0 (1) |  | 0.01976 | nan |
| mf_aorta_scale | 0.1406 ± 0 (1) |  | 0.07989 | nan |
| mf_liver_scale | 0.1008 ± 0 (1) |  | 0.04287 | nan |
| mf_spleen_scale | 0.1392 ± 0 (1) |  | 0.05792 | nan |
| aorta_peak_ratio_vs_mf | 0.6283 ± 0 (1) |  | 0.8483 | nan |
| aorta_fwhm_s | 301.4 ± 0 (1) |  | 345.7 | nan |
| aorta_ttp_s | 52.1 ± 0 (1) |  | 53.36 | nan |
| aorta_neg_frac | 0.03947 ± 0 (1) |  | 0 | nan |
| aorta_rise_mono | 1 ± 0 (1) |  | 1 | nan |
| cortex_medulla_late_corr | 0.9093 ± 0 (1) |  | 0.9933 | nan |
| train_kNMSE | nan ± nan (0) |  | nan | nan |
| val_kNMSE | nan ± nan (0) |  | nan | nan |
| test_kNMSE | nan ± nan (0) |  | nan | nan |
| wall_s | 5774 ± 0 (1) |  | nan | nan |
| peak_gpu_mb | 1.287e+04 ± 0 (1) |  | nan | nan |
| params | nan ± nan (0) |  | nan | nan |

model-free aorta FWHM (s): 196.26623376623377

## reference peak correction

the 31-spoke model-free reference (6.8 s window) clips the first-pass peak; factor = peak at an 11-spoke window / peak at 31 spokes (`mf_peak_check.py`, per-spoke normalized, approved rois). corrected ratio = ratio vs the 31-spoke reference / factor. state this with every peak comparison.

| slice | roi | clip factor | sub16 corrected peak ratio | GRASP-v2 corrected | GRASP-Pro corrected |
|---|---|---|---|---|---|
| 21 | liver | 1.06 | 0.82 (raw 0.88) | 0.96 (raw 1.02) | - |
| 21 | spleen | 1.07 | 0.71 (raw 0.76) | 1.00 (raw 1.07) | - |
| 21 | aorta | 1.09 | 0.64 (raw 0.70) | 0.80 (raw 0.87) | - |
| 24 | liver | 1.09 | 0.75 (raw 0.82) | 0.93 (raw 1.01) | - |
| 24 | spleen | 1.08 | 0.66 (raw 0.72) | 0.93 (raw 1.01) | - |
| 24 | aorta | 1.02 | 0.63 (raw 0.64) | 0.81 (raw 0.83) | - |
| 27 | liver | 1.13 | 0.75 (raw 0.84) | 0.89 (raw 1.00) | - |
| 27 | spleen | 1.09 | 0.70 (raw 0.76) | 0.93 (raw 1.01) | - |
| 27 | aorta | 1.07 | 0.59 (raw 0.63) | 0.79 (raw 0.85) | - |
