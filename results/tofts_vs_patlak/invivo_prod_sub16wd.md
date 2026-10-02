# in vivo (meas_p3_dce, slices 18/19/21, k80 = 1368/1708 views (v%10<8), VAL v%10==8 for early stop, TEST v%10==9 untouched; same views for every method)

rulers: mf_* = NRMSE vs model-free NUFFT ROI curve on its 240-pt grid (affine = raw+affine fit; scale = baseline-subtracted single scale). physical bounds on aorta. *_kNMSE = complex k-space NMSE at held-out spokes (NIK only). CS rows are references, NOT truth; CS held-out blocked (magnitude-only files).

## slice 18
| metric | sub16 mean±SD (n) |  | GRASP-v2 k80 | GRASP-Pro f80match |
|---|---|---|---|
| cortex_peak_ratio | 0.7885 ± 0 (1) |  | 0.7884 | 0.9924 |
| cortex_washout_ratio | 0.9231 ± 0 (1) |  | 1.046 | 1.012 |
| medulla_peak_ratio | 0.9113 ± 0 (1) |  | 1.025 | 0.958 |
| medulla_washout_ratio | 0.8464 ± 0 (1) |  | 1.046 | 0.933 |
| aorta_peak_ratio | 0.5495 ± 0 (1) |  | 0.7434 | 0.6819 |
| aorta_washout_ratio | 0.8985 ± 0 (1) |  | 1.008 | 0.8691 |
| mf_aorta_affine | 0.2177 ± 0 (1) |  | 0.09051 | 0.2972 |
| mf_cortex_affine | 0.06896 ± 0 (1) |  | 0.04518 | 0.05357 |
| mf_medulla_affine | 0.06053 ± 0 (1) |  | 0.0288 | 0.05778 |
| mf_liver_affine | 0.02989 ± 0 (1) |  | 0.02232 | 0.03045 |
| mf_aorta_scale | 0.2763 ± 0 (1) |  | 0.1248 | 0.3905 |
| mf_cortex_scale | 0.1061 ± 0 (1) |  | 0.07049 | 0.08244 |
| mf_medulla_scale | 0.09009 ± 0 (1) |  | 0.04372 | 0.08555 |
| aorta_peak_ratio_vs_mf | 0.5495 ± 0 (1) |  | 0.7434 | 0.6819 |
| aorta_fwhm_s | 36.86 ± 0 (1) |  | 47.62 | 18.43 |
| aorta_ttp_s | 64.73 ± 0 (1) |  | 63.19 | 69.34 |
| aorta_neg_frac | 0.0125 ± 0 (1) |  | 0 | 0 |
| aorta_rise_mono | 0.725 ± 0 (1) |  | 1 | 0.8605 |
| cortex_medulla_late_corr | 0.1039 ± 0 (1) |  | -0.01639 | 0.4411 |
| train_kNMSE | 0.02892 ± 0 (1) |  | nan | nan |
| val_kNMSE | 0.03457 ± 0 (1) |  | nan | nan |
| test_kNMSE | 0.03523 ± 0 (1) |  | nan | nan |
| wall_s | 3979 ± 0 (1) |  | nan | nan |
| peak_gpu_mb | 1.246e+04 ± 0 (1) |  | nan | nan |
| params | 5.558e+06 ± 0 (1) |  | nan | nan |

model-free aorta FWHM (s): 15.359859566998232

TEST-spoke k-space NMSE per |k| annulus:
| annulus | sub16 |
|---|---|
| 0.00-0.06 | 3.102e-02 |
| 0.06-0.12 | 5.366e-02 |
| 0.12-0.19 | 8.345e-02 |
| 0.19-0.25 | 1.081e-01 |
| 0.25-0.31 | 1.293e-01 |
| 0.31-0.38 | 1.463e-01 |
| 0.38-0.44 | 1.909e-01 |
| 0.44-0.50 | 2.341e-01 |
| 0.50-0.56 | 2.700e-01 |
| 0.56-0.62 | 3.038e-01 |
| 0.62-0.69 | 3.316e-01 |
| 0.69-0.75 | 3.855e-01 |
| 0.75-0.81 | 4.602e-01 |
| 0.81-0.88 | 5.428e-01 |
| 0.88-0.94 | 6.099e-01 |
| 0.94-1.00 | 6.745e-01 |

## slice 19
| metric | sub16 mean±SD (n) |  | GRASP-v2 k80 | GRASP-Pro f80match |
|---|---|---|---|
| cortex_peak_ratio | 0.8001 ± 0 (1) |  | 0.7471 | 0.853 |
| cortex_washout_ratio | 0.9265 ± 0 (1) |  | 1.046 | 1.056 |
| medulla_peak_ratio | 0.8923 ± 0 (1) |  | 1.015 | 1.062 |
| medulla_washout_ratio | 0.8498 ± 0 (1) |  | 1.038 | 0.9018 |
| aorta_peak_ratio | 0.6443 ± 0 (1) |  | 0.6472 | 0.3791 |
| aorta_washout_ratio | 0.7684 ± 0 (1) |  | 1.003 | 1.042 |
| mf_aorta_affine | 0.2874 ± 0 (1) |  | 0.1323 | 0.3489 |
| mf_cortex_affine | 0.06167 ± 0 (1) |  | 0.0576 | 0.06224 |
| mf_medulla_affine | 0.056 ± 0 (1) |  | 0.0423 | 0.07265 |
| mf_liver_affine | 0.02911 ± 0 (1) |  | 0.02405 | 0.03211 |
| mf_aorta_scale | 0.3871 ± 0 (1) |  | 0.1761 | 0.4434 |
| mf_cortex_scale | 0.09378 ± 0 (1) |  | 0.08963 | 0.09467 |
| mf_medulla_scale | 0.0823 ± 0 (1) |  | 0.0619 | 0.1055 |
| aorta_peak_ratio_vs_mf | 0.6443 ± 0 (1) |  | 0.6472 | 0.3791 |
| aorta_fwhm_s | 12.29 ± 0 (1) |  | 92.16 | 56.83 |
| aorta_ttp_s | 67.8 ± 0 (1) |  | 64.73 | 69.34 |
| aorta_neg_frac | 0.04583 ± 0 (1) |  | 0 | 0.004167 |
| aorta_rise_mono | 0.7619 ± 0 (1) |  | 1 | 0.8837 |
| cortex_medulla_late_corr | 0.5527 ± 0 (1) |  | 0.05177 | 0.9518 |
| train_kNMSE | 0.06359 ± 0 (1) |  | nan | nan |
| val_kNMSE | 0.07038 ± 0 (1) |  | nan | nan |
| test_kNMSE | 0.07073 ± 0 (1) |  | nan | nan |
| wall_s | 3972 ± 0 (1) |  | nan | nan |
| peak_gpu_mb | 1.246e+04 ± 0 (1) |  | nan | nan |
| params | 5.558e+06 ± 0 (1) |  | nan | nan |

model-free aorta FWHM (s): 15.359859566998232

TEST-spoke k-space NMSE per |k| annulus:
| annulus | sub16 |
|---|---|
| 0.00-0.06 | 6.960e-02 |
| 0.06-0.12 | 5.260e-02 |
| 0.12-0.19 | 8.887e-02 |
| 0.19-0.25 | 8.698e-02 |
| 0.25-0.31 | 7.869e-02 |
| 0.31-0.38 | 7.599e-02 |
| 0.38-0.44 | 8.304e-02 |
| 0.44-0.50 | 1.125e-01 |
| 0.50-0.56 | 1.437e-01 |
| 0.56-0.62 | 2.002e-01 |
| 0.62-0.69 | 2.426e-01 |
| 0.69-0.75 | 2.987e-01 |
| 0.75-0.81 | 3.450e-01 |
| 0.81-0.88 | 3.746e-01 |
| 0.88-0.94 | 4.163e-01 |
| 0.94-1.00 | 4.407e-01 |

## slice 21
| metric | sub16 mean±SD (n) |  | GRASP-v2 k80 | GRASP-Pro f80match |
|---|---|---|---|
| cortex_peak_ratio | 0.7627 ± 0 (1) |  | 0.7667 | 0.638 |
| cortex_washout_ratio | 0.9538 ± 0 (1) |  | 1.048 | 0.9215 |
| medulla_peak_ratio | 0.9246 ± 0 (1) |  | 1.008 | 0.9193 |
| medulla_washout_ratio | 0.8831 ± 0 (1) |  | 1.069 | 0.897 |
| aorta_peak_ratio | 0.6318 ± 0 (1) |  | 0.7097 | 0.2285 |
| aorta_washout_ratio | 1.051 ± 0 (1) |  | 0.9795 | 0.7434 |
| mf_aorta_affine | 0.2406 ± 0 (1) |  | 0.1013 | 0.384 |
| mf_cortex_affine | 0.06892 ± 0 (1) |  | 0.04423 | 0.1056 |
| mf_medulla_affine | 0.07852 ± 0 (1) |  | 0.02758 | 0.07839 |
| mf_liver_affine | 0.03435 ± 0 (1) |  | 0.02368 | 0.03818 |
| mf_aorta_scale | 0.3075 ± 0 (1) |  | 0.1419 | 0.5084 |
| mf_cortex_scale | 0.1049 ± 0 (1) |  | 0.06828 | 0.1632 |
| mf_medulla_scale | 0.1155 ± 0 (1) |  | 0.04174 | 0.1151 |
| aorta_peak_ratio_vs_mf | 0.6318 ± 0 (1) |  | 0.7097 | 0.2285 |
| aorta_fwhm_s | 44.54 ± 0 (1) |  | 52.22 | 98.3 |
| aorta_ttp_s | 69.34 ± 0 (1) |  | 64.73 | 207.6 |
| aorta_neg_frac | 0.008333 ± 0 (1) |  | 0 | 0.02083 |
| aorta_rise_mono | 0.7442 ± 0 (1) |  | 1 | 0.6466 |
| cortex_medulla_late_corr | 0.17 ± 0 (1) |  | -0.1166 | 0.9698 |
| train_kNMSE | 0.03455 ± 0 (1) |  | nan | nan |
| val_kNMSE | 0.04205 ± 0 (1) |  | nan | nan |
| test_kNMSE | 0.04177 ± 0 (1) |  | nan | nan |
| wall_s | 8048 ± 0 (1) |  | nan | nan |
| peak_gpu_mb | 1.231e+04 ± 0 (1) |  | nan | nan |
| params | 5.558e+06 ± 0 (1) |  | nan | nan |

model-free aorta FWHM (s): 16.89584552369808

TEST-spoke k-space NMSE per |k| annulus:
| annulus | sub16 |
|---|---|
| 0.00-0.06 | 3.631e-02 |
| 0.06-0.12 | 5.673e-02 |
| 0.12-0.19 | 1.073e-01 |
| 0.19-0.25 | 1.752e-01 |
| 0.25-0.31 | 2.369e-01 |
| 0.31-0.38 | 2.993e-01 |
| 0.38-0.44 | 3.715e-01 |
| 0.44-0.50 | 4.793e-01 |
| 0.50-0.56 | 6.184e-01 |
| 0.56-0.62 | 7.634e-01 |
| 0.62-0.69 | 8.710e-01 |
| 0.69-0.75 | 1.009e+00 |
| 0.75-0.81 | 1.081e+00 |
| 0.81-0.88 | 1.161e+00 |
| 0.88-0.94 | 1.209e+00 |
| 0.94-1.00 | 1.242e+00 |

## reference peak correction

the 31-spoke model-free reference (6.8 s window) clips the first-pass peak; factor = peak at an 11-spoke window / peak at 31 spokes (`mf_peak_check.py`, per-spoke normalized, approved rois). corrected ratio = ratio vs the 31-spoke reference / factor. state this with every peak comparison.

| slice | roi | clip factor | sub16 corrected peak ratio | GRASP-v2 corrected | GRASP-Pro corrected |
|---|---|---|---|---|---|
| 18 | cortex | 1.12 | 0.70 (raw 0.79) | 0.70 (raw 0.79) | 0.88 (raw 0.99) |
| 18 | medulla | 1.07 | 0.85 (raw 0.91) | 0.96 (raw 1.02) | 0.90 (raw 0.96) |
| 18 | aorta | 1.09 | 0.50 (raw 0.55) | 0.68 (raw 0.74) | 0.62 (raw 0.68) |
| 19 | cortex | 1.09 | 0.73 (raw 0.80) | 0.68 (raw 0.75) | 0.78 (raw 0.85) |
| 19 | medulla | 1.06 | 0.84 (raw 0.89) | 0.96 (raw 1.02) | 1.00 (raw 1.06) |
| 19 | aorta | 1.09 | 0.59 (raw 0.64) | 0.59 (raw 0.65) | 0.35 (raw 0.38) |
| 21 | cortex | 1.07 | 0.71 (raw 0.76) | 0.72 (raw 0.77) | 0.60 (raw 0.64) |
| 21 | medulla | 1.04 | 0.89 (raw 0.92) | 0.97 (raw 1.01) | 0.89 (raw 0.92) |
| 21 | aorta | 1.10 | 0.57 (raw 0.63) | 0.64 (raw 0.71) | 0.21 (raw 0.23) |
