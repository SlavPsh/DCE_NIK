# in vivo (meas_topqmri_p14, slices 21/24/27, k80 = 1368/1708 views (v%10<8), VAL v%10==8 for early stop, TEST v%10==9 untouched; same views for every method)

rulers: mf_* = NRMSE vs model-free NUFFT ROI curve on its 240-pt grid (affine = raw+affine fit; scale = baseline-subtracted single scale). physical bounds on aorta. *_kNMSE = complex k-space NMSE at held-out spokes (NIK only). CS rows are references, NOT truth; CS held-out blocked (magnitude-only files).

## slice 21
| metric | tofts8 mean±SD (n) |  | GRASP-v2 k80 | GRASP-Pro f80match |
|---|---|---|---|
| liver_peak_ratio | 0.9775 ± 0.0068 (3) |  | 1.005 | 0.7193 |
| liver_washout_ratio | 1.046 ± 0.0076 (3) |  | 1.073 | 0.8801 |
| spleen_peak_ratio | 0.9596 ± 0.007 (3) |  | 1.069 | 1.051 |
| spleen_washout_ratio | 1.006 ± 0.0055 (3) |  | 1.128 | 1.067 |
| aorta_peak_ratio | 0.9691 ± 0.016 (3) |  | 0.8454 | 0.792 |
| aorta_washout_ratio | 1.106 ± 0.073 (3) |  | 1.125 | 0.9252 |
| mf_aorta_affine | 0.04743 ± 0.00094 (3) |  | 0.06276 | 0.1551 |
| mf_liver_affine | 0.0403 ± 7e-05 (3) |  | 0.02897 | 0.06528 |
| mf_spleen_affine | 0.03167 ± 0.00088 (3) |  | 0.03147 | 0.05841 |
| mf_static_affine | 0.01979 ± 0.00028 (3) |  | 0.01614 | 0.03601 |
| mf_aorta_scale | 0.07309 ± 0.0016 (3) |  | 0.1004 | 0.246 |
| mf_liver_scale | 0.08056 ± 9.5e-05 (3) |  | 0.05826 | 0.1315 |
| mf_spleen_scale | 0.05488 ± 0.0018 (3) |  | 0.05689 | 0.1023 |
| aorta_peak_ratio_vs_mf | 0.9691 ± 0.016 (3) |  | 0.8454 | 0.792 |
| aorta_fwhm_s | 269.7 ± 23 (3) |  | 345.7 | 234.3 |
| aorta_ttp_s | 52.94 ± 2.2 (3) |  | 53.36 | 62.23 |
| aorta_neg_frac | 0 ± 0 (3) |  | 0 | 0.006579 |
| aorta_rise_mono | 1 ± 0 (3) |  | 1 | 0.8511 |
| cortex_medulla_late_corr | 0.9764 ± 0.0028 (3) |  | 0.9812 | 0.3957 |
| train_kNMSE | 0.2508 ± 0.00028 (3) |  | nan | nan |
| val_kNMSE | 0.2717 ± 0.00045 (3) |  | nan | nan |
| test_kNMSE | 0.2745 ± 0.00081 (3) |  | nan | nan |
| wall_s | 5558 ± 1.4 (3) |  | nan | nan |
| peak_gpu_mb | 1.284e+04 ± 0 (3) |  | nan | nan |
| params | 5.531e+06 ± 0 (3) |  | nan | nan |

model-free aorta FWHM (s): 164.61038961038963

TEST-spoke k-space NMSE per |k| annulus:
| annulus | tofts8 |
|---|---|
| 0.00-0.06 | 2.541e-01 |
| 0.06-0.12 | 5.783e-01 |
| 0.12-0.19 | 7.771e-01 |
| 0.19-0.25 | 8.398e-01 |
| 0.25-0.31 | 8.139e-01 |
| 0.31-0.38 | 8.076e-01 |
| 0.38-0.44 | 8.496e-01 |
| 0.44-0.50 | 8.708e-01 |
| 0.50-0.56 | 8.601e-01 |
| 0.56-0.62 | 8.775e-01 |
| 0.62-0.69 | 9.321e-01 |
| 0.69-0.75 | 9.372e-01 |
| 0.75-0.81 | 9.699e-01 |
| 0.81-0.88 | 9.570e-01 |
| 0.88-0.94 | 1.006e+00 |
| 0.94-1.00 | 1.024e+00 |

## slice 24
| metric | tofts8 mean±SD (n) |  | GRASP-v2 k80 | GRASP-Pro f80match |
|---|---|---|---|
| liver_peak_ratio | 0.9803 ± 0.03 (3) |  | 0.9963 | 0.6594 |
| liver_washout_ratio | 1.038 ± 0.024 (3) |  | 1.071 | 0.7786 |
| spleen_peak_ratio | 0.8901 ± 0.016 (3) |  | 1.006 | 0.5909 |
| spleen_washout_ratio | 0.9727 ± 0.024 (3) |  | 1.082 | 0.6307 |
| aorta_peak_ratio | 0.8871 ± 0.022 (3) |  | 0.8087 | 0.4435 |
| aorta_washout_ratio | 0.9881 ± 0.056 (3) |  | 1.093 | 0.4443 |
| mf_aorta_affine | 0.05116 ± 0.0049 (3) |  | 0.08018 | 0.1783 |
| mf_liver_affine | 0.02884 ± 0.00047 (3) |  | 0.02434 | 0.06781 |
| mf_spleen_affine | 0.03302 ± 0.00071 (3) |  | 0.03766 | 0.0698 |
| mf_static_affine | 0.02319 ± 0.00086 (3) |  | 0.01905 | 0.05442 |
| mf_aorta_scale | 0.07308 ± 0.0069 (3) |  | 0.1182 | 0.2871 |
| mf_liver_scale | 0.05946 ± 0.001 (3) |  | 0.05001 | 0.1391 |
| mf_spleen_scale | 0.05714 ± 0.0015 (3) |  | 0.06732 | 0.1199 |
| aorta_peak_ratio_vs_mf | 0.8871 ± 0.022 (3) |  | 0.8087 | 0.4435 |
| aorta_fwhm_s | 286.2 ± 47 (3) |  | 345.7 | 70.91 |
| aorta_ttp_s | 49.56 ± 0 (3) |  | 69.82 | 59.69 |
| aorta_neg_frac | 0 ± 0 (3) |  | 0 | 0.009868 |
| aorta_rise_mono | 1 ± 0 (3) |  | 1 | 0.8 |
| cortex_medulla_late_corr | 0.9895 ± 0.0031 (3) |  | 0.9912 | -0.02402 |
| train_kNMSE | 0.2388 ± 0.00061 (3) |  | nan | nan |
| val_kNMSE | 0.258 ± 0.00053 (3) |  | nan | nan |
| test_kNMSE | 0.2633 ± 0.00041 (3) |  | nan | nan |
| wall_s | 5554 ± 3.7 (3) |  | nan | nan |
| peak_gpu_mb | 1.284e+04 ± 0 (3) |  | nan | nan |
| params | 5.531e+06 ± 0 (3) |  | nan | nan |

model-free aorta FWHM (s): 162.07792207792212

TEST-spoke k-space NMSE per |k| annulus:
| annulus | tofts8 |
|---|---|
| 0.00-0.06 | 2.421e-01 |
| 0.06-0.12 | 6.107e-01 |
| 0.12-0.19 | 8.356e-01 |
| 0.19-0.25 | 8.119e-01 |
| 0.25-0.31 | 8.712e-01 |
| 0.31-0.38 | 8.283e-01 |
| 0.38-0.44 | 1.005e+00 |
| 0.44-0.50 | 8.229e-01 |
| 0.50-0.56 | 8.766e-01 |
| 0.56-0.62 | 8.905e-01 |
| 0.62-0.69 | 7.887e-01 |
| 0.69-0.75 | 8.506e-01 |
| 0.75-0.81 | 9.129e-01 |
| 0.81-0.88 | 8.727e-01 |
| 0.88-0.94 | 8.046e-01 |
| 0.94-1.00 | 7.919e-01 |

## slice 27
| metric | tofts8 mean±SD (n) |  | GRASP-v2 k80 | GRASP-Pro f80match |
|---|---|---|---|
| liver_peak_ratio | 0.9866 ± 0.02 (3) |  | 0.9778 | 0.7799 |
| liver_washout_ratio | 1.039 ± 0.0085 (3) |  | 1.047 | 0.8155 |
| spleen_peak_ratio | 0.8919 ± 0.023 (3) |  | 1.005 | 0.5465 |
| spleen_washout_ratio | 0.962 ± 0.021 (3) |  | 1.08 | 0.7082 |
| aorta_peak_ratio | 0.8438 ± 0.0052 (3) |  | 0.8151 | 0.4317 |
| aorta_washout_ratio | 1.015 ± 0.056 (3) |  | 1.06 | 0.475 |
| mf_aorta_affine | 0.05527 ± 0.0038 (3) |  | 0.0594 | 0.1828 |
| mf_liver_affine | 0.02438 ± 8.4e-05 (3) |  | 0.02041 | 0.05849 |
| mf_spleen_affine | 0.03261 ± 0.00062 (3) |  | 0.03264 | 0.09962 |
| mf_static_affine | 0.02114 ± 0.00055 (3) |  | 0.02001 | 0.04029 |
| mf_aorta_scale | 0.07692 ± 0.0051 (3) |  | 0.08333 | 0.2832 |
| mf_liver_scale | 0.05174 ± 0.0001 (3) |  | 0.04324 | 0.1236 |
| mf_spleen_scale | 0.05757 ± 0.0012 (3) |  | 0.06003 | 0.1746 |
| aorta_peak_ratio_vs_mf | 0.8438 ± 0.0052 (3) |  | 0.8151 | 0.4317 |
| aorta_fwhm_s | 330.5 ± 20 (3) |  | 345.7 | 46.85 |
| aorta_ttp_s | 55.47 ± 10 (3) |  | 68.56 | 59.69 |
| aorta_neg_frac | 0 ± 0 (3) |  | 0 | 0.01316 |
| aorta_rise_mono | 1 ± 0 (3) |  | 1 | 0.8 |
| cortex_medulla_late_corr | 0.9916 ± 0.00087 (3) |  | 0.9975 | 0.2277 |
| train_kNMSE | 0.2154 ± 0.00067 (3) |  | nan | nan |
| val_kNMSE | 0.2227 ± 0.00099 (3) |  | nan | nan |
| test_kNMSE | 0.2262 ± 0.0007 (3) |  | nan | nan |
| wall_s | 5555 ± 0.48 (3) |  | nan | nan |
| peak_gpu_mb | 1.284e+04 ± 0 (3) |  | nan | nan |
| params | 5.531e+06 ± 0 (3) |  | nan | nan |

model-free aorta FWHM (s): 196.26623376623377

TEST-spoke k-space NMSE per |k| annulus:
| annulus | tofts8 |
|---|---|
| 0.00-0.06 | 2.174e-01 |
| 0.06-0.12 | 5.125e-01 |
| 0.12-0.19 | 5.923e-01 |
| 0.19-0.25 | 6.721e-01 |
| 0.25-0.31 | 7.228e-01 |
| 0.31-0.38 | 7.890e-01 |
| 0.38-0.44 | 8.561e-01 |
| 0.44-0.50 | 8.905e-01 |
| 0.50-0.56 | 9.331e-01 |
| 0.56-0.62 | 9.805e-01 |
| 0.62-0.69 | 1.013e+00 |
| 0.69-0.75 | 1.026e+00 |
| 0.75-0.81 | 1.064e+00 |
| 0.81-0.88 | 1.073e+00 |
| 0.88-0.94 | 1.108e+00 |
| 0.94-1.00 | 1.123e+00 |
