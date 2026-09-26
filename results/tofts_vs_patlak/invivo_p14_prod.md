# in vivo (meas_p3_dce, slices 21/24/27, k80 = 1368/1708 views (v%10<8), VAL v%10==8 for early stop, TEST v%10==9 untouched; same views for every method)

rulers: mf_* = NRMSE vs model-free NUFFT ROI curve on its 240-pt grid (affine = raw+affine fit; scale = baseline-subtracted single scale). physical bounds on aorta. *_kNMSE = complex k-space NMSE at held-out spokes (NIK only). CS rows are references, NOT truth; CS held-out blocked (magnitude-only files).

## slice 21
| metric | tofts8 mean±SD (n) | Patlak mean±SD (n) | sub16 mean±SD (n) | free mean±SD (n) | Δ Patlak−tofts8 | Δ sub16−tofts8 | Δ free−tofts8 | GRASP-v2 k80 | GRASP-Pro f80match |
|---|---|---|---|---|---|---|---|---|---|
| liver_peak_ratio | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | 1.005 | 0.7193 |
| liver_washout_ratio | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | 1.073 | 0.8801 |
| spleen_peak_ratio | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | 1.069 | 1.051 |
| spleen_washout_ratio | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | 1.128 | 1.067 |
| aorta_peak_ratio | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | 0.8454 | 0.792 |
| aorta_washout_ratio | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | 1.125 | 0.9252 |
| mf_aorta_affine | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | 0.06276 | 0.1551 |
| mf_liver_affine | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | 0.02897 | 0.06528 |
| mf_spleen_affine | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | 0.03147 | 0.05841 |
| mf_static_affine | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | 0.01614 | 0.03601 |
| mf_aorta_scale | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | 0.1004 | 0.246 |
| mf_liver_scale | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | 0.05826 | 0.1315 |
| mf_spleen_scale | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | 0.05689 | 0.1023 |
| aorta_peak_ratio_vs_mf | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | 0.8454 | 0.792 |
| aorta_fwhm_s | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | 345.7 | 234.3 |
| aorta_ttp_s | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | 53.36 | 62.23 |
| aorta_neg_frac | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | 0 | 0.006579 |
| aorta_rise_mono | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | 1 | 0.8511 |
| cortex_medulla_late_corr | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | 0.9812 | 0.3957 |
| train_kNMSE | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | nan | nan |
| val_kNMSE | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | nan | nan |
| test_kNMSE | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | nan | nan |
| wall_s | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | nan | nan |
| peak_gpu_mb | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | nan | nan |
| params | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | nan | nan |

model-free aorta FWHM (s): 164.61038961038963

## slice 24
| metric | tofts8 mean±SD (n) | Patlak mean±SD (n) | sub16 mean±SD (n) | free mean±SD (n) | Δ Patlak−tofts8 | Δ sub16−tofts8 | Δ free−tofts8 | GRASP-v2 k80 | GRASP-Pro f80match |
|---|---|---|---|---|---|---|---|---|---|
| liver_peak_ratio | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | 0.9963 | 0.6594 |
| liver_washout_ratio | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | 1.071 | 0.7786 |
| spleen_peak_ratio | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | 1.006 | 0.5909 |
| spleen_washout_ratio | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | 1.082 | 0.6307 |
| aorta_peak_ratio | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | 0.8087 | 0.4435 |
| aorta_washout_ratio | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | 1.093 | 0.4443 |
| mf_aorta_affine | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | 0.08018 | 0.1783 |
| mf_liver_affine | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | 0.02434 | 0.06781 |
| mf_spleen_affine | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | 0.03766 | 0.0698 |
| mf_static_affine | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | 0.01905 | 0.05442 |
| mf_aorta_scale | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | 0.1182 | 0.2871 |
| mf_liver_scale | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | 0.05001 | 0.1391 |
| mf_spleen_scale | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | 0.06732 | 0.1199 |
| aorta_peak_ratio_vs_mf | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | 0.8087 | 0.4435 |
| aorta_fwhm_s | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | 345.7 | 70.91 |
| aorta_ttp_s | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | 69.82 | 59.69 |
| aorta_neg_frac | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | 0 | 0.009868 |
| aorta_rise_mono | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | 1 | 0.8 |
| cortex_medulla_late_corr | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | 0.9912 | -0.02402 |
| train_kNMSE | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | nan | nan |
| val_kNMSE | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | nan | nan |
| test_kNMSE | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | nan | nan |
| wall_s | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | nan | nan |
| peak_gpu_mb | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | nan | nan |
| params | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | nan | nan |

model-free aorta FWHM (s): 162.07792207792212

## slice 27
| metric | tofts8 mean±SD (n) | Patlak mean±SD (n) | sub16 mean±SD (n) | free mean±SD (n) | Δ Patlak−tofts8 | Δ sub16−tofts8 | Δ free−tofts8 | GRASP-v2 k80 | GRASP-Pro f80match |
|---|---|---|---|---|---|---|---|---|---|
| liver_peak_ratio | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | 0.9778 | 0.7799 |
| liver_washout_ratio | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | 1.047 | 0.8155 |
| spleen_peak_ratio | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | 1.005 | 0.5465 |
| spleen_washout_ratio | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | 1.08 | 0.7082 |
| aorta_peak_ratio | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | 0.8151 | 0.4317 |
| aorta_washout_ratio | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | 1.06 | 0.475 |
| mf_aorta_affine | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | 0.0594 | 0.1828 |
| mf_liver_affine | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | 0.02041 | 0.05849 |
| mf_spleen_affine | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | 0.03264 | 0.09962 |
| mf_static_affine | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | 0.02001 | 0.04029 |
| mf_aorta_scale | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | 0.08333 | 0.2832 |
| mf_liver_scale | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | 0.04324 | 0.1236 |
| mf_spleen_scale | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | 0.06003 | 0.1746 |
| aorta_peak_ratio_vs_mf | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | 0.8151 | 0.4317 |
| aorta_fwhm_s | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | 345.7 | 46.85 |
| aorta_ttp_s | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | 68.56 | 59.69 |
| aorta_neg_frac | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | 0 | 0.01316 |
| aorta_rise_mono | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | 1 | 0.8 |
| cortex_medulla_late_corr | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | 0.9975 | 0.2277 |
| train_kNMSE | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | nan | nan |
| val_kNMSE | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | nan | nan |
| test_kNMSE | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | nan | nan |
| wall_s | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | nan | nan |
| peak_gpu_mb | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | nan | nan |
| params | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | nan ± nan (0) | +nan | +nan | +nan | nan | nan |

model-free aorta FWHM (s): 196.26623376623377
