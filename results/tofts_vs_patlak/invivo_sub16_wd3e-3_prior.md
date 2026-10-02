# in vivo (meas_p3_dce, slices 21, k80 = 1368/1708 views (v%10<8), VAL v%10==8 for early stop, TEST v%10==9 untouched; same views for every method)

rulers: mf_* = NRMSE vs model-free NUFFT ROI curve on its 240-pt grid (affine = raw+affine fit; scale = baseline-subtracted single scale). physical bounds on aorta. *_kNMSE = complex k-space NMSE at held-out spokes (NIK only). CS rows are references, NOT truth; CS held-out blocked (magnitude-only files).

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
| wall_s | 4015 ± 0 (1) |  | nan | nan |
| peak_gpu_mb | 1.246e+04 ± 0 (1) |  | nan | nan |
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
