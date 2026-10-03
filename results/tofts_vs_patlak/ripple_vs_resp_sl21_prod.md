# ripple vs respiration, p3 slice 21: ripple = high-pass temporal std / mean; corr_nav = correlation of the high-passed curve with the k-centre respiratory navigator; nrmse vs model-free before / after smoothing the recon curve with the reference window (6.8 s)

| recon | frames | aorta_ripple | cortex_ripple | medulla_ripple | aorta_corr_nav | cortex_corr_nav | medulla_corr_nav | aorta_nrmse | cortex_nrmse | medulla_nrmse | aorta_nrmse_smoothed | cortex_nrmse_smoothed | medulla_nrmse_smoothed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| model-free 31-spoke (control) | - | 0.108 | 0.027 | 0.021 | 0.025 | -0.135 | 0.505 | - | - | - | - | - | - |
| sub16 wd3e-3+prior | 342 | 0.123 | 0.039 | 0.050 | -0.588 | -0.397 | 0.562 | 0.925 | 0.918 | 0.914 | 0.927 | 0.918 | 0.915 |
| sub16 nowarm | 342 | 0.047 | 0.018 | 0.016 | -0.087 | -0.036 | 0.008 | 0.922 | 0.924 | 0.918 | 0.924 | 0.925 | 0.918 |
| sub16 w0_10 | 342 | 0.021 | 0.007 | 0.006 | -0.005 | -0.002 | -0.001 | 0.926 | 0.922 | 0.913 | 0.926 | 0.922 | 0.913 |
| tofts8 | 342 | 0.048 | 0.013 | 0.009 | -0.007 | 0.001 | 0.009 | 0.912 | 0.906 | 0.903 | 0.913 | 0.906 | 0.903 |
| NIK-free | 342 | 0.024 | 0.006 | 0.004 | -0.004 | 0.015 | 0.041 | 0.917 | 0.916 | 0.910 | 0.918 | 0.916 | 0.910 |
| GRASP | 142 | 0.137 | 0.030 | 0.013 | -0.015 | -0.001 | 0.044 | 0.944 | 0.937 | 0.934 | 0.944 | 0.937 | 0.934 |
