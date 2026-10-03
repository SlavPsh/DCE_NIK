# ripple vs respiration, p3 slice 21: ripple = high-pass temporal std / mean; corr_nav = correlation of the high-passed curve with the k-centre respiratory navigator; nrmse vs model-free before / after smoothing the recon curve with the reference window (6.8 s)

| recon | frames | aorta_ripple | cortex_ripple | medulla_ripple | aorta_ripple_smoothed | cortex_ripple_smoothed | medulla_ripple_smoothed | aorta_corr_nav | cortex_corr_nav | medulla_corr_nav | aorta_nrmse | cortex_nrmse | medulla_nrmse | aorta_nrmse_smoothed | cortex_nrmse_smoothed | medulla_nrmse_smoothed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| model-free 31-spoke (control) | - | 0.108 | 0.027 | 0.021 | - | - | - | 0.025 | -0.135 | 0.505 | - | - | - | - | - | - |
| sub16 wd3e-3+prior | 342 | 0.123 | 0.039 | 0.050 | 0.038 | 0.013 | 0.014 | -0.588 | -0.397 | 0.562 | 0.303 | 0.105 | 0.115 | 0.276 | 0.086 | 0.092 |
| sub16 nowarm | 342 | 0.047 | 0.018 | 0.016 | 0.026 | 0.011 | 0.008 | -0.087 | -0.036 | 0.008 | 0.255 | 0.097 | 0.093 | 0.251 | 0.089 | 0.087 |
| sub16 w0_10 | 342 | 0.021 | 0.007 | 0.006 | 0.016 | 0.006 | 0.005 | -0.005 | -0.002 | -0.001 | 0.270 | 0.099 | 0.102 | 0.271 | 0.099 | 0.101 |
| tofts8 | 342 | 0.048 | 0.013 | 0.009 | 0.039 | 0.010 | 0.006 | -0.007 | 0.001 | 0.009 | 0.057 | 0.054 | 0.038 | 0.064 | 0.057 | 0.035 |
| NIK-free | 342 | 0.024 | 0.006 | 0.004 | 0.021 | 0.005 | 0.003 | -0.004 | 0.015 | 0.041 | 0.145 | 0.077 | 0.057 | 0.155 | 0.079 | 0.057 |
| GRASP | 142 | 0.137 | 0.030 | 0.013 | 0.108 | 0.025 | 0.011 | -0.015 | -0.001 | 0.044 | 0.128 | 0.067 | 0.040 | 0.139 | 0.071 | 0.042 |
