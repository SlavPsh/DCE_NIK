# ripple vs respiration, p14 slice 24: ripple = high-pass temporal std / mean; corr_nav = correlation of the high-passed curve with the k-centre respiratory navigator; nrmse vs model-free before / after smoothing the recon curve with the reference window (5.6 s)

| recon | frames | aorta_ripple | liver_ripple | spleen_ripple | aorta_ripple_smoothed | liver_ripple_smoothed | spleen_ripple_smoothed | aorta_corr_nav | liver_corr_nav | spleen_corr_nav | aorta_nrmse | liver_nrmse | spleen_nrmse | aorta_nrmse_smoothed | liver_nrmse_smoothed | spleen_nrmse_smoothed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| model-free 31-spoke (control) | - | 0.047 | 0.017 | 0.024 | - | - | - | 0.094 | -0.193 | -0.029 | - | - | - | - | - | - |
| sub16 wd3e-3+prior | 431 | 0.043 | 0.021 | 0.021 | 0.018 | 0.007 | 0.007 | -0.020 | -0.043 | -0.037 | 0.308 | 0.103 | 0.206 | 0.302 | 0.089 | 0.201 |
| tofts8 | 431 | 0.014 | 0.003 | 0.005 | 0.010 | 0.002 | 0.004 | -0.005 | 0.010 | -0.001 | 0.081 | 0.059 | 0.055 | 0.085 | 0.059 | 0.055 |
| NIK-free | 431 | 0.004 | 0.001 | 0.002 | 0.004 | 0.001 | 0.002 | 0.007 | -0.012 | -0.012 | 0.187 | 0.067 | 0.154 | 0.188 | 0.067 | 0.155 |
| GRASP | 179 | 0.023 | 0.004 | 0.009 | 0.020 | 0.004 | 0.008 | -0.005 | 0.030 | 0.005 | 0.115 | 0.050 | 0.065 | 0.118 | 0.051 | 0.066 |
