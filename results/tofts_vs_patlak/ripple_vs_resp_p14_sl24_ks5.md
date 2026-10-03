# ripple vs respiration, p14 slice 24: ripple = high-pass temporal std / mean; corr_nav = correlation of the high-passed curve with the k-centre respiratory navigator; nrmse vs model-free before / after smoothing the recon curve with the reference window (5.6 s)

| recon | frames | aorta_ripple | liver_ripple | spleen_ripple | aorta_corr_nav | liver_corr_nav | spleen_corr_nav | aorta_nrmse | liver_nrmse | spleen_nrmse | aorta_nrmse_smoothed | liver_nrmse_smoothed | spleen_nrmse_smoothed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| model-free 31-spoke (control) | - | 0.047 | 0.017 | 0.024 | 0.094 | -0.193 | -0.029 | - | - | - | - | - | - |
| sub16 wd3e-3+prior | 431 | 0.043 | 0.021 | 0.021 | -0.020 | -0.043 | -0.037 | 0.953 | 0.922 | 0.942 | 0.955 | 0.923 | 0.941 |
| tofts8 | 431 | 0.014 | 0.003 | 0.005 | -0.005 | 0.010 | -0.001 | 0.907 | 0.902 | 0.915 | 0.908 | 0.902 | 0.915 |
| NIK-free | 431 | 0.004 | 0.001 | 0.002 | 0.007 | -0.012 | -0.012 | 0.922 | 0.902 | 0.916 | 0.922 | 0.902 | 0.916 |
| GRASP | 179 | 0.023 | 0.004 | 0.009 | -0.005 | 0.030 | 0.005 | 0.938 | 0.935 | 0.935 | 0.938 | 0.935 | 0.935 |
