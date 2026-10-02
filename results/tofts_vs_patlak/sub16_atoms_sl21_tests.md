# sub16 atom diagnostic, p3 slice 21: respiratory content of the learned atoms (power above 0.1 Hz / total) and oscillation of the roi curves (high-pass temporal std / mean) vs curve nrmse

| run | rank | resp_frac_mean | resp_frac_max | n_atoms_resp_gt_0p3 | aorta_osc | cortex_osc | medulla_osc | aorta_nrmse | cortex_nrmse | medulla_nrmse | best_heldout |
|---|---|---|---|---|---|---|---|---|---|---|---|
| model-free (aorta osc only) | - | - | - | - | 0.108 | - | - | - | - | - | - |
| sub16 prod (pca warm 100 fr) | 16 | 0.013 | 0.029 | 0 | 0.085 | 0.028 | 0.028 | 0.936 | 0.936 | 0.933 | 0.291 |
| tofts8 prod | 8 | 0.003 | 0.013 | 0 | 0.048 | 0.013 | 0.009 | 0.912 | 0.906 | 0.903 | 0.546 |
| sub16 nowarm | 16 | 0.002 | 0.005 | 0 | 0.033 | 0.014 | 0.012 | 0.934 | 0.938 | 0.934 | 0.357 |
| sub16 nocap | 16 | 0.003 | 0.005 | 0 | 0.029 | 0.012 | 0.011 | 0.933 | 0.941 | 0.940 | 0.360 |
| sub16 toftsinit | 16 | 0.009 | 0.016 | 0 | 0.086 | 0.031 | 0.034 | 0.929 | 0.935 | 0.934 | 0.285 |
| sub16 phitv0.03 | 16 | 0.011 | 0.018 | 0 | 0.089 | 0.024 | 0.030 | 0.929 | 0.936 | 0.937 | 0.285 |
| sub16 phitv0.3 | 16 | 0.007 | 0.014 | 0 | 0.067 | 0.022 | 0.018 | 0.926 | 0.932 | 0.926 | 0.310 |
| sub16 w0_10 | 16 | 0.004 | 0.010 | 0 | 0.011 | 0.004 | 0.003 | 0.936 | 0.934 | 0.921 | 0.409 |
| sub16 ortho | 16 | 0.820 | 0.964 | 16 | 0.308 | 0.308 | 0.312 | 1.011 | 1.022 | 1.021 | 0.885 |
