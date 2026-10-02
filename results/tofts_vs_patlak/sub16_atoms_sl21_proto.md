# sub16 atom diagnostic, p3 slice 21: respiratory content of the learned atoms (power above 0.1 Hz / total) and oscillation of the roi curves (high-pass temporal std / mean) vs curve nrmse

| run | rank | resp_frac_mean | resp_frac_max | n_atoms_resp_gt_0p3 | aorta_osc | cortex_osc | medulla_osc | aorta_nrmse | cortex_nrmse | medulla_nrmse | best_heldout |
|---|---|---|---|---|---|---|---|---|---|---|---|
| model-free (aorta osc only) | - | - | - | - | 0.108 | - | - | - | - | - | - |
| sub16 051 | 16 | 0.049 | 0.139 | 0 | 0.106 | 0.036 | 0.051 | 0.927 | 0.918 | 0.914 | 0.290 |
| sub16 prod | 16 | 0.013 | 0.029 | 0 | 0.085 | 0.028 | 0.028 | 0.936 | 0.936 | 0.933 | 0.291 |
| sub16 wd3e-3 prior | 16 | 0.037 | 0.106 | 0 | 0.123 | 0.039 | 0.050 | 0.925 | 0.918 | 0.914 | 0.291 |
| sub16 wd1e-2 noprior | 16 | 0.012 | 0.027 | 0 | 0.075 | 0.023 | 0.026 | 0.930 | 0.937 | 0.934 | 0.290 |
