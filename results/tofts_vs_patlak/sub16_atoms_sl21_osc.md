# sub16 atom diagnostic, p3 slice 21: respiratory content of the learned atoms (power above 0.1 Hz / total) and oscillation of the roi curves (high-pass temporal std / mean) vs curve nrmse

| run | rank | resp_frac_mean | resp_frac_max | n_atoms_resp_gt_0p3 | aorta_osc | cortex_osc | medulla_osc | aorta_nrmse | cortex_nrmse | medulla_nrmse | best_heldout |
|---|---|---|---|---|---|---|---|---|---|---|---|
| model-free (aorta osc only) | - | - | - | - | 0.108 | - | - | - | - | - | - |
| sub16 wd3e-3 prior (pca warm w0 30) | 16 | 0.037 | 0.106 | 0 | 0.123 | 0.039 | 0.050 | 0.925 | 0.918 | 0.914 | 0.291 |
| tofts8 prod | 8 | 0.003 | 0.013 | 0 | 0.048 | 0.013 | 0.009 | 0.912 | 0.906 | 0.903 | 0.546 |
| sub16 wd3e-3 nowarm | 16 | 0.002 | 0.008 | 0 | 0.047 | 0.018 | 0.016 | 0.922 | 0.924 | 0.918 | 0.366 |
| sub16 wd3e-3 w0_10 | 16 | 0.003 | 0.008 | 0 | 0.021 | 0.007 | 0.006 | 0.926 | 0.922 | 0.913 | 0.399 |
| sub16 wd3e-3 nowarm_w0_10 | 16 | 0.005 | 0.012 | 0 | 0.014 | 0.005 | 0.004 | 0.919 | 0.924 | 0.913 | 0.425 |
