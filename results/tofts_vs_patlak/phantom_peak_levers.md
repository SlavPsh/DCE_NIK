# phantom no-motion z15, tofts16 w768 ks2.5 40k steps best-VAL: atom scale (unit-norm vs unit-rms), wd 3e-3 vs 1e-2, per-atom wd on the residual atoms (wdf) vs truth

peak = baseline-subtracted first-pass peak / truth peak under the ONE global truth-derived image scale (xph_eval); peak_pm = same after matching the
90 to 200 s plateau to truth (first-pass deficit only). scale = global factor the recon needed (1 = correct amplitude). image metrics vs truth over the body, 40 frames.

| arm | best_step | scale | cortex_peak | cortex_peak_pm | medulla_peak | medulla_peak_pm | aorta_peak | aorta_peak_pm | cortex_nrmse | medulla_nrmse | aorta_nrmse | haarpsi | ssim | img_nrmse | test_knmse |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| phantom standard s0 (unit-norm atoms wd 3e-3) | 32000 | 0.00704 | 0.946 | 0.952 | 0.972 | 0.983 | 1.04 | 0.981 | 0.0147 | 0.00973 | 0.0788 | 0.904 | 0.936 | 0.0288 | 0.000733 |
| phantom standard s1 | 40000 | 0.007 | 0.944 | 0.954 | 0.97 | 0.983 | 1.03 | 0.989 | 0.015 | 0.0107 | 0.0773 | 0.906 | 0.936 | 0.0284 | 0.000647 |
| phantom standard s2 | 38000 | 0.00704 | 0.947 | 0.955 | 0.969 | 0.984 | 1.04 | 0.994 | 0.0152 | 0.0108 | 0.0721 | 0.901 | 0.936 | 0.0287 | 0.000699 |
| rms1_wd3e-3 | 24000 | 0.00702 | 0.973 | 0.956 | 1.01 | 0.998 | 1.05 | 0.979 | 0.0146 | 0.0126 | 0.0664 | 0.893 | 0.929 | 0.0269 | 0.000278 |
| rms1_wd1e-2 | 40000 | 0.00693 | 0.97 | 0.957 | 1 | 0.986 | 1.06 | 0.987 | 0.014 | 0.0133 | 0.0721 | 0.898 | 0.933 | 0.0266 | 0.000309 |
| rms1_wd1e-2_wdf0 | 40000 | 0.0069 | 0.885 | 0.898 | 0.953 | 0.982 | 1.02 | 0.989 | 0.028 | 0.019 | 0.0609 | 0.808 | 0.894 | 0.0352 | 0.00335 |
| rms1_wd1e-2_wdf1e-3 | 34000 | 0.00687 | 0.959 | 0.946 | 1.02 | 1 | 1.05 | 1.02 | 0.0145 | 0.0127 | 0.0366 | 0.879 | 0.924 | 0.0268 | 0.00093 |
| rms1_wd3e-3_wdf0 | 38000 | 0.00703 | 0.967 | 0.956 | 1.01 | 1 | 1.06 | 0.968 | 0.018 | 0.018 | 0.0871 | 0.893 | 0.934 | 0.0284 | 0.000586 |
