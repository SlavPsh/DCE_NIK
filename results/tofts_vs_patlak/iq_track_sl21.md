# image quality vs curve fidelity, slice 21, k80 (image metrics vs cs100 = grasp-pro all spokes, body-masked, window-matched frames at 90 and 300 s; airE = rms in air / rms in body; static_noise = high-pass temporal std in the static roi; curves vs model-free)

| recon | haarpsi_90 | ssim_90 | psnr_90 | airE_90 | haarpsi_300 | airE_300 | static_noise | cortex_peak | cortex_washout | cortex_nrmse | medulla_nrmse |
|---|---|---|---|---|---|---|---|---|---|---|---|
| model-free | 0.710 | 0.831 | 27.844 | 0.415 | 0.702 | 0.413 | 0.021 | 0.977 | 0.977 | 0.023 | 0.023 |
| cs100 (GRASP-Pro all spokes) | 1.000 | 1.000 | 125.651 | 0.157 | 1.000 | 0.135 | 0.020 | 0.837 | 1.024 | 0.084 | 0.101 |
| NIK-free | 0.799 | 0.900 | 30.663 | 0.162 | 0.839 | 0.168 | 0.005 | 0.727 | 0.880 | 0.149 | 0.079 |
| NIK-sub16 | 0.777 | 0.894 | 30.737 | 0.169 | 0.831 | 0.172 | 0.018 | 0.756 | 0.920 | 0.125 | 0.099 |
| NIK-tofts8 old (3k, restore) | 0.743 | 0.903 | 29.164 | 0.129 | 0.806 | 0.130 | 0.002 | 0.539 | 0.728 | 0.302 | 0.260 |
| NIK-tofts8 new (10k, rms1) | 0.716 | 0.851 | 28.676 | 0.180 | 0.815 | 0.152 | 0.004 | 0.839 | 0.996 | 0.057 | 0.042 |
| NIK-tofts new (10k, rms1) | 0.654 | 0.793 | 27.661 | 0.248 | 0.817 | 0.168 | 0.015 | 0.715 | 0.971 | 0.092 | 0.052 |
| NIK-patlak new (10k) | 0.778 | 0.920 | 30.636 | 0.145 | 0.798 | 0.141 | 0.001 | 0.940 | 0.873 | 0.284 | 0.234 |
| GRASP-Pro | 0.671 | 0.888 | 27.263 | 0.155 | 0.921 | 0.158 | 0.045 | 0.646 | 0.925 | 0.216 | 0.132 |
| GRASP | 0.740 | 0.885 | 28.516 | 0.134 | 0.785 | 0.133 | 0.001 | 0.767 | 1.047 | 0.075 | 0.079 |