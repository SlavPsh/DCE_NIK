# model-free reference: first-pass peak vs sliding-window width, slice 18 (nufft, ramp dcf, sense combine, approved rois; enhancement = baseline-subtracted roi mean)

| window (spokes) | window (s) | frames | aorta peak (ratio to 31) / ttp s / late noise | cortex peak (ratio to 31) / ttp s / late noise | medulla peak (ratio to 31) / ttp s / late noise |
|---|---|---|---|---|---|
| 31 | 6.8 | 240 | 0.00213 (1.00) / 63 / 0.010 | 0.00086 (1.00) / 71 / 0.018 | 0.00072 (1.00) / 200 / 0.022 |
| 21 | 4.6 | 338 | 0.00148 (0.69) / 63 / 0.012 | 0.00059 (0.69) / 72 / 0.022 | 0.00050 (0.69) / 202 / 0.032 |
| 15 | 3.3 | 566 | 0.00109 (0.51) / 63 / 0.019 | 0.00045 (0.53) / 73 / 0.021 | 0.00036 (0.50) / 184 / 0.021 |
| 11 | 2.4 | 850 | 0.00083 (0.39) / 63 / 0.025 | 0.00034 (0.40) / 72 / 0.030 | 0.00027 (0.38) / 197 / 0.031 |
| 7 | 1.5 | 852 | 0.00054 (0.25) / 63 / 0.030 | 0.00023 (0.27) / 72 / 0.052 | 0.00019 (0.26) / 197 / 0.053 |

reading: ratio > 1 at narrow windows = the 31-spoke reference under-reads the peak by that factor (temporal smoothing of the window); the late noise column shows what the narrower window costs