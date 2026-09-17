# model-free reference: first-pass peak vs sliding-window width, slice 21 (nufft, ramp dcf, sense combine, approved rois; enhancement = baseline-subtracted roi mean)

| window (spokes) | window (s) | frames | aorta peak (ratio to 31) / ttp s / late noise | cortex peak (ratio to 31) / ttp s / late noise | medulla peak (ratio to 31) / ttp s / late noise |
|---|---|---|---|---|---|
| 31 | 6.8 | 240 | 0.00191 (1.00) / 63 / 0.015 | 0.00097 (1.00) / 71 / 0.014 | 0.00072 (1.00) / 203 / 0.020 |
| 21 | 4.6 | 338 | 0.00137 (0.71) / 63 / 0.017 | 0.00067 (0.68) / 72 / 0.019 | 0.00049 (0.68) / 202 / 0.028 |
| 15 | 3.3 | 566 | 0.00100 (0.52) / 63 / 0.026 | 0.00049 (0.51) / 72 / 0.017 | 0.00036 (0.50) / 207 / 0.024 |
| 11 | 2.4 | 850 | 0.00075 (0.39) / 63 / 0.024 | 0.00037 (0.38) / 72 / 0.025 | 0.00027 (0.37) / 200 / 0.033 |
| 7 | 1.5 | 852 | 0.00050 (0.26) / 64 / 0.044 | 0.00025 (0.25) / 72 / 0.048 | 0.00019 (0.27) / 200 / 0.061 |

reading: ratio > 1 at narrow windows = the 31-spoke reference under-reads the peak by that factor (temporal smoothing of the window); the late noise column shows what the narrower window costs