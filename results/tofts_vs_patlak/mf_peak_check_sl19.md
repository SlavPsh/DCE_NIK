# model-free reference: first-pass peak vs sliding-window width, slice 19 (nufft, ramp dcf, sense combine, approved rois; enhancement = baseline-subtracted roi mean)

| window (spokes) | window (s) | frames | aorta peak (ratio to 31) / ttp s / late noise | cortex peak (ratio to 31) / ttp s / late noise | medulla peak (ratio to 31) / ttp s / late noise |
|---|---|---|---|---|---|
| 31 | 6.8 | 240 | 0.00214 (1.00) / 63 / 0.009 | 0.00100 (1.00) / 71 / 0.016 | 0.00076 (1.00) / 200 / 0.019 |
| 21 | 4.6 | 338 | 0.00150 (0.70) / 63 / 0.011 | 0.00068 (0.68) / 69 / 0.021 | 0.00053 (0.69) / 199 / 0.031 |
| 15 | 3.3 | 566 | 0.00110 (0.51) / 63 / 0.016 | 0.00050 (0.50) / 68 / 0.019 | 0.00038 (0.50) / 198 / 0.017 |
| 11 | 2.4 | 850 | 0.00083 (0.39) / 63 / 0.023 | 0.00039 (0.39) / 72 / 0.024 | 0.00029 (0.38) / 68 / 0.026 |
| 7 | 1.5 | 852 | 0.00053 (0.25) / 63 / 0.033 | 0.00026 (0.26) / 68 / 0.044 | 0.00019 (0.25) / 191 / 0.052 |

reading: ratio > 1 at narrow windows = the 31-spoke reference under-reads the peak by that factor (temporal smoothing of the window); the late noise column shows what the narrower window costs