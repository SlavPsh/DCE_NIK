# model-free reference: first-pass peak vs sliding-window width, slice 21 (nufft, ramp dcf, sense combine, approved rois; enhancement = baseline-subtracted roi mean)

| window (spokes) | window (s) | frames | aorta peak (ratio to 31) / ttp s / late noise | liver peak (ratio to 31) / ttp s / late noise | spleen peak (ratio to 31) / ttp s / late noise |
|---|---|---|---|---|---|
| 31 | 5.6 | 304 | 0.00001 (1.00) / 52 / 0.027 | 0.00001 (1.00) / 99 / 0.028 | 0.00001 (1.00) / 80 / 0.027 |
| 21 | 3.8 | 428 | 0.00001 (1.01) / 52 / 0.024 | 0.00001 (0.98) / 120 / 0.020 | 0.00001 (1.03) / 80 / 0.040 |
| 15 | 2.7 | 715 | 0.00001 (1.03) / 53 / 0.048 | 0.00001 (1.04) / 130 / 0.041 | 0.00001 (1.04) / 82 / 0.028 |
| 11 | 2.0 | 1074 | 0.00001 (1.09) / 53 / 0.040 | 0.00001 (1.06) / 98 / 0.061 | 0.00001 (1.07) / 78 / 0.042 |

reading: ratio > 1 at narrow windows = the 31-spoke reference under-reads the peak by that factor (temporal smoothing of the window); the late noise column shows what the narrower window costs