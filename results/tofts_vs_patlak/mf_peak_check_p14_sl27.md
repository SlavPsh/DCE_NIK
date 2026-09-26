# model-free reference: first-pass peak vs sliding-window width, slice 27 (nufft, ramp dcf, sense combine, approved rois; enhancement = baseline-subtracted roi mean)

| window (spokes) | window (s) | frames | aorta peak (ratio to 31) / ttp s / late noise | liver peak (ratio to 31) / ttp s / late noise | spleen peak (ratio to 31) / ttp s / late noise |
|---|---|---|---|---|---|
| 31 | 5.6 | 304 | 0.00001 (1.00) / 52 / 0.020 | 0.00001 (1.00) / 101 / 0.034 | 0.00001 (1.00) / 84 / 0.032 |
| 21 | 3.8 | 428 | 0.00001 (1.04) / 52 / 0.017 | 0.00001 (1.01) / 118 / 0.036 | 0.00001 (1.02) / 77 / 0.050 |
| 15 | 2.7 | 715 | 0.00001 (1.09) / 53 / 0.042 | 0.00001 (1.06) / 111 / 0.046 | 0.00001 (1.05) / 77 / 0.033 |
| 11 | 2.0 | 1074 | 0.00001 (1.07) / 52 / 0.032 | 0.00001 (1.13) / 112 / 0.068 | 0.00001 (1.09) / 86 / 0.042 |

reading: ratio > 1 at narrow windows = the 31-spoke reference under-reads the peak by that factor (temporal smoothing of the window); the late noise column shows what the narrower window costs