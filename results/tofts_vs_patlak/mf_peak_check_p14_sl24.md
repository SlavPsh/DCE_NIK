# model-free reference: first-pass peak vs sliding-window width, slice 24 (nufft, ramp dcf, sense combine, approved rois; enhancement = baseline-subtracted roi mean)

| window (spokes) | window (s) | frames | aorta peak (ratio to 31) / ttp s / late noise | liver peak (ratio to 31) / ttp s / late noise | spleen peak (ratio to 31) / ttp s / late noise |
|---|---|---|---|---|---|
| 31 | 5.6 | 304 | 0.00001 (1.00) / 47 / 0.028 | 0.00000 (1.00) / 99 / 0.029 | 0.00000 (1.00) / 84 / 0.030 |
| 21 | 3.8 | 428 | 0.00001 (1.01) / 51 / 0.025 | 0.00000 (0.98) / 96 / 0.024 | 0.00000 (1.01) / 77 / 0.049 |
| 15 | 2.7 | 715 | 0.00001 (1.03) / 53 / 0.056 | 0.00001 (1.05) / 117 / 0.044 | 0.00000 (1.03) / 77 / 0.032 |
| 11 | 2.0 | 1074 | 0.00001 (1.02) / 52 / 0.032 | 0.00001 (1.09) / 112 / 0.064 | 0.00001 (1.08) / 86 / 0.045 |

reading: ratio > 1 at narrow windows = the 31-spoke reference under-reads the peak by that factor (temporal smoothing of the window); the late noise column shows what the narrower window costs