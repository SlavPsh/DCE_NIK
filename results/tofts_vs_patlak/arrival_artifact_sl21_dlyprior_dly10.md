# artifacts at contrast arrival, p3 slice 21; arrival 55 s, aif peak 64 s; windows pre 0-40 s, arrival 50-75 s, plateau 90-150 s, late 200-375 s

## (1) artifact level per window (air energy / fine-scale fraction / temporal roughness)

| arm | pre | arrival | plateau | late |
|---|---|---|---|---|
| model-free 31-spoke | 0.563 / 0.323 / 0.234 | 0.545 / 0.343 / 0.261 | 0.464 / 0.316 / 0.244 | 0.459 / 0.300 / 0.234 |
| tofts8 dly10 | 0.173 / 0.225 / 0.002 | 0.176 / 0.258 / 0.041 | 0.142 / 0.236 / 0.005 | 0.143 / 0.222 / 0.002 |
| tofts8 prod in-coil | 0.157 / 0.226 / 0.002 | 0.165 / 0.263 / 0.050 | 0.132 / 0.240 / 0.007 | 0.135 / 0.224 / 0.003 |
| tofts8 prod out-coil | 0.162 / 0.222 / 0.001 | 0.157 / 0.253 / 0.036 | 0.133 / 0.237 / 0.005 | 0.136 / 0.221 / 0.002 |
| GRASP | 0.173 / 0.184 / 0.024 | 0.154 / 0.214 / 0.026 | 0.135 / 0.199 / 0.021 | 0.137 / 0.184 / 0.021 |

## (2) tofts8 dly10: per-atom coefficient images

| atom | n_eff spokes | air energy | fine fraction | rms body | share at arrival | share late |
|---|---|---|---|---|---|---|
| 0 | 427 | 0.118 | 0.220 | 4.23e-05 | 0.54 | 0.51 |
| 1 | 440 | 0.171 | 0.235 | 2.01e-05 | 0.24 | 0.42 |
| 2 | 254 | 0.166 | 0.225 | 1.64e-05 | 0.11 | 0.03 |
| 3 | 195 | 0.334 | 0.351 | 3.97e-06 | 0.06 | 0.02 |
| 4 | 71 | 0.371 | 0.357 | 1.93e-06 | 0.01 | 0.00 |
| 5 | 143 | 0.561 | 0.413 | 2.62e-06 | 0.01 | 0.01 |
| 6 | 136 | 0.570 | 0.416 | 1.74e-06 | 0.00 | 0.00 |
| 7 | 439 | 0.756 | 0.478 | 2.56e-06 | 0.02 | 0.00 |

## (3) tofts8 dly10: k-space residual per spoke by window

| window | spokes | NMSE | MSE (normalized) |
|---|---|---|---|
| pre | 147 | 0.4748 | 0.4253 |
| arrival | 90 | 0.3217 | 0.3495 |
| plateau | 219 | 0.4061 | 0.5422 |
| late | 638 | 0.4497 | 0.6087 |
