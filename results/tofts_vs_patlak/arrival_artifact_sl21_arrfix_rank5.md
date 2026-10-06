# artifacts at contrast arrival, p3 slice 21; arrival 55 s, aif peak 64 s; windows pre 0-40 s, arrival 50-75 s, plateau 90-150 s, late 200-375 s

## (1) artifact level per window (air energy / fine-scale fraction / temporal roughness)

| arm | pre | arrival | plateau | late |
|---|---|---|---|---|
| model-free 31-spoke | 0.563 / 0.323 / 0.234 | 0.545 / 0.343 / 0.261 | 0.464 / 0.316 / 0.244 | 0.459 / 0.300 / 0.234 |
| tofts5 rank5 | 0.156 / 0.227 / 0.001 | 0.181 / 0.273 / 0.032 | 0.128 / 0.241 / 0.004 | 0.128 / 0.223 / 0.001 |
| tofts8 prod in-coil | 0.157 / 0.226 / 0.002 | 0.165 / 0.263 / 0.050 | 0.132 / 0.240 / 0.007 | 0.135 / 0.224 / 0.003 |
| tofts8 prod out-coil | 0.162 / 0.222 / 0.001 | 0.157 / 0.253 / 0.036 | 0.133 / 0.237 / 0.005 | 0.136 / 0.221 / 0.002 |
| GRASP | 0.173 / 0.184 / 0.024 | 0.154 / 0.214 / 0.026 | 0.135 / 0.199 / 0.021 | 0.137 / 0.184 / 0.021 |

## (2) tofts5 rank5: per-atom coefficient images

| atom | n_eff spokes | air energy | fine fraction | rms body | share at arrival | share late |
|---|---|---|---|---|---|---|
| 0 | 427 | 0.116 | 0.224 | 4.22e-05 | 0.55 | 0.52 |
| 1 | 440 | 0.168 | 0.240 | 1.99e-05 | 0.25 | 0.42 |
| 2 | 254 | 0.150 | 0.231 | 1.62e-05 | 0.11 | 0.03 |
| 3 | 195 | 0.319 | 0.395 | 4.76e-06 | 0.08 | 0.03 |
| 4 | 71 | 0.320 | 0.422 | 2.93e-06 | 0.02 | 0.00 |

## (3) tofts5 rank5: k-space residual per spoke by window

| window | spokes | NMSE | MSE (normalized) |
|---|---|---|---|
| pre | 147 | 0.4777 | 0.4356 |
| arrival | 90 | 0.3524 | 0.3855 |
| plateau | 219 | 0.4482 | 0.6295 |
| late | 638 | 0.4867 | 0.6984 |
