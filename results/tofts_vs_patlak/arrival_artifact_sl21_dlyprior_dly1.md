# artifacts at contrast arrival, p3 slice 21; arrival 55 s, aif peak 64 s; windows pre 0-40 s, arrival 50-75 s, plateau 90-150 s, late 200-375 s

## (1) artifact level per window (air energy / fine-scale fraction / temporal roughness)

| arm | pre | arrival | plateau | late |
|---|---|---|---|---|
| model-free 31-spoke | 0.563 / 0.323 / 0.234 | 0.545 / 0.343 / 0.261 | 0.464 / 0.316 / 0.244 | 0.459 / 0.300 / 0.234 |
| tofts8 dly1 | 0.159 / 0.225 / 0.002 | 0.168 / 0.261 / 0.047 | 0.132 / 0.241 / 0.006 | 0.136 / 0.224 / 0.003 |
| tofts8 prod in-coil | 0.157 / 0.226 / 0.002 | 0.165 / 0.263 / 0.050 | 0.132 / 0.240 / 0.007 | 0.135 / 0.224 / 0.003 |
| tofts8 prod out-coil | 0.162 / 0.222 / 0.001 | 0.157 / 0.253 / 0.036 | 0.133 / 0.237 / 0.005 | 0.136 / 0.221 / 0.002 |
| GRASP | 0.173 / 0.184 / 0.024 | 0.154 / 0.214 / 0.026 | 0.135 / 0.199 / 0.021 | 0.137 / 0.184 / 0.021 |

## (2) tofts8 dly1: per-atom coefficient images

| atom | n_eff spokes | air energy | fine fraction | rms body | share at arrival | share late |
|---|---|---|---|---|---|---|
| 0 | 427 | 0.114 | 0.220 | 4.26e-05 | 0.54 | 0.51 |
| 1 | 440 | 0.162 | 0.239 | 2.01e-05 | 0.24 | 0.42 |
| 2 | 254 | 0.153 | 0.226 | 1.65e-05 | 0.11 | 0.03 |
| 3 | 195 | 0.290 | 0.361 | 4.12e-06 | 0.06 | 0.02 |
| 4 | 71 | 0.290 | 0.373 | 2.24e-06 | 0.01 | 0.00 |
| 5 | 143 | 0.462 | 0.425 | 2.8e-06 | 0.01 | 0.01 |
| 6 | 136 | 0.402 | 0.427 | 2.23e-06 | 0.01 | 0.01 |
| 7 | 439 | 0.576 | 0.474 | 2.9e-06 | 0.02 | 0.00 |

## (3) tofts8 dly1: k-space residual per spoke by window

| window | spokes | NMSE | MSE (normalized) |
|---|---|---|---|
| pre | 147 | 0.4720 | 0.4228 |
| arrival | 90 | 0.3166 | 0.342 |
| plateau | 219 | 0.4021 | 0.5377 |
| late | 638 | 0.4467 | 0.6056 |
