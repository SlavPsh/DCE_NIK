# artifacts at contrast arrival, p3 slice 21; arrival 55 s, aif peak 64 s; windows pre 0-40 s, arrival 50-75 s, plateau 90-150 s, late 200-375 s

## (1) artifact level per window (air energy / fine-scale fraction / temporal roughness)

| arm | pre | arrival | plateau | late |
|---|---|---|---|---|
| model-free 31-spoke | 0.563 / 0.323 / 0.234 | 0.545 / 0.343 / 0.261 | 0.464 / 0.316 / 0.244 | 0.459 / 0.300 / 0.234 |
| tofts8 k100 in-coil | 0.159 / 0.230 / 0.002 | 0.166 / 0.270 / 0.040 | 0.134 / 0.242 / 0.005 | 0.132 / 0.222 / 0.002 |
| tofts8 k80 in-coil | 0.157 / 0.226 / 0.002 | 0.165 / 0.263 / 0.050 | 0.132 / 0.240 / 0.007 | 0.135 / 0.224 / 0.003 |
| GRASP all | 0.165 / 0.187 / 0.020 | 0.147 / 0.217 / 0.025 | 0.129 / 0.204 / 0.019 | 0.131 / 0.187 / 0.019 |

## (2) tofts8 k100 in-coil: per-atom coefficient images

| atom | n_eff spokes | air energy | fine fraction | rms body | share at arrival | share late |
|---|---|---|---|---|---|---|
| 0 | 427 | 0.116 | 0.221 | 4.3e-05 | 0.54 | 0.51 |
| 1 | 440 | 0.156 | 0.238 | 2.01e-05 | 0.24 | 0.42 |
| 2 | 254 | 0.154 | 0.231 | 1.65e-05 | 0.11 | 0.03 |
| 3 | 195 | 0.261 | 0.363 | 4.19e-06 | 0.06 | 0.02 |
| 4 | 71 | 0.268 | 0.422 | 2.17e-06 | 0.01 | 0.00 |
| 5 | 143 | 0.411 | 0.433 | 2.48e-06 | 0.01 | 0.01 |
| 6 | 136 | 0.404 | 0.445 | 1.79e-06 | 0.00 | 0.00 |
| 7 | 439 | 0.635 | 0.486 | 2.48e-06 | 0.02 | 0.00 |

## (3) tofts8 k100 in-coil: k-space residual per spoke by window

| window | spokes | NMSE | MSE (normalized) |
|---|---|---|---|
| pre | 147 | 0.5147 | 0.4768 |
| arrival | 90 | 0.3705 | 0.4084 |
| plateau | 219 | 0.4429 | 0.6109 |
| late | 638 | 0.4805 | 0.675 |
