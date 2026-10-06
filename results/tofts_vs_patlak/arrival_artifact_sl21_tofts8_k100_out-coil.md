# artifacts at contrast arrival, p3 slice 21; arrival 55 s, aif peak 64 s; windows pre 0-40 s, arrival 50-75 s, plateau 90-150 s, late 200-375 s

## (1) artifact level per window (air energy / fine-scale fraction / temporal roughness)

| arm | pre | arrival | plateau | late |
|---|---|---|---|---|
| model-free 31-spoke | 0.563 / 0.323 / 0.234 | 0.545 / 0.343 / 0.261 | 0.464 / 0.316 / 0.244 | 0.459 / 0.300 / 0.234 |
| tofts8 k100 out-coil | 0.163 / 0.221 / 0.001 | 0.158 / 0.258 / 0.032 | 0.132 / 0.235 / 0.004 | 0.132 / 0.217 / 0.002 |
| tofts8 k80 in-coil | 0.157 / 0.226 / 0.002 | 0.165 / 0.263 / 0.050 | 0.132 / 0.240 / 0.007 | 0.135 / 0.224 / 0.003 |
| GRASP all | 0.165 / 0.187 / 0.020 | 0.147 / 0.217 / 0.025 | 0.129 / 0.204 / 0.019 | 0.131 / 0.187 / 0.019 |

## (2) tofts8 k100 out-coil: per-atom coefficient images

| atom | n_eff spokes | air energy | fine fraction | rms body | share at arrival | share late |
|---|---|---|---|---|---|---|
| 0 | 427 | 0.110 | 0.217 | 4.23e-05 | 0.55 | 0.51 |
| 1 | 440 | 0.161 | 0.230 | 1.97e-05 | 0.24 | 0.42 |
| 2 | 254 | 0.157 | 0.222 | 1.62e-05 | 0.11 | 0.03 |
| 3 | 195 | 0.302 | 0.344 | 4.07e-06 | 0.06 | 0.02 |
| 4 | 71 | 0.305 | 0.401 | 2.06e-06 | 0.01 | 0.00 |
| 5 | 143 | 0.481 | 0.425 | 2.4e-06 | 0.01 | 0.01 |
| 6 | 136 | 0.542 | 0.452 | 1.41e-06 | 0.00 | 0.00 |
| 7 | 439 | 0.717 | 0.487 | 2.23e-06 | 0.01 | 0.00 |

## (3) tofts8 k100 out-coil: k-space residual per spoke by window

| window | spokes | NMSE | MSE (normalized) |
|---|---|---|---|
| pre | 147 | 0.4802 | 0.4444 |
| arrival | 90 | 0.3582 | 0.3957 |
| plateau | 219 | 0.4183 | 0.5706 |
| late | 638 | 0.4494 | 0.6209 |
