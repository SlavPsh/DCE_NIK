# artifacts at contrast arrival, p14 slice 24; arrival 40 s, aif peak 49 s; windows pre 0-40 s, arrival 35-60 s, plateau 90-150 s, late 200-390 s

## (1) artifact level per window (air energy / fine-scale fraction / temporal roughness)

| arm | pre | arrival | plateau | late |
|---|---|---|---|---|
| model-free 31-spoke | 0.576 / 0.334 / 0.294 | 0.543 / 0.337 / 0.309 | 0.447 / 0.304 / 0.294 | 0.467 / 0.308 / 0.284 |
| tofts8 out-coil k100 | 0.129 / 0.222 / 0.001 | 0.148 / 0.227 / 0.025 | 0.107 / 0.201 / 0.003 | 0.108 / 0.206 / 0.002 |
| GRASP all spokes | 0.175 / 0.204 / 0.026 | 0.155 / 0.205 / 0.031 | 0.123 / 0.177 / 0.024 | 0.133 / 0.177 / 0.023 |

## (2) tofts8 out-coil k100: per-atom coefficient images

| atom | n_eff spokes | air energy | fine fraction | rms body | share at arrival | share late |
|---|---|---|---|---|---|---|
| 0 | 1167 | 0.106 | 0.206 | 1.62e-05 | 0.58 | 0.70 |
| 1 | 786 | 0.110 | 0.205 | 4.42e-06 | 0.18 | 0.25 |
| 2 | 196 | 0.125 | 0.219 | 4.08e-06 | 0.07 | 0.01 |
| 3 | 144 | 0.109 | 0.172 | 1.49e-06 | 0.13 | 0.02 |
| 4 | 108 | 0.206 | 0.219 | 5.4e-07 | 0.02 | 0.01 |
| 5 | 178 | 0.283 | 0.297 | 4.67e-07 | 0.00 | 0.00 |
| 6 | 230 | 0.643 | 0.392 | 3.94e-07 | 0.00 | 0.00 |
| 7 | 500 | 0.741 | 0.384 | 5.24e-07 | 0.01 | 0.01 |

## (3) tofts8 out-coil k100: k-space residual per spoke by window

| window | spokes | NMSE | MSE (normalized) |
|---|---|---|---|
| pre | 178 | 0.4625 | 0.4282 |
| arrival | 111 | 0.4160 | 0.4207 |
| plateau | 264 | 0.4291 | 0.6007 |
| late | 840 | 0.4627 | 0.6312 |
