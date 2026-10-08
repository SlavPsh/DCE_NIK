# artifacts at contrast arrival, p14 slice 24; arrival 40 s, aif peak 49 s; windows pre 0-40 s, arrival 35-60 s, plateau 90-150 s, late 200-390 s

## (1) artifact level per window (air energy / fine-scale fraction / temporal roughness)

| arm | pre | arrival | plateau | late |
|---|---|---|---|---|
| model-free 31-spoke | 0.576 / 0.334 / 0.294 | 0.543 / 0.337 / 0.309 | 0.447 / 0.304 / 0.294 | 0.467 / 0.308 / 0.284 |
| tofts8 in-coil k100 | 0.131 / 0.224 / 0.002 | 0.164 / 0.232 / 0.039 | 0.108 / 0.196 / 0.004 | 0.108 / 0.198 / 0.004 |
| GRASP all spokes | 0.175 / 0.204 / 0.026 | 0.155 / 0.205 / 0.031 | 0.123 / 0.177 / 0.024 | 0.133 / 0.177 / 0.023 |

## (2) tofts8 in-coil k100: per-atom coefficient images

| atom | n_eff spokes | air energy | fine fraction | rms body | share at arrival | share late |
|---|---|---|---|---|---|---|
| 0 | 1167 | 0.109 | 0.199 | 1.65e-05 | 0.57 | 0.69 |
| 1 | 786 | 0.105 | 0.194 | 4.56e-06 | 0.18 | 0.25 |
| 2 | 196 | 0.127 | 0.220 | 4.13e-06 | 0.07 | 0.01 |
| 3 | 144 | 0.118 | 0.188 | 1.56e-06 | 0.13 | 0.02 |
| 4 | 108 | 0.253 | 0.275 | 5.84e-07 | 0.02 | 0.01 |
| 5 | 178 | 0.337 | 0.353 | 5.16e-07 | 0.00 | 0.00 |
| 6 | 230 | 0.438 | 0.403 | 6.33e-07 | 0.00 | 0.00 |
| 7 | 500 | 0.477 | 0.396 | 7.44e-07 | 0.02 | 0.02 |

## (3) tofts8 in-coil k100: k-space residual per spoke by window

| window | spokes | NMSE | MSE (normalized) |
|---|---|---|---|
| pre | 178 | 0.4989 | 0.4652 |
| arrival | 111 | 0.4444 | 0.4505 |
| plateau | 264 | 0.4522 | 0.635 |
| late | 840 | 0.5068 | 0.7002 |
