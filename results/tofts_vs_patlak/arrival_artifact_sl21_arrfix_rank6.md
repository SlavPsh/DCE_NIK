# artifacts at contrast arrival, p3 slice 21; arrival 55 s, aif peak 64 s; windows pre 0-40 s, arrival 50-75 s, plateau 90-150 s, late 200-375 s

## (1) artifact level per window (air energy / fine-scale fraction / temporal roughness)

| arm | pre | arrival | plateau | late |
|---|---|---|---|---|
| model-free 31-spoke | 0.563 / 0.323 / 0.234 | 0.545 / 0.343 / 0.261 | 0.464 / 0.316 / 0.244 | 0.459 / 0.300 / 0.234 |
| tofts6 rank6 | 0.155 / 0.228 / 0.001 | 0.178 / 0.270 / 0.031 | 0.128 / 0.238 / 0.004 | 0.131 / 0.222 / 0.001 |
| tofts8 prod in-coil | 0.157 / 0.226 / 0.002 | 0.165 / 0.263 / 0.050 | 0.132 / 0.240 / 0.007 | 0.135 / 0.224 / 0.003 |
| tofts8 prod out-coil | 0.162 / 0.222 / 0.001 | 0.157 / 0.253 / 0.036 | 0.133 / 0.237 / 0.005 | 0.136 / 0.221 / 0.002 |
| GRASP | 0.173 / 0.184 / 0.024 | 0.154 / 0.214 / 0.026 | 0.135 / 0.199 / 0.021 | 0.137 / 0.184 / 0.021 |

## (2) tofts6 rank6: per-atom coefficient images

| atom | n_eff spokes | air energy | fine fraction | rms body | share at arrival | share late |
|---|---|---|---|---|---|---|
| 0 | 427 | 0.116 | 0.222 | 4.25e-05 | 0.55 | 0.51 |
| 1 | 440 | 0.166 | 0.239 | 2e-05 | 0.24 | 0.42 |
| 2 | 254 | 0.149 | 0.229 | 1.63e-05 | 0.11 | 0.03 |
| 3 | 195 | 0.292 | 0.371 | 4.36e-06 | 0.07 | 0.02 |
| 4 | 71 | 0.290 | 0.390 | 2.46e-06 | 0.02 | 0.00 |
| 5 | 143 | 0.449 | 0.444 | 3.14e-06 | 0.02 | 0.02 |

## (3) tofts6 rank6: k-space residual per spoke by window

| window | spokes | NMSE | MSE (normalized) |
|---|---|---|---|
| pre | 147 | 0.4806 | 0.4346 |
| arrival | 90 | 0.3484 | 0.3794 |
| plateau | 219 | 0.4506 | 0.6329 |
| late | 638 | 0.4770 | 0.6718 |
