# artifacts at contrast arrival, p3 slice 21; arrival 55 s, aif peak 64 s; windows pre 0-40 s, arrival 50-75 s, plateau 90-150 s, late 200-375 s

## (1) artifact level per window (air energy / fine-scale fraction / temporal roughness)

| arm | pre | arrival | plateau | late |
|---|---|---|---|---|
| model-free 31-spoke | 0.563 / 0.323 / 0.234 | 0.545 / 0.343 / 0.261 | 0.464 / 0.316 / 0.244 | 0.459 / 0.300 / 0.234 |
| patlak+prior | 0.149 / 0.236 / 0.000 | 0.141 / 0.265 / 0.016 | 0.123 / 0.232 / 0.001 | 0.123 / 0.217 / 0.001 |

## (2) patlak+prior: per-atom coefficient images

| atom | n_eff spokes | air energy | fine fraction | rms body | share at arrival | share late |
|---|---|---|---|---|---|---|
| 0 | 427 | 0.141 | 0.399 | 2.78e-05 | 0.42 | 0.10 |
| 1 | 773 | 0.288 | 0.313 | 2.41e-05 | 0.02 | 0.32 |
| 2 | 1368 | 0.149 | 0.236 | 3.66e-05 | 0.56 | 0.58 |

## (3) patlak+prior: k-space residual per spoke by window

| window | spokes | NMSE | MSE (normalized) |
|---|---|---|---|
| pre | 147 | 0.5343 | 0.5459 |
| arrival | 90 | 0.4666 | 0.5602 |
| plateau | 219 | 0.4970 | 0.7526 |
| late | 638 | 0.4993 | 0.7318 |
