# tofts basis span vs trained arms, in vivo slice 19, k80 (reference = model-free 31-spoke nufft, roi enhancement, baseline subtracted)

ratios are amplitude relative to the model-free curve: first-pass peak (20 to 210 s) and washout mean (t > 200 s). projection = least squares of the model-free curve onto the first R atoms at the frame times (what a perfect fit inside the span could reach).

| curve | aorta peak / washout | cortex peak / washout | medulla peak / washout | liver peak / washout | body voxel err (all / dynamic part) |
|---|---|---|---|---|---|
| projection rank 3 | 0.99 / 1.02 | 1.00 / 0.90 | 0.96 / 0.92 | 0.67 / 0.94 | 0.255 / 0.577 |
| projection rank 5 | 0.99 / 1.02 | 0.99 / 1.00 | 1.01 / 1.01 | 0.76 / 0.99 | 0.240 / 0.544 |
| projection rank 8 | 0.99 / 1.02 | 0.96 / 0.99 | 0.91 / 1.00 | 0.79 / 0.98 | 0.234 / 0.528 |
| projection rank 12 | 1.01 / 1.00 | 0.96 / 1.00 | 0.94 / 1.00 | 0.79 / 0.99 | 0.230 / 0.517 |
| NIK-tofts (trained) | 0.67 / 0.79 | 0.39 / 0.57 | 0.51 / 0.56 | 1.25 / 1.27 | |
| NIK-tofts8 (trained) | 0.70 / 0.83 | 0.40 / 0.58 | 0.50 / 0.57 | 1.04 / 1.27 | |
| NIK-patlak (trained) | 0.97 / 1.08 | 0.69 / 0.79 | 0.82 / 0.81 | 0.87 / 1.10 | |

basis aif vs model-free aorta: time to peak 63.8 s vs 63.2 s; plateau / peak 0.23 vs 0.23