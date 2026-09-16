# tofts basis span vs trained arms, in vivo slice 18, k80 (reference = model-free 31-spoke nufft, roi enhancement, baseline subtracted)

ratios are amplitude relative to the model-free curve: first-pass peak (20 to 210 s) and washout mean (t > 200 s). projection = least squares of the model-free curve onto the first R atoms at the frame times (what a perfect fit inside the span could reach).

| curve | aorta peak / washout | cortex peak / washout | medulla peak / washout | liver peak / washout | body voxel err (all / dynamic part) |
|---|---|---|---|---|---|
| projection rank 3 | 0.98 / 1.03 | 0.99 / 0.90 | 0.88 / 0.93 | 0.68 / 0.94 | 0.246 / 0.558 |
| projection rank 5 | 0.99 / 1.03 | 1.02 / 1.01 | 0.94 / 1.00 | 0.79 / 1.00 | 0.231 / 0.524 |
| projection rank 8 | 0.99 / 1.03 | 0.95 / 1.00 | 0.99 / 1.00 | 0.82 / 0.99 | 0.225 / 0.508 |
| projection rank 12 | 1.01 / 1.00 | 0.96 / 1.01 | 0.99 / 1.00 | 0.82 / 1.00 | 0.221 / 0.498 |
| NIK-tofts (trained) | 1.05 / 0.91 | 0.75 / 0.94 | 0.91 / 0.91 | 1.56 / 1.11 | |
| NIK-tofts8 (trained) | 0.95 / 1.06 | 0.84 / 0.97 | 0.92 / 0.92 | 0.93 / 1.18 | |
| NIK-patlak (trained) | 0.97 / 1.00 | 0.97 / 0.89 | 0.87 / 0.89 | 0.73 / 1.01 | |

basis aif vs model-free aorta: time to peak 63.8 s vs 63.2 s; plateau / peak 0.23 vs 0.23