# tofts basis span vs trained arms, in vivo slice 21, k80 (reference = model-free 31-spoke nufft, roi enhancement, baseline subtracted)

ratios are amplitude relative to the model-free curve: first-pass peak (20 to 210 s) and washout mean (t > 200 s). projection = least squares of the model-free curve onto the first R atoms at the frame times (what a perfect fit inside the span could reach).

| curve | aorta peak / washout | cortex peak / washout | medulla peak / washout | liver peak / washout | body voxel err (all / dynamic part) |
|---|---|---|---|---|---|
| projection rank 3 | 0.99 / 1.02 | 1.00 / 0.90 | 0.73 / 0.93 | 0.68 / 0.94 | 0.264 / 0.604 |
| projection rank 5 | 1.00 / 1.02 | 1.00 / 1.01 | 0.83 / 1.00 | 0.77 / 0.99 | 0.251 / 0.572 |
| projection rank 8 | 1.00 / 1.02 | 0.96 / 1.00 | 0.93 / 1.00 | 0.81 / 0.99 | 0.244 / 0.554 |
| projection rank 12 | 1.00 / 0.99 | 0.97 / 1.00 | 0.94 / 1.00 | 0.81 / 0.98 | 0.238 / 0.542 |
| NIK-tofts (trained) | 0.64 / 0.82 | 0.48 / 0.68 | 0.65 / 0.71 | 1.60 / 1.64 | |
| NIK-tofts8 (trained) | 0.76 / 0.96 | 0.54 / 0.73 | 0.67 / 0.75 | 1.31 / 1.46 | |
| NIK-patlak (trained) | 0.99 / 1.02 | 0.81 / 0.84 | 0.69 / 0.90 | 0.79 / 1.11 | |

basis aif vs model-free aorta: time to peak 63.8 s vs 63.2 s; plateau / peak 0.23 vs 0.23