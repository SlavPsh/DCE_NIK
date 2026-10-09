# artifacts at contrast arrival, p3 slice 18; arrival 55 s, aif peak 64 s; windows pre 0-40 s, arrival 50-75 s, plateau 90-150 s, late 200-375 s

## (1) artifact level per window (air energy / fine-scale fraction / temporal roughness)

| arm | pre | arrival | plateau | late |
|---|---|---|---|---|
| model-free 31-spoke | 0.440 / 0.306 / 0.250 | 0.417 / 0.319 / 0.272 | 0.355 / 0.293 / 0.261 | 0.350 / 0.279 / 0.256 |
| patlak span + 13 free atoms | 0.149 / 0.231 / 0.013 | 0.148 / 0.244 / 0.016 | 0.126 / 0.229 / 0.014 | 0.124 / 0.219 / 0.013 |
| sub16 (wd 3e-3) k100 | 0.145 / 0.260 / 0.097 | 0.138 / 0.261 / 0.101 | 0.123 / 0.244 / 0.098 | 0.121 / 0.244 / 0.097 |
| tofts8 k100 | 0.087 / 0.224 / 0.001 | 0.132 / 0.257 / 0.044 | 0.087 / 0.221 / 0.005 | 0.082 / 0.207 / 0.002 |
| NIK-free k100 | 0.169 / 0.240 / 0.027 | 0.165 / 0.244 / 0.020 | 0.142 / 0.235 / 0.018 | 0.138 / 0.221 / 0.018 |
| GRASP all spokes | 0.129 / 0.201 / 0.030 | 0.132 / 0.216 / 0.037 | 0.107 / 0.201 / 0.029 | 0.104 / 0.189 / 0.029 |
