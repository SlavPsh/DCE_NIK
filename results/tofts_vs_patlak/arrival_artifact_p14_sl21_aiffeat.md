# artifacts at contrast arrival, p14 slice 21; arrival 40 s, aif peak 51 s; windows pre 0-40 s, arrival 35-60 s, plateau 90-150 s, late 200-390 s

## (1) artifact level per window (air energy / fine-scale fraction / temporal roughness)

| arm | pre | arrival | plateau | late |
|---|---|---|---|---|
| model-free 31-spoke | 0.603 / 0.344 / 0.312 | 0.566 / 0.348 / 0.326 | 0.464 / 0.307 / 0.303 | 0.485 / 0.310 / 0.293 |
| patlak span + 13 free atoms | 0.227 / 0.241 / 0.018 | 0.211 / 0.238 / 0.023 | 0.177 / 0.214 / 0.015 | 0.183 / 0.215 / 0.015 |
| sub16 (wd 3e-3) k100 | 0.189 / 0.239 / 0.118 | 0.184 / 0.235 / 0.118 | 0.159 / 0.217 / 0.101 | 0.164 / 0.222 / 0.103 |
| tofts8 k100 | 0.145 / 0.234 / 0.003 | 0.180 / 0.241 / 0.057 | 0.119 / 0.206 / 0.007 | 0.118 / 0.206 / 0.006 |
| NIK-free k100 | 0.208 / 0.241 / 0.015 | 0.192 / 0.237 / 0.017 | 0.151 / 0.202 / 0.009 | 0.160 / 0.207 / 0.009 |
| GRASP all spokes | 0.201 / 0.203 / 0.032 | 0.179 / 0.204 / 0.041 | 0.137 / 0.170 / 0.031 | 0.147 / 0.170 / 0.030 |
