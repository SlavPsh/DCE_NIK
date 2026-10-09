# artifacts at contrast arrival, p3 slice 19; arrival 55 s, aif peak 64 s; windows pre 0-40 s, arrival 50-75 s, plateau 90-150 s, late 200-375 s

## (1) artifact level per window (air energy / fine-scale fraction / temporal roughness)

| arm | pre | arrival | plateau | late |
|---|---|---|---|---|
| model-free 31-spoke | 0.472 / 0.309 / 0.241 | 0.443 / 0.331 / 0.271 | 0.377 / 0.303 / 0.255 | 0.377 / 0.287 / 0.248 |
| patlak span + 13 free atoms | 0.146 / 0.230 / 0.012 | 0.136 / 0.248 / 0.015 | 0.120 / 0.227 / 0.012 | 0.122 / 0.218 / 0.011 |
| sub16 (wd 3e-3) k100 | 0.143 / 0.258 / 0.103 | 0.133 / 0.263 / 0.104 | 0.118 / 0.245 / 0.098 | 0.116 / 0.243 / 0.096 |
| tofts8 k100 | 0.084 / 0.203 / 0.000 | 0.105 / 0.239 / 0.028 | 0.079 / 0.208 / 0.003 | 0.079 / 0.193 / 0.001 |
| NIK-free k100 | 0.174 / 0.238 / 0.025 | 0.162 / 0.253 / 0.027 | 0.142 / 0.232 / 0.024 | 0.142 / 0.219 / 0.024 |
| GRASP all spokes | 0.131 / 0.203 / 0.021 | 0.121 / 0.221 / 0.024 | 0.100 / 0.214 / 0.020 | 0.102 / 0.198 / 0.019 |
