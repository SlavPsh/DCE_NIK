# per-slice intensity: before the recon vs after it

the profile is the per-slice median inside that slice's own body mask; `prep` is preview.npy, the per-slice time-average straight after the kz ifft, before any reconstruction. the smooth trend (coil sensitivity along z, slab profile, anatomy) is separated from the slice-to-slice residual, because only the residual can stripe a coronal cut: a smooth trend reads as shading.

| profile | min | max | max/min | residual std % | residual max % | median jump % | max jump % | jumps > 15% |
|---|---|---|---|---|---|---|---|---|
| prep (post kz-fft) | 1.059e-05 | 3.105e-05 | 2.93 | 0.5 | 3.2 | 3.4 | 9.1 | 0 |
| nufft | 5.369e-06 | 1.639e-05 | 3.05 | 1.0 | 4.3 | 4.0 | 12.1 | 0 |
| lam0.02 | 9.657e-06 | 3.079e-05 | 3.19 | 1.0 | 4.2 | 3.8 | 12.8 | 0 |
| lam0.08 | 9.591e-06 | 3.049e-05 | 3.18 | 1.1 | 4.8 | 4.1 | 16.0 | 1 |
| lam0.25 | 8.921e-06 | 2.937e-05 | 3.29 | 3.6 | 12.2 | 4.8 | 21.6 | 4 |
| glam0.02 | 9.611e-06 | 3.076e-05 | 3.20 | 1.1 | 4.8 | 4.1 | 13.3 | 0 |
| glam0.08 | 9.068e-06 | 3.045e-05 | 3.36 | 0.9 | 4.4 | 3.9 | 15.4 | 1 |
| glam0.25 | 7.132e-06 | 2.968e-05 | 4.16 | 1.2 | 4.3 | 3.9 | 23.2 | 2 |

b1 den median over z: 1.060 to 1.356 (b1 is normalised per slice in prep, so the coil combine carries a per-slice scale of its own)

files: zprofile.png, zprofile.csv
