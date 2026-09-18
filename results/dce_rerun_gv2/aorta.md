# aorta median intensity curves, DCE_Rerun grasp v2

roi: tracked lumen disc r=4 px (17 mm across), 490 px over z 1-10 (10 slices), centre (y=69, x=109), centroid wander 3.5 px, median ttp 48 s
found on **nufft** (method-neutral) by a vessel-disc matched filter (r=5 px, fill >= 0.7) over voxels above 0.45 of the 99.9th percentile peak with ttp <= 70 s and more than 15 px inside the body, tracked in z, then the earliest median ttp of 2 tube candidates; identical roi for every tag
a fixed lumen disc is used rather than a brightness-thresholded mask, which would bias the median toward the brightest voxels and inflate the peak

| tag | baseline [1e-5] | peak [1e-5] | rel. peak | ttp [s] | plateau/peak |
|---|---|---|---|---|---|
| nufft | 1.17 | 7.14 | +513% | 48 | 0.22 |
| lam0.02 | 2.43 | 14.08 | +480% | 44 | 0.21 |
| lam0.08 | 2.46 | 12.96 | +427% | 44 | 0.22 |
| lam0.25 | 2.29 | 10.59 | +363% | 48 | 0.25 |
| glam0.02 | 2.45 | 14.16 | +479% | 44 | 0.21 |
| glam0.08 | 2.47 | 13.49 | +446% | 44 | 0.21 |
| glam0.25 | 2.19 | 10.71 | +390% | 44 | 0.24 |

tube candidates (the selected one is the earliest):

| centre (y, x) | slices | z range | disc fill | wander px | median ttp s | median peak |
|---|---|---|---|---|---|---|
| (68, 113) **selected** | 10 | 1-10 | 1.00 | 3.5 | 48.0 | 6.408e-05 |
| (111, 95) | 13 | 14-26 | 0.88 | 1.6 | 58.9 | 4.021e-05 |

files: aorta_roi_check.png (roi on anatomy, axial + both through-plane cuts + coverage and timing maps + centre track), aorta_curves.png, aorta_curves.csv, aorta_roi.npz
override with AORTA_YX="y,x" and ZRANGE="z0,z1" if the overlay shows the disc off the vessel
