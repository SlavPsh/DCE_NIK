# aorta median intensity curves, DCE_Rerun grasp v2

roi: tracked lumen disc r=4 px (17 mm across), 931 px over z 13-31 (19 slices), centre (y=111, x=95), centroid wander 2.8 px, median ttp 59 s
found on **nufft** (method-neutral) by a vessel-disc matched filter (r=5 px, fill >= 0.7) over voxels above 0.45 of the 99.9th percentile peak with ttp <= 70 s and more than 15 px inside the body, tracked in z, then the earliest median ttp of 1 tube candidates; identical roi for every tag
a fixed lumen disc is used rather than a brightness-thresholded mask, which would bias the median toward the brightest voxels and inflate the peak

| tag | baseline [1e-5] | peak [1e-5] | rel. peak | ttp [s] | plateau/peak |
|---|---|---|---|---|---|
| nufft | 0.49 | 4.42 | +794% | 59 | 0.15 |
| lam0.02 | 0.87 | 8.49 | +878% | 59 | 0.17 |
| lam0.08 | 0.88 | 7.70 | +773% | 59 | 0.19 |
| lam0.25 | 0.87 | 6.16 | +611% | 59 | 0.22 |
| glam0.02 | 0.87 | 8.52 | +884% | 59 | 0.17 |
| glam0.08 | 0.88 | 7.79 | +783% | 59 | 0.19 |
| glam0.25 | 0.90 | 6.53 | +626% | 59 | 0.22 |

tube candidates (the selected one is the earliest):

| centre (y, x) | slices | z range | disc fill | wander px | core px | median ttp s | median peak |
|---|---|---|---|---|---|---|---|
| (111, 95) **selected** | 19 | 13-31 | 0.77 | 2.8 | 7 | 58.9 | 3.957e-05 |

files: aorta_roi_check.png (roi on anatomy, axial + both through-plane cuts + coverage and timing maps + centre track), aorta_curves.png, aorta_curves.csv, aorta_roi.npz
override with AORTA_YX="y,x" and ZRANGE="z0,z1" if the overlay shows the disc off the vessel
