# aorta median intensity curves, DCE_Rerun grasp v2

roi: 147 px, z 18-33, centre (y=146, x=3), from **nufft** by time to peak in the 35 to 160 s window, largest blob per slice, 1 px erosion; identical roi applied to every tag
selected component: 16 slices, median cross-section 12 px (50 mm2), centroid wander 1.5 px, median ttp 41 s

| tag | baseline [1e-5] | peak [1e-5] | rel. peak | ttp [s] | plateau/peak |
|---|---|---|---|---|---|
| nufft | 0.71 | 2.12 | +198% | 37 | 0.09 |
| lam0.02 | 0.90 | 3.35 | +273% | 37 | 0.08 |
| lam0.08 | 0.81 | 2.71 | +236% | 37 | 0.12 |
| lam0.25 | 0.72 | 1.49 | +107% | 37 | 0.26 |
| glam0.02 | 0.89 | 3.31 | +272% | 37 | 0.09 |
| glam0.08 | 0.80 | 2.77 | +246% | 37 | 0.12 |
| glam0.25 | 0.75 | 1.60 | +112% | 37 | 0.27 |

files: aorta_roi_check.png (roi on anatomy, three views + peak/ttp maps), aorta_curves.png, aorta_curves.csv, aorta_roi.npz
