# Sanity checks — physical no-motion NIK vs GRASP-Pro

| # | check | result | evidence |
|---|---|---|---|
| 1 | XCAT physical truth is the only accuracy reference | **PASS** | `GroundTruth.img` (SPGR signal); no reconstruction used as truth |
| 2 | Task-4 basis-matched `sim.npz` NOT used | **PASS** | physical file `...210718.mat` only |
| 3 | NIK and GRASP-Pro receive identical source measurements | **PASS** | same train-spoke k-space (from one z-FFT per slice), same b1, same traj |
| 4 | Input spoke sets identical | **PASS** | both reconstruct from train angles {0-4} (5/frame) |
| 5 | Noise realization identical | **PASS** | single fixed k-space from the one sim file |
| 6 | Trajectory units correct | **PASS** | geometry validated (static corr 0.953); sign convention resolved |
| 7 | Coil ordering correct | **PASS** | same `b1[zi]` for both methods |
| 8 | Spatial grids aligned | **PASS** | 220 grid; recon→truth transform fixed per operator (NIK rot180 / GRASP identity), each verified by static-image corr |
| 9 | NIK and truth evaluated at GRASP-Pro frame times | **PASS** | native frame centres; NIK queried there; truth sampled there |
| 10 | Magnitude comparison uses one common truth-derived scale | **PASS** | one global scalar per method (fit on body), applied to all frames |
| 11 | Anatomical mask identical | **PASS** | shared body mask (`labels>0`) |
| 12 | ROIs identical | **PASS** | aorta 36 / cortex 13 / medulla 37 for all methods |
| 13 | NIK truth never used for tuning | **PASS** | width/k_sigma + checkpoints by **validation kNMSE only** |
| 14 | GRASP-Pro truth never used for tuning | **PASS** | established weights (K=5, TV 0.001/0.0005), unchanged |
| 15 | NIK test spokes untouched until final eval | **PASS** | test = angle {6}; used only in the final k-space metric |
| 16 | Hyperparameter grids defined before truth evaluation | **PASS** | widths {256,512,768}, k_sigma {1.75,2.5,3.5} prespecified |
| 17 | No previous results overwritten | **PASS** | all outputs under `results/xcat_physical_nomotion_nik_vs_grasp/` |
| 18 | No motion data entered this experiment | **PASS** | only the respPeriod-"N/A" no-motion file loaded |

Notes / documented asymmetries (see report §15): GROG→cufinufft-NUFFT operator substitution for GRASP-Pro;
NIK temporal rank (F0=3 / sub5=5 / free=∞) differs from GRASP K=5 (R=16 not run); GRASP frame-based → no
exactly-comparable held-out k-space; single slice (zi=10); PK-map comparison deferred (no fitter exists).
