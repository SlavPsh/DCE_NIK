# Centering / convention audit (sibling-bug sweep for the 1px even-grid rot180)

Guard: `recon_asserts.py` (A1 geometry, A2 point-source round trip, A3 Parseval, A4 D4 orientation, A5 shape/dtype/finite; all RAISE). A2 offsets below are the measured delta round-trip peak error (b2_audit.py).

## B5 table
| # | site (file:line) | op | grid parity | convention | A2 offset | verdict | in a deliverable? |
|---|---|---|---|---|---|---|---|
| 1 | xph_img_eval.py:24 | rot180 `[::-1,::-1]` (NIK image lane: sub5/12/16, free) | even 220 | rotated about (N-1)/2=109.5 | OLD (-1,-1) | **WAS WRONG -> FIXED** to `roll(flip,1)` (about N/2) | YES (all image-lane spatial). Fixed. |
| 2 | xph_eval.py:27 | rot180 `[::-1,::-1]` (NIK F0, reconstruct_pathC) | even 220 | same as #1 | (-1,-1) (same primitive) | **WRONG, NOT YET FIXED** | YES (F0 spatial + PK). Fix in Step 2. |
| 3 | xph_grasp_run.py:8 | rot180 `[::-1,::-1]` (old GRASP lib path) | even 220 | same as #1 | (-1,-1) | WRONG but PATH UNUSED (current GRASP = NUFFT/ksweep, orient=id) | NO |
| 4 | xph_grasp_nufft_run.py / ksweep | orient() search -> "id" (GRASP baseline) | even 220 | no rot applied | NUFFT (0,0) | CORRECT | YES (GRASP). OK |
| 5 | grasp_pro_py/fftc.py:16 ifft2c_mri | fftshift/fft/fftshift per axis | even 220 | DC at N//2 | (0,0), parseval 1e-16 | CORRECT for even; LATENT for odd N (fftshift!=ifftshift) | YES. OK (all grids even) |
| 6 | grasp_pro_py/fftc.py:55 crop_img | central crop (nx-nxnew)//2 | even diff (40,68) | assumes even diff | (0,0) | CORRECT | YES. OK |
| 7 | nik_adapter.py:81 cartesian_grid | `(arange-N//2)/(N//2)` | even 220 | DC at N//2 | (0,0) via #5 | CORRECT | YES. OK |
| 8 | xph_common.py:10 _embed | transpose + center-embed at (N-152)//2=34 | even | center 110, applied to truth+labels alike | n/a (consistent) | CORRECT (consistent both sides) | YES. OK |
| 9 | nik_recon.py nufft2d_recon (in-vivo render) | cufinufft type1 adjoint, kx*pi, ramp dcf; fft2d_uniform = ifftshift/ifft2/fftshift; NO array-flip rot180 | even | proper centered transform | **MEASURED (P1): peak at N-p exactly, off-by-one (0,0)** | **CENTERED, no 1px bug** (180-rot is a benign sign-convention artifact; sub-pixel error = 0) | YES (in-vivo). Verified re: this bug class |

## P2 guards
`grasp_pro_py/fftc.py`: `ifft2c_mri`/`fft2c_mri`/`_fwd_axis`/`_adj_axis` now raise `ValueError` on ODD dim (fftshift-both-sides mis-centers for odd N); `crop_img` raises on ODD crop difference. No effect on the even-220 pipeline; fails loudly if any odd grid ever reaches it.

## B3 coordinate conventions (boundaries)
- Trajectory sign: SIGN=-1, negated trajs, 2*pi (xph_grasp_nufft); NIK coords kx=2*traj. NUFFT round trip (0,0) and recon registers with NO flip (A4 clean) -> sign/order consistent. The historical 180deg kx/ky negation is absorbed by the rot180 alignment; after the centering fix it lands at (0,0).
- Normalized-k range: kx=2*traj in [-1,1]; cartesian_grid in [-1,1); support_radius=1.0 (mask |coord|>1). Consistent.

## B4 ROI grid vs recon
ROIs defined on the truth/label grid (RO=220, truth orientation); recons align to the same. The +1px recon shift perturbs ROI-median CURVES negligibly (curve-NRMSE from a 1px shift: aorta 0.0005, cortex 0.0098, medulla 0.0049; peaks unchanged), because ROIs are large (33-663 px). So temporal + PK results are ROBUST to the shift; only the SPATIAL metrics were affected (SSIM +0.16 after fix). 10.2% of body pixels change >5% range under 1px (edge density).

## Conclusion
Exactly ONE new sibling found: NIK-F0 render (site #2, xph_eval.py:27), same rot180 primitive, offset (-1,-1). All other centering ops verified correct by A2. The bug class is isolated to the image-space array-flip rot180. Curves/PK are robust to the 1px shift; only spatial metrics need re-baselining. In-vivo path is code-clean for this bug (no array-flip rot180) but A2 on real data is deferred to Step 3.
