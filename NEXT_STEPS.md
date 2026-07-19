# DCE_NIK — next steps (rewritten 2026-07-18 after the render-bug finding)

## HEADLINE: the "NIK is blurry" story was a RENDER BUG, now fixed
`reconstruct_cartesian` masks |k| > support_radius at render time. It must equal the
sampled-disk radius ~1.0 (model coords = 2*traj_norm, data reaches |x|=0.997). It was set
to **0.5** (the traj_norm max -- a 2x units slip), which zeroed the OUTER HALF of sampled
k-space = the high-freq half -> ~2x blur.
- Proof (render_support_check.py, SAME model both ways): HF-ratio 7.5e-7 -> 8.5e-6,
  i.e. **8% -> 93% of CS-100** high-freq. Visually: blurry -> sharp (support_check.png).
- FIXED: train_grasp_nik --support-radius default 0.5 -> 1.0 (committed).
- This INVALIDATES a chain of earlier conclusions (all render artifacts): "NIK loses to CS
  spatially", the "held-out paradox", the 30x power-spectrum deficit, and the urgency of
  hash-grid / FF-bandwidth / SENSE as BLUR fixes. NIK was fitting high-|k| all along.

## Corrected picture (post-fix, freq/depth sweep, rank-16, slice 13)
- Perceptual vs CS-100: DISTS 0.09 -> **0.024**, HaarPSI 0.92 -> **0.96** (CS-70 = 0.018 / 0.976).
  NIK is now a near-spatial-PEER to CS; residual gap = minor streaking/grain, NOT blur.
- Temporal win STANDS (R-sweep, unaffected by render): R=5 swing 39% (=CS rank-5 cap),
  R>=6 swing 43-45% (dynamics CS's rank-5 can't represent). Sweet spot R~10-20.
- freq/depth barely move quality -> the render fix was the lever, recipe is settled:
  wire_ff_subspace rank ~16, k_sigma 1.5-2.5, w0 62-80, depth 12, dcf0, env0.75.

## In flight
- **Re-baseline at support=1.0** (results_rebase_*, waiter): full-rank + R=5/10/16/20 vs
  CS-70/CS-100, unified table (held-out/swing/nav/HF/DISTS/HaarPSI). Confirms NIK~=CS + temporal win.

## OPEN next steps (ranked, corrected)
1. **Step-response / sharp-bolus test** -- NOW THE TOP ITEM. The temporal claim rests on the
   navigator NIK trains on (circular). Inject a known temporal signal / find the sharp bolus
   and show NIK resolves it -> distinguishes real temporal resolution from smooth interpolation.
2. **Spoke-reduction frontier** -- the ACTUAL GOAL ("fewer spokes"). Everything is at 70%.
   Now reachable since spatial quality is fixed. Push below 70%, find where NIK holds and CS breaks.
3. **Aorta ROI** for dynamics -- cheap eval upgrade (sharp arterial bolus = best temporal test).
4. **Multi-slice generalization** -- everything is slice 13.
5. Residual grain (minor, DISTS 0.024 vs CS 0.018) IF worth chasing: coils-as-output / SENSE
   forward (over-determination via fixed coil maps), or the temporal low-rank we already have.
6. Hardening: end-frame DC instability (k=0 dip at t=1).

## DE-PRIORITIZED (were blur fixes; blur was the render)
- Hash-grid encoding, higher FF bandwidth, radial |k|-warp (wire_ff_res_radial): all aimed
  at blur that was a render artifact. freq/depth sweep confirms they don't move quality much.
- SENSE / coils-as-output: still valid but only for the small residual grain, not blur.

## Built + committed (reference)
- WIRE_FF_SUBSPACE (factorized low-rank, rank knob) + warmstart_phi (PCA init).
- WIRE_FF_RES_RADIAL (|k|-warp diagnostic). Working model wire_ff_res untouched.
- Checkpointing (--resume, self-cleaning). Render fix (support_radius 1.0).
- Tooling: sweep_aggregate / freq_aggregate / rebase_aggregate / cs_heldout_loss (didn't validate).

## Constraints (user decisions)
- Stay in k-space for now (no image-domain learned priors).
- Don't touch the autoresearch orchestrator. No Co-Authored-By trailer in commits.
