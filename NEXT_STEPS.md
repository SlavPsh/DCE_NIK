# DCE_NIK — next steps (as of 2026-07-16)

## In flight (check these first)
- **Port verification** (train_grasp_nik winning recipe on slice 13, MIG job).
  Last seen held-out ~0.367 and dropping toward the autoresearch target **0.323**.
  When done: confirm held-out ~0.32 + sensible swing/nav-corr, then trust the port.
  (results_nik_verify/, waiter-based; resume-capable via --resume if it times out.)
- **VFA 3D-joint recon** (grasp_pro_py/recon_vfa_3d.py). Compare vs the 2D per-slice
  baseline (results_vfa/vfa_compare.png). Checks PF-recovery + 3D-joint quality.

## Just built (committed, working model untouched)
- `WIRE_FF_SUBSPACE_KXY_COIL_T_REIM` in nik_model.py — factorized low-rank NIK:
  k(x,y,t,coil) = sum_r A_r(x,y,coil)*Phi_r(t). A_r = WIRE+FF+residual backbone,
  Phi_r = small SIREN on FF(t), continuous in t. **Rank R = temporal-DoF knob.**
- `warmstart_phi()` helper — init Phi from the k-center PCA basis.
- Wired: `train_grasp_nik --model wire_ff_subspace --rank R [--phi-hidden/-depth/-w0]`.
  Default model stays wire_ff_res (byte-identical).

## Next increments for the factorized model (in order)
1. **Wire warm-start into the training loop**: compute the K=R k-center PCA temporal
   basis from the navigator at train time, call warmstart_phi() before the main loop.
   NEEDED because the bilinear A*Phi is non-convex (scale/sign ambiguity) -> unstable
   without a good Phi init. Do this BEFORE the sweep.
2. **R-sweep** {5, 8, 12, 16, 20} vs the full-rank joint wire_ff_res as reference.
   Hypothesis (from the DoF analysis): a sweet spot R>5 matches CS spatially (denoised)
   while keeping NIK's temporal-DoF edge. DO NOT fix R=5 (that = CS's cap).
3. **Eval upgrades** (adopt regardless):
   - **aorta ROI** for dynamics (sharp arterial bolus = best temporal-resolution test)
   - **CS held-out MSE** as the bar (anchor NIK's held-out number vs CS on same spokes)

## Bigger picture (ranked, after the factorized model)
1. Factorized low-rank (above) — attacks the spatial grain, the one measured gap vs CS.
   (Alternative if it's too rigid: E9 soft nuclear-norm penalty on the joint model.)
2. **Step-response / sharp-bolus test** — credibility: the temporal claim currently
   rests on the navigator NIK trains on (circular). Distinguish real resolution from
   smooth interpolation.
3. **Spoke-reduction frontier** — the actual goal ("fewer spokes"); everything so far is
   at 70%. Run once the quality gap is closed.
4. Hardening: end-frame DC instability (k=0 dip at t=1), multi-slice generalization,
   noise-floor DoF cutoff.

## Constraints (user decisions)
- **Stay in k-space** for now (no image-domain learned priors; low-rank is k-space-native
  and provably the same constraint via unitary Fourier).
- Don't touch the autoresearch orchestrator.
- No Co-Authored-By trailer in commits.
