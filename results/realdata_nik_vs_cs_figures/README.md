# Real in-vivo NIK-vs-CS figures (regenerated from cached results, no re-reconstruction)

CRITICAL CAVEAT: the real data has NO ground truth. The reference is a CS-f100 or model-free NUFFT
RECONSTRUCTION, so these figures measure CONSISTENCY / CS-likeness / denoising character, NOT accuracy.
This is exactly why the physical-XCAT experiment (with true GT) was built. Treat as supporting evidence.

- fig1_spoke_frontier.png  : HaarPSI vs spoke fraction (slice 13), CS vs NIK. CS is higher at every
  fraction BECAUSE the reference is CS-f100 itself (CS-likeness confound), not proof of higher accuracy.
- fig2_pk_cov.png          : PK-map INTER-SLICE CoV (Ktrans, vp; aorta/cortex/medulla/liver), NIK-F0/F2
  vs reconstruct-then-fit CS. Lower = more consistent. NIK CoV comparable-to-better (e.g. Ktrans aorta
  NIK-F0 13.2 vs CS-fit 20.8). Supports quantitative consistency, not physical PK (signal-domain coeffs).
- fig3_denoising.png       : temporal oscillation amplitude + respiratory-band power fraction (cortex).
  CS oscillates more than the reference but with LOW respiratory content -> broadband NOISE, not
  physiology. NIK's temporal representation is cleaner. (real, defensible temporal finding.)
- fig4_realdata_montage.png: slice-21 dynamic, model-free reference vs NIK(f25) vs CS(f25), scale-matched.
  CS sharpest/cleanest, NIK hazier -> CS >= NIK on real-data image quality (consistent with XCAT).
- fig5_realdata_aorta_curve.png: aorta ROI curve, model-free ref vs NIK vs CS.

Pairs with the physical-XCAT (true-GT) comparison in ../xcat_physical_nomotion_nik_vs_grasp/.
