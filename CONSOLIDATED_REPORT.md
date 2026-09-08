# Consolidated batch report — NIK vs GRASP-Pro CS, kidney slices (f25)

Fixed protocol (WIRE d12/h512, FF k256/σ2.5 t32/σ1.5, coil-8, 40k steps, dcf off, env 0.75,
support 1.0, keep_f25 = same spokes as CS). Standing metric guards applied: corrected spatial
eval (2 rulers, signed bgE, diff-structure), realized rank = SVD of the complex recon,
ROI-averaged temporal curves vs streak-free model-free, oscillation reported with divergence.
All 19 configs analyzed, 0 missing.

## P1 — multi-slice replication (R16 + full on sl18/19/20, sl21 from P2)

**Spatial win is ruler-dependent, NOT "every ruler" as the slice-13 n=1 claimed.**
- **Ruler A (pre-contrast sharpness): R16 beats CS at every slice** (18: .585 vs .542; 19: .607 vs .575; 20: .592 vs .534; 21: .623 vs .594). Replicates.
- **Ruler B (non-enhancing temporal mean): CS beats R16 at every slice** (18: .838 vs .797; 19: .885 vs .810; 21: .877 vs .853). The win does not hold here.
- Background energy: at f25 both add streak energy (bgE > ref_bgE); R16 ≈ CS or slightly worse. Full-rank is worse on both rulers everywhere (realized rank ~20 → overfits streaks; the rank cage helps spatially).

**Temporal cortical bias does NOT robustly replicate as a CS failure.**
- The ~12% CS cortical over-plateau was a **slice-21 effect**. At sl18/19 CS matches the real cortex curve well (nRMSE 0.052 / 0.053); NIK-full is worse there (0.206 / 0.214).
- NIK-full never tracks the real cortex better than CS on sl18/19; it is smooth (osc ~0), so a confident over-fit, not noise.
- Medulla control: at sl21 NIK diverges from CS on **medulla too**, i.e. a global offset, not a cortex-specific correction (exactly the failure the control was meant to catch).

**P1 verdict: spatial R16>CS replicates on the pre-contrast ruler only; on the streak-free
mean ruler CS wins. The temporal cortical lead does not replicate — NIK does not correct a CS
cortex bias (which itself is mostly absent on 18/19).**

## P2 — rank pareto (slice 21)

**Headline: the rank knob saturates.** Nominal R8, R16, R32, R64 ALL collapse to **realized
complex rank 5** (= CS's K). Their spatial (ruler A ~0.62) and temporal (cortex nRMSE ~0.24)
are essentially identical. Full-rank realizes rank 23 but is spatially worse (0.543). There is
**no knee — the "curve" is a point.** The clustering IS the saturation finding.

**Does any config beat CS on both axes?** On (ruler A, cortex-nRMSE) the R8–R64 cluster sits
marginally up-and-left of CS (0.62/0.24 vs CS 0.59/0.28) — better on both. **But this is
contradicted by ruler B** (CS wins) and the cortex-nRMSE edge rides on sl21's anomalously poor
CS cortex fit. Not a robust both-axes win.

## P3 — Patlak / PK basis (slice 21)

AIF gate **PASSED** (TTP 65s, sharp 8s rise, monotonic, recirculation bump). Fixed Patlak basis
Φ=[AIF, ∫AIF, baseline] + F free dims implemented (new `wire_ff_patlak`).
- **vp/Ktrans maps are anatomically plausible** (kidneys/vessels bright in vp; cortex bright in
  Ktrans) and **agree with a conventional Patlak fit of the CS recon**: Ktrans r = 0.80 (F0) →
  0.89 (F2/F4); vp r = 0.65 → 0.75. Non-physical negative Ktrans fraction 18% (F0) → 8% (F4).
- **Spatial quality holds at ~CS** (ruler A 0.591 F0 ≈ CS 0.594) at realized rank **2** (F0).
- **Failure mode seen, as predicted:** pure Patlak (F0, rank 2) **under**-estimates the cortex
  late plateau (0.43 vs real ~0.62) — the hard prior cannot represent cortical washout; adding
  free dims pulls it back up (F2 0.62, F4 0.86). So rank 2–3 smears like a too-hard prior.

**P3 verdict: the Patlak basis produces clinically-shaped, CS-consistent Ktrans/vp maps at rank
2–5 and holds spatial quality, but the hard (F0) prior misfits cortical washout; it does not
beat CS on the spatial rulers.**

## Matrix (vs CS at f25; + better, − worse, ~ within tol)

| slice | cfg | realized rank | ruler A | ruler B | cortex nRMSE | medulla nRMSE |
|---|---|---|---|---|---|---|
| 18 | R16 | 5 | + | − | − | ~ |
| 18 | full | 21 | − | − | − | + |
| 19 | R16 | 5 | + | − | − | ~ |
| 19 | full | 20 | − | − | − | ~ |
| 20 | R16 | 5 | + | − | + | + |
| 20 | full | 22 | ~ | − | + | + |
| 21 | R8 | 5 | + | − | + | + |
| 21 | R16 | 5 | + | − | + | + |
| 21 | R32 | 5 | + | − | + | + |
| 21 | R64 | 5 | + | − | + | + |
| 21 | full | 23 | − | − | + | + |
| 21 | PK_F0 | 2 | ~ | − | + | − |
| 21 | PK_F2 | 4 | ~ | − | + | − |
| 21 | PK_F4 | 5 | + | − | + | ~ |
| 21 | full_f100 | 36 | + | + | + | ~ | (non-comparative: all-spoke, degenerate)

**No config is `+` on both spatial rulers at f25.** Every NIK config that wins ruler A loses
ruler B. The only all-`+` row is full_f100, which is non-comparative (uses all spokes).

## Anomalies / metric-guard flags

- **Slice 20 is unreliable**: reference bgE 0.479 (vs ~0.09 on 18/19/21) and an empty aorta ROI
  → its ruler/segmentation is degenerate. Its temporal "+" for NIK should be discounted.
- **full_f100** included only as the non-comparative all-spoke reference (A 0.807, B 0.929);
  confirms f25 is the discriminative comparison point.
- **medulla divergence** accompanies cortex divergence at sl21 → NIK's temporal difference is a
  global offset, not a cortex-specific correction (the control fired).
- Realized rank read from complex-recon SVD throughout; checkpoint `rank` field never used.
- Per-voxel model-free avoided (streak-dominated); all temporal curves ROI-averaged.

Figures: `P1_replication.png`, `P2_rank_pareto.png`, `P3_patlak_maps.png`,
`gate_readout_slice21.png`, `aif_gate_slice21.png` (all datetime-prefixed in figures/).
