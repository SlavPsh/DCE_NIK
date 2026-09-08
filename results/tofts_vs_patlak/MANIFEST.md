# tofts_vs_patlak: comparison manifest (recovered Patlak setup, reused verbatim unless noted)

| item | phantom (xcat no-motion, z15) | in vivo (meas_p3_dce.dat, slices 18/19/21) |
|---|---|---|
| data | `XCAT-ERIC/results/simulation_results_20260816T210718.mat`, slice ZI=15 (labelGT idx 0, z-flip trap) | `grasp_pro_py/results_ref/slice_{18,19,21}.npz` (kdata_radial, b1, traj_grog) |
| motion pair | `…20260816T223443.mat` respPeriod 5.0, same layout, no prior NIK run | n/a |
| kinetic generator | ExtKety per label (`pkLUT`): cortex(23) ke 0.7 ve 0.9 vp 0.1 dt 8; medulla(24) ke 0.7 ve 0.8 vp 0.2 dt 8; aorta(36) ke 0 ve 0 vp 0.6 dt 15. ke in min^-1 (physiological; AIF rates are min^-1), dt in s (injection 12 s). **NOT Patlak-only.** | unknown |
| AIF (what Patlak sees) | `aif_xph.npz['aif_frame']`: normalized SIGNAL-domain enhancement, analytic Cosine4 -> SPGR (aif_gt.py) | `aif_slice{Z}.npz['aif_frame']`: normalized SIGNAL-domain ROI mean, 342 pts (aif_gate.py) |
| MR signal params | sim: TR 4.66 ms, alpha 18, TE 1.7, r1 3.5, B0 3T, T1_blood 1664 ms | header: TR 4.66 ms, flip 18, TE 1.68 -> identical SPGR |
| spokes | 7 angles/frame: TRAIN [0-4], VAL [5], TEST [6] (`xph_pipeline`) | Patlak used `keep_f25` (488/1710). here: train = keep_f25, VAL = complement ∩ v%10==8, TEST = complement ∩ v%10==9 (untouched). deviation documented. |
| model | `wire_ff_patlak`, n_free 0, width 768, k_sigma 2.5, FIX depth 12/w0 62/s0 15/k_freq 256/t_freq 32/t_sigma 1.5/coil_embed 8/env 0.75 | `wire_ff_patlak --patlak-free 0`, trainer defaults hidden 512 depth 12 |
| optimizer | Adam lr 1e-5 wd 3e-3, batch 16384, 40000 steps, ckpt every 2000 | Adam lr 1e-5, batch 65536, 40000 steps, ReduceLROnPlateau on heldout |
| ckpt rule | offline: best VAL k-space NMSE (`xph_eval.py`) | best heldout (val) loss. NOTE the established keep_f25 runs had NO early stopping (keep-file bug); both arms here use `--keep-heldout` with the explicit VAL set |
| seeds | 0,1,2 exist for F0 (`checkpoints/w768_ks2.5_s*`) -> REUSED as control | seed 0 only existed -> control RERUN for seeds 0,1,2 |
| eval | `reconstruct_pathC` (OVERSAMPLE 2), `kspace_nmse` 3 shells (<0.3,<0.7,rest) + 16 annuli (`xph_hikdiag`), `truth_at`, `rois`, l3 metrics | `consolidated.slice_ctx` ROIs, model-free ref, task_S osc/respfrac, held-out spokes |
| coil maps | true XCAT coils | `slice_{Z}.npz['b1']` (same as Patlak) |
| Tofts forward | DCE-NET @ 6c320ff via `dcenet_adapter` (`ext_tofts_sampled`, validated 2.4e-3 vs analytic) | same |
| AIF access rule | Tofts basis built from the SAME normalized signal AIF file Patlak uses (linear signal proxy, documented). analytic-cp+SPGR variant = secondary check only | same, linear signal proxy (absolute scale lost in the npz -> exact SPGR inversion impossible; documented) |
