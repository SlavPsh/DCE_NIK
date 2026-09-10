# dce_nik resume (2026-09-10). read this first in a new chat.

## how work runs now (luna decommissioned, helios has no claude)
laptop (claude, env `dce`) commits `jobs/queue/<n>.sh` (sbatch script) or `jobs/probe/<n>.sh` (inline, 60 s) -> `helios_agent.sh` (slurm defq, every 2 min) pulls, submits, commits `jobs/done/<n>`, `jobs/log/`, small results (json/md/png/csv) -> `python helios_fetch.py` on the laptop = git pull + jobs table + wandb (`dce_nik`) summaries in `results/_wandb/`. see `jobs/README.md`. training scripts log to wandb via `nik_wandb.py`. gpu quota: one 2g.24gb slice at a time (`QOSMaxGRESPerUser`), so arrays serialize.

## question
does a k-space inr (nik) beat compressed sensing (grasp pro, classic grasp v2) for radial dce-mri: images and contrast curves. phantom (xcat, truth) + in vivo (meas_p3_dce, slices 18/19/21, no truth).

## deliverables (the presentation)
- `results/xcat_physical_nomotion_nik_vs_grasp/NIK_vs_GRASPv2_combined_report.ipynb` (92 cells, 0 errors): nik vs grasp v2 + section 10 tofts vs patlak. lowercase, no dashes.
- `results/xcat_physical_nomotion_nik_vs_grasp/NIK_vs_CS_combined_report.ipynb` (61 cells): nik vs grasp pro (k12 cv-fair), sections 1 to 8 shared with the v2 notebook.
- `results/tofts_vs_patlak/REPORT.md`, `MANIFEST.md`, `REPRO.md`: tofts arm.
- `CONSOLIDATED_REPORT.md`, `report.md`: older text summaries, superseded where they disagree with the notebooks.

## figures (wandb, group tofts_figs, refreshed by every 00x_tofts job's eval stage)
https://wandb.ai/spshenov-university-of-amsterdam/dce_nik/groups/tofts_figs : per slice a run `figs_invivo_sl<Z>_<jid>` with panel (3 phases x arms + curves), roi overlay, all-seed curves vs model-free (raw scale), peak-frame difference, metric bars, test annuli. generator `tofts_figs_wandb.py`, queue `jobs/queue/003_tofts_figs.sh` (rename to rerun).
seen 2026-09-10 (sl21): tofts arms have the right curve shape but only ~60% of the model-free amplitude in the aorta peak and ~65% in the late medulla, patlak has the amplitude but the wrong medulla shape; the affine ruler hides the amplitude deficit, the raw-scale curves show it. sl18 (3k-step runs): the tissue amplitude deficit is nik-wide, patlak included (cortex and medulla plateaus at ~0.65 to 0.75 of model-free for all three nik arms, grasp v2 and pro on model-free); tofts arms additionally under-peak the aorta (0.57). hypothesis: early stopping on the static-dominated held-out k-space ruler (restored step 2000 in every run) selects under-fit dynamics; test = same runs without restore (final weights at 3k/8k) and compare raw-scale curves. relates to the nik_2901994 dynamics-metric flaw. roi overlay: cortex and medulla masks are 56 px each, the `liver` mask (1974 px) sits posterior over spine and muscle, not on the liver.

## established, with the number and where it lives
| fact | number | source |
|---|---|---|
| single fair setting vs grasp v2: 25 spf lam 0.25 on phantom (ideal-corner rule), carried to in vivo as nline 12; matched spokes (1720 / 1368) | | v2 nb sec 9, `v2_sweep/final_single_setting.json` |
| phantom images at that setting: tie | haarpsi 0.902 grasp vs 0.899 nik-sub16; sharpness_rel 1.004 vs 0.858 at equal bg noise; grasp psnr +1.6 db | `v2_sweep/image_metrics_panel.json` |
| phantom tissue curves: nik-free wins | cortex 0.011 vs 0.073, medulla 0.014 vs 0.051; aorta 0.081 vs 0.102 | `v2_sweep/frontier.json` |
| phantom bolus peak | nik +4 to +7%, grasp v2 -19% (lam 0.25), -7% (lam 0.02) | same |
| tuned grasp v2 (40 spf lam 0.02) beats nik on everything on the phantom | haarpsi 0.966, aorta 0.059 | v2 nb sec 9 |
| in vivo k80 pair (1368 spokes both): grasp v2 ahead on the model-free curve ruler | aorta 0.153 vs 0.234, cortex 0.084 vs 0.196 | `v2_sweep_invivo/v2_vs_nik_invivo_k80.json` |
| that ruler is biased: physical bound | aorta fwhm grasp v2 k80 53 s, nik 23 s, model-free 17 s, truth aif 12.5 s | `invivo_neutral_ruler.py` |
| nik vs grasp pro (k12 cv-fair) phantom | haarpsi 0.90 vs 0.87, psnr 37.4 vs 38.5, ssim 0.92 vs 0.95, aorta curve 0.08 vs 0.16 | cs nb sec 1, `l3_rebaseline*.csv` |
| in vivo temporal denoising real (cs oscillation = broadband noise) | nik 15 to 50x less oscillation | cs nb sec 2, `task_S*.json` |
| in vivo pk inter-slice consistency supports quantification | vp cov nik 2 to 5 vs cs 11 to 37 | `task2*.json` |
| output-coil beats input-coil on phantom, parity in vivo, 8x cheaper | psnr 37.2 vs 36.1 | cs nb sec 3 |
| tofts basis vs patlak, phantom no-motion | ssim 0.84 to 0.94, psnr +6 db, cortex 0.109 to 0.015, held-out k 13x; aorta dc offset +0.02 | `results/tofts_vs_patlak/phantom_nomotion.md` |
| tofts vs patlak, motion phantom | psnr +1.9 db, cortex 0.164 to 0.112, aorta worse | `phantom_motion.md` |
| tofts vs patlak, in vivo | kidney curves better, cortex/medulla identity broken (0.998 to 0.7 to 0.9), aorta peak 0.5 vs 0.9 (patlak by construction), held-out k 8% worse globally (k centre), better at |k|>0.19 | `invivo.md` |
| tofts rank 8 (forced) vs rank 12 (rule), in vivo sl21, 3 seeds, 2026-09-10 | test knmse 0.294 vs 0.311 (patlak 0.289): k-centre gap mostly closed (annulus 0 0.276 vs 0.294, patlak 0.270), best of the three on all 15 outer annuli; cortex/medulla affine 0.117/0.061 vs 0.139/0.075 (patlak 0.155/0.139); aorta affine 0.145 vs 0.178 (patlak 0.053, grasp v2 0.149); late corr -0.22 (model-free sign). basis rule not met at 8 (first-pass err 0.0285 > 0.01) | `results/tofts_vs_patlak/invivo_r8.md`, `jobs/queue/001_tofts_sl21_rank8.sh`, wandb `gnik_tofts8_sl21_s*` |
| tofts rank 8 at 3k steps, slices 18/19, 3 seeds each (queue 002, 2026-09-10) | test knmse 0.3265 / 0.3289 = patlak 0.3271 / 0.3291 (rank 12: 0.349 / 0.356), k-centre annulus 0.297 / 0.288 = patlak; cortex affine 0.082 / 0.088 (r12 0.095 / 0.117, patlak 0.152), medulla 0.056 / 0.050 (r12 0.062 / 0.069, patlak 0.139); aorta affine 0.156 / 0.156 (patlak 0.060 / 0.037), peak ratio 0.57 / 0.56; wall 990 s vs 8000 s, same quality -> 3k steps is the in vivo protocol from now on | `results/tofts_vs_patlak/invivo_r8.md`, `jobs/queue/002_tofts_sl18_19_rank8_3k.sh`, wandb `gnik_tofts8_sl1*` |
| every in vivo tofts_vs_patlak run (patlak, tofts r12, r8; slices 18/19/21; 3 seeds) restores the step 2000 weights | held-out mse rises monotonically from step 1000 (sl21 tofts 0.56 at 2k to 0.75 at 40k, patlak 0.62 to 0.88) while train falls; warmup 2000 pins the restored step; the reported wall_s 8000 is the 40k run, the evaluated model is ~400 s of training | `jobs/log/probe_002_restored_best.out` |

## retracted, do not resurrect
- "cs wins at every spoke fraction": pca basis leak (basis from all spokes). fair: level, nik ahead at 50%.
- "grasp v2 aif damping is structural": lam artifact; lam 0.02 fixes it.
- "grasp v2 6.7x worse than pro": 5 spf operating-point artifact (v2 nb section 1 still shows it, known, left as is).
- "nik and grasp use different b1": noise probe outside the body; in-body diff 1e-4.
- "nik overshoots cortex" and "cs over-reads cortex 12%": per-roi peak-normalization artifacts; use raw + affine.
- "render is a lossy readout, feed spokes to mcnufft": wrong, non-realizability is not readout loss.
- "output-coil under-reads bolus 14%": baseline-norm artifact; calibrated equal.
- "aif time grid mismatch 0.44 s": grids only; all axes share view_time.

## tested, negative, do not repeat
temporal tv on atoms (phi_tv) and on k-space (ktv21, l2,1): null. sense-forward a (image inr + b1): works, loses to sub16. gaussian activation: bad at every w0. rank knob: saturates (realized rank ~5). dcf_power 1.0: fits noise. per-frame data consistency: strictly hurts. binning-free pk differentiator: not real at 5 to 15 spf. spirit on nik: inert. post-hoc tv: no.

## open
- neutral in-vivo ruler: held-out spokes for cs need complex-valued grasp saves (`grasp_v2_real.py` saves magnitude). the one thing that would settle in-vivo curves.
- sense-b (k-space model + b1 factorization in k-space): only untested item on the image axis. kernel study done (r=2 taps capture 58%, factorization residual 8 db), training not run.
- tofts in vivo: rank 8 closes the k-centre gap on all three slices (2026-09-10). open: the amplitude deficit of the tofts arms (aorta peak ~0.57 of model-free, late medulla ~0.65, see figures), the basis rule threshold (0.01 rejects rank 8), rank 5 (patlak span + 2) as the next point on the curve, and the roi masks (cortex/medulla 56 px, `liver` mask not liver).
- v2 notebook section 1 still at 5 spf (~3 h to repoint).
- fwhm bound on slices 18/19/20 for grasp v2.

## pipeline facts that bite
- coords: in vivo x in [-1,1] (traj_norm*2), phantom kx in [-0.5,0.5] (`_dataset` doubles). radius bins: check the range.
- render: 2x oversample + rot180 with 1px roll (even grid) + support radius 1.0. three render bugs were found this way; any new image-vs-truth number goes through `recon_asserts.check_recon`.
- `--spoke-keep-file` alone disables early stopping and lr schedule; add `--keep-heldout` or `--spoke-heldout-file`.
- in vivo early stop: `--warmup-steps 2000` means the restored model is the step 2000 one in every run so far; the 40k steps and the plateau scheduler never matter. any in vivo number is a 2000 step model.
- agent: logs of running jobs must not be committed (a later pull rewrites the inode, slurm keeps writing to the old one); the agent defers by script name. changing `helios_agent.sh` needs `scancel -n helios-agent` + `sbatch` (slurm copies the script at submit).
- in-vivo run variants: `results_sl21_k80` is the fair one; `results_full_sl21_matched` had no early stop; `results_spoke_full_slice21` is random 70%.
- metrics: masked haarpsi/ssim, same window-averaged truth for every method, one global ls scale, roi mean on the same masks (`xph_common.rois`, `consolidated.slice_ctx`). curves vs model-free: raw + affine.
- time axis in vivo: everything derives from `view_time = v/1709` times 375 s.
- `srun --jobid=<session>` runs 2 tasks; add `--ntasks=1`. system python3 is 3.6; use the torch29 python.
- gpu: luna-01 only; mig 1g (phantom, 5 gb) / 4g (in vivo, 12 gb); 2g slices often unschedulable when 1g are full.

## reproduce
- v2 notebook: `build_gv2_nb.py` clones the cs notebook and patches sections 9; section 10 and the prose rewrite of 2026-09-08 were applied to the ipynb directly (rebuild from the builder would lose them; patch the builder before rebuilding).
- tofts: `results/tofts_vs_patlak/REPRO.md`.
- phantom nik: `xph_train.py` / `xph_eval.py`, `XPH_SIM=motion` for the motion sim. grasp v2 phantom: `xph_v2_sweep.py <G>` with `LAM_FRAC`. in vivo grasp v2: `grasp_v2/grasp_v2_real.py` (NLINE, LAM_FRAC, KEEP80).
