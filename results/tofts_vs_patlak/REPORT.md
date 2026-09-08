# nik_tofts_subspace vs patlak: results (2026-09-08)

model: `WIRE_FF_TOFTS_KXY_COIL_T_REIM` (nik_model.py), fixed real temporal atoms = orthonormalized patlak span [aif, int aif, 1] + svd of an aif-conditioned extended-tofts dictionary (dce-net forward, ke 0.05 to 3 /min log-uniform, ve 0.02 to 0.9, vp 0.005 to 0.6, dt 0 to 20 s) projected outside that span. rank = smallest of {5,8,12,16} with <=1% projection error on fresh curves, first pass and washout: 16 phantom, 12 in vivo. coefficient net, coil embedding, loss, optimizer, k-space training, render, eval: identical to the patlak arm. no learned rate. patlak checkpoint converts losslessly (verify.json).

## headline

| dataset | metric | patlak | tofts | delta | cs ref (grasp v2 25 spf lam0.25, not truth) |
|---|---|---|---|---|---|
| phantom no-motion | ssim (68 fr) | 0.8384 ± 0.0094 | 0.9367 ± 0.00025 | +0.09823 | 0.9199 |
| phantom no-motion | psnr db | 30.98 ± 0.39 | 37.14 ± 0.064 | +6.168 | 38.34 |
| phantom no-motion | haarpsi | 0.73 ± 0.01 | 0.905 ± 0.0019 | +0.175 | 0.9023 |
| phantom no-motion | aorta curve nrmse | 0.05156 ± 0.0051 | 0.07606 ± 0.0029 | +0.0245 | 0.1055 |
| phantom no-motion | cortex curve nrmse | 0.1085 ± 0.00043 | 0.01494 ± 0.00021 | -0.09359 | 0.07321 |
| phantom no-motion | medulla curve nrmse | 0.1308 ± 0.00029 | 0.0104 ± 0.00048 | -0.1204 | 0.05057 |
| phantom no-motion | aorta peak err % | 8.961 ± 3.7 | 3.794 ± 0.54 | -5.167 | -22.71 |
| phantom no-motion | held-out (test angle) k-space nmse | 0.00875 ± 4.5e-05 | 0.000693 ± 3.5e-05 | -0.008057 | nan |
| phantom motion | ssim (68 fr) | 0.7808 ± 0.0047 | 0.7988 ± 0.0021 | +0.01802 | 0.8279 |
| phantom motion | psnr db | 30.69 ± 0.071 | 32.6 ± 0.091 | +1.906 | 34.1 |
| phantom motion | haarpsi | 0.6532 ± 0.0027 | 0.6723 ± 0.0024 | +0.01907 | 0.7104 |
| phantom motion | aorta curve nrmse | 0.06382 ± 0.0033 | 0.09873 ± 0.0036 | +0.03491 | 0.114 |
| phantom motion | cortex curve nrmse | 0.1643 ± 6.3e-05 | 0.1122 ± 3.6e-05 | -0.0521 | 0.1404 |
| phantom motion | medulla curve nrmse | 0.1356 ± 0.00017 | 0.05408 ± 0.00041 | -0.08156 | 0.0758 |
| phantom motion | aorta peak err % | -0.7756 ± 0.45 | -7.332 ± 1.1 | -6.557 | -24.49 |
| phantom motion | held-out (test angle) k-space nmse | 0.01347 ± 9.5e-05 | 0.009441 ± 0.00044 | -0.004029 | nan |
| in vivo sl18 | aorta vs model-free (affine nrmse) | 0.05971 ± 0.0087 | 0.1864 ± 0.0021 | +0.1267 | 0.1402 (grasp v2 f25, 122 fr) |
| in vivo sl18 | cortex vs model-free | 0.1523 ± 0.00071 | 0.09547 ± 0.0024 | -0.05682 | 0.05866 (grasp v2 f25, 122 fr) |
| in vivo sl18 | medulla vs model-free | 0.1386 ± 0.00031 | 0.06162 ± 0.0018 | -0.07699 | 0.04879 (grasp v2 f25, 122 fr) |
| in vivo sl18 | aorta peak / model-free | 0.8651 ± 0.018 | 0.5027 ± 0.037 | -0.3624 | 0.5593 (grasp v2 f25, 122 fr) |
| in vivo sl18 | cortex-medulla late corr (model-free: negative) | 0.9982 ± 0.00025 | 0.7073 ± 0.036 | -0.2909 | 0.1459 (grasp v2 f25, 122 fr) |
| in vivo sl18 | held-out VAL k-space nmse | 0.3082 ± 0.0003 | 0.3356 ± 0.00043 | +0.02735 | nan (grasp v2 f25, 122 fr) |
| in vivo sl18 | held-out TEST k-space nmse | 0.3271 ± 0.0015 | 0.349 ± 0.00072 | +0.02198 | nan (grasp v2 f25, 122 fr) |
| in vivo sl19 | aorta vs model-free (affine nrmse) | 0.03691 ± 0.0014 | 0.172 ± 0.0068 | +0.1351 | 0.2056 (grasp v2 f25, 122 fr) |
| in vivo sl19 | cortex vs model-free | 0.1523 ± 0.00053 | 0.1166 ± 0.0023 | -0.03567 | 0.07572 (grasp v2 f25, 122 fr) |
| in vivo sl19 | medulla vs model-free | 0.141 ± 0.00021 | 0.06879 ± 0.00046 | -0.07222 | 0.07064 (grasp v2 f25, 122 fr) |
| in vivo sl19 | aorta peak / model-free | 0.9295 ± 0.011 | 0.5075 ± 0.022 | -0.422 | 0.4481 (grasp v2 f25, 122 fr) |
| in vivo sl19 | cortex-medulla late corr (model-free: negative) | 0.9988 ± 2e-05 | 0.7097 ± 0.028 | -0.2891 | 0.5916 (grasp v2 f25, 122 fr) |
| in vivo sl19 | held-out VAL k-space nmse | 0.3126 ± 0.0017 | 0.3406 ± 0.00034 | +0.02807 | nan (grasp v2 f25, 122 fr) |
| in vivo sl19 | held-out TEST k-space nmse | 0.3291 ± 0.00084 | 0.3556 ± 0.00067 | +0.02659 | nan (grasp v2 f25, 122 fr) |
| in vivo sl21 | aorta vs model-free (affine nrmse) | 0.05288 ± 0.0087 | 0.1776 ± 0.014 | +0.1247 | 0.1489 (grasp v2 f25, 122 fr) |
| in vivo sl21 | cortex vs model-free | 0.1546 ± 0.0003 | 0.139 ± 0.0021 | -0.01556 | 0.07513 (grasp v2 f25, 122 fr) |
| in vivo sl21 | medulla vs model-free | 0.1393 ± 0.00038 | 0.07494 ± 0.00081 | -0.06437 | 0.05774 (grasp v2 f25, 122 fr) |
| in vivo sl21 | aorta peak / model-free | 0.902 ± 0.03 | 0.559 ± 0.022 | -0.343 | 0.5394 (grasp v2 f25, 122 fr) |
| in vivo sl21 | cortex-medulla late corr (model-free: negative) | 0.996 ± 0.00044 | 0.9235 ± 0.013 | -0.07251 | 0.6047 (grasp v2 f25, 122 fr) |
| in vivo sl21 | held-out VAL k-space nmse | 0.2753 ± 0.0018 | 0.3027 ± 0.00098 | +0.02737 | nan (grasp v2 f25, 122 fr) |
| in vivo sl21 | held-out TEST k-space nmse | 0.2893 ± 0.0022 | 0.3114 ± 0.0006 | +0.0221 | nan (grasp v2 f25, 122 fr) |

all rows: mean ± sd over paired seeds 0,1,2. phantom: 5 of 7 angles/frame train, angle 5 val (checkpoint selection), angle 6 test. in vivo: keep_f25 (488/1710) train, complement v%10==8 val (early stop), v%10==9 test (never touched). cs rows are reference recons, not truth.

## in-vivo held-out k-space by |k| annulus (test spokes, mean over slices and seeds)
| |k| band | patlak | tofts |
|---|---|---|
| 0.00-0.06 | 0.286 | 0.309 |
| 0.06-0.12 | 0.574 | 0.592 |
| 0.12-0.19 | 0.657 | 0.680 |
| 0.19-0.25 | 0.814 | 0.845 |
| 0.25-0.31 | 0.882 | 0.915 |
| 0.31-0.38 | 0.778 | 0.848 |
| 0.38-0.44 | 0.646 | 0.762 |
| 0.44-0.50 | 0.906 | 0.917 |
| 0.50-0.56 | 1.017 | 0.979 |
| 0.56-0.62 | 0.858 | 0.875 |
| 0.62-0.69 | 1.030 | 0.962 |
| 0.69-0.75 | 1.123 | 1.004 |
| 0.75-0.81 | 1.119 | 1.004 |
| 0.81-0.88 | 1.238 | 1.068 |
| 0.88-0.94 | 1.250 | 1.085 |
| 0.94-1.00 | 1.284 | 1.124 |

tofts worse only in the k centre (energy-dominated global nmse), better at every band above 0.19 kmax. patlak's outer bands exceed 1.0 (worse than predicting zero).

## resources (both arms, identical network except the head width)
| setting | per-step | full 40k run | best step | peak gpu | params |
|---|---|---|---|---|---|
| phantom, 1g.10gb mig, batch 16384, width 768 | 0.31 s (both) | 12.3k s (both) | patlak 6k to 34k (no-motion), 4k to 16k (motion); tofts 32k to 40k, 4k to 16k (motion) | 5.0 gb (both) | 12.21m vs 12.25m |
| in vivo, 4g.40gb mig, batch 65536, hidden 512 | 0.20 s (both) | 8.03k s (both, ± 6 s) | best-heldout checkpoint | 12.3 gb (both) | 5.52m vs 5.54m |

## read
- phantom (truth available): tofts better on every image metric and on the kidney curves, by a wide margin on no-motion (ssim 0.84 to 0.94, cortex curve 0.109 to 0.015) and a small one under motion. aorta curve slightly worse in both sims: a regional dc offset present in all nik variants (sign varies) plus, under motion, roi mixing fitted by ext-tofts atoms. no bug found (aorta curve is in both spans to 1e-4; extra-atom content 0.03%).
- in vivo (no truth): kidney curves closer to the model-free reference (cortex 0.15 to 0.10-0.14, medulla 0.14 to 0.06-0.07) and the patlak-forced cortex == medulla late-phase identity is broken (corr 0.998 to 0.71-0.92; model-free says the two differ). aorta curve worse and bolus peak 0.50-0.56 vs 0.87-0.93 of model-free. caveat on that aorta ruler: the in-vivo aif atom IS the model-free aorta roi curve, so patlak's aorta match is by construction. held-out k-space: tofts fits train better, val/test 8% worse globally, driven by the k centre; better at |k| > 0.19 kmax.
- verdict: tofts basis is the right model on the phantom (the generator is ext-kety) and is a clear quantification gain there. in vivo the evidence is mixed: tissue curves and high-|k| consistency improve, k-centre consistency and the aorta bolus degrade. not a general recon win; a physics-basis result.

## flaws to state
- phantom aif = analytic generator aif (oracle); in vivo aif = model-free nufft on all 1710 spokes (val/test spokes leak into both arms' basis; ~6.8 s temporal blur of the basis bolus).
- tofts dictionary prior chosen after inspecting the phantom pklut (covers it by construction).
- one selection rule, rank per aif (16 phantom, 12 in vivo).
- 'wall_s' inside phantom checkpoints = time to best checkpoint, not run time; the table above uses per-step and full-run times.
- in-vivo cs held-out k-space blocked (magnitude-only cs files). motion-phantom cs ref = grasp v2 25 spf lam 0.25 run on the motion sim (haarpsi 0.71).

## status
complete: basis + rank study, verification 3a-3e, phantom no-motion (3+3 seeds), phantom motion (3+3), in vivo 18/19/21 (3+3 each), grasp v2 motion ref, evaluation. nothing running or blocked.

## files
MANIFEST.md, REPRO.md, verify.json, basis_*.npz (+ basis_*.log rank study), phantom_{nomotion,motion}.{json,md}, invivo.{json,md}, invivo/<arm>_sl<Z>_s<seed>/ (model_slice, nik_slice cplx), logs/. phantom checkpoints under results/xcat_physical_{nomotion,motion}_nik_vs_grasp/checkpoints/w768_ks2.5_s*_tofts16 (best + last).