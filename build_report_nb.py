"""Build the combined phantom+real-data comparison notebook: headlines + metric tables + embedded figures.
minimal text, lowercase, no em dashes, noun/adjective fragments. phantom = vs ground truth (accuracy);
real = vs cs reference, no gt (consistency)."""
import os, nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbclient import NotebookClient

B = "/net/beegfs/users/P101440/DCE_NIK"
PH = f"{B}/results/xcat_physical_nomotion_nik_vs_grasp"; FP = f"{PH}/figures"
RD = f"{B}/results/realdata_nik_vs_cs_figures/figures"
C = []
def md(s): C.append(new_markdown_cell(s))
def code(s): C.append(new_code_cell(s))

code("import os; os.chdir('/net/beegfs/users/P101440/DCE_NIK')\n"
     "import warnings; warnings.filterwarnings('ignore')\n"
     "import json, numpy as np, pandas as pd\n"
     "from IPython.display import Image, display\n"
     "pd.set_option('display.max_columns', None); pd.set_option('display.width', 220)\n"
     "def low(df):\n"
     "    df = df.copy(); df.columns = [str(c).lower() for c in df.columns]\n"
     "    df[df.columns[0]] = df[df.columns[0]].astype(str).str.lower(); return df")

md("# nik vs cs / grasp-pro, dce-mri recon\n\n"
   "| dataset | reference | measure |\n|---|---|---|\n"
   "| phantom, xcat no-motion | ground truth | accuracy |\n"
   "| real in-vivo | cs / model-free recon, no gt | consistency |")

# ===================== PHANTOM =====================
md("# 1. phantom, xcat no-motion, vs ground truth\n\n"
   "methods: cs-file (sim recon, strong cs ref), grasp-pro (fair k=12, held-out cv selected), nik f0 / sub5 / sub16 / free. truth-scaled, median rois.\n\n"
   "spoke budget: nik and grasp-pro identical, 5 spokes/frame (angles 0-4, spoke-matched). cs-file = sim reference recon, 7 spokes/frame (full data, 40% more), not matched.")

md("### roi placement, truth. aorta red, cortex green, medulla orange")
code(f"display(Image('{FP}/fig0_roi_check.png'))")

md("### image metrics, body-masked, vs truth (2x oversampled render, re-baselined)\n\n"
   "best nik haarpsi 0.90 beats grasp 0.87 (first phantom metric nik wins); psnr gap 4.3 -> 1.2 db; ssim 0.93 vs 0.95. single-seed mean and complex seed-average, vs grasp-k12.")
code(f"reb = pd.read_csv('{PH}/l3_rebaseline.csv').round(4)\n"
     "display(low(reb))")

md("### spatial: 1px render bug fixed, residual = smooth low-|k| bias\n\n"
   "the nik render aligned to truth with im[::-1,::-1] = a 180deg rotation about (n-1)/2 not n/2 (even grid) = a +1px shift, nik only. invisible to spectra/held-out-nmse/curve corr; degrades ssim smoothly. fixed (integer roll); ssim 0.76 -> 0.91-0.93. residual gap to grasp: mostly a smooth low-|k| bias (91% low-|k| error; +4.2 db post-hoc), traceable to the k-space normalization envelope (not coil/sense). 6-point envelope sweep (0.5-1.0): env 0.75 already minimizes the bias; the knob is a dead end at source (pushing it is a pareto slide). post-hoc-removable only.")
code(f"display(Image('{FP}/fig_bias_locate.png'))")

md("### render fix 2: single-fov ifft aliasing (this week)\n\n"
   "the render imaged the coordinate network on a single-fov k-grid, which aliases broadband. standard 2x oversampling (query on 2x grid, ifft, crop) fixes it: +1.7 to +5.3 db psnr vs truth on every seed, and it also collapsed the seed spread. this was the third render bug (support radius, 1px rot180, now aliasing), so most of the apparent nik-vs-grasp spatial gap was rendering, not the model.\n\n"
   "caveat: haarpsi is one metric (nik wins haarpsi, not 'beats grasp'); psnr and ssim still favour grasp. this is a correctness fix, not a modelling gain.")
code("z = np.load('{}/arrays/compare_recons.npz'.format('/net/beegfs/users/P101440/DCE_NIK/results/xcat_physical_nomotion_nik_vs_grasp'))\n"
     "import matplotlib.pyplot as plt\n"
     "pk = 1  # frames = [early, peak, late]\n"
     "body = z['body']; ys, xs = np.where(body); y0, y1, x0, x1 = ys.min(), ys.max()+1, xs.min(), xs.max()+1\n"
     "crop = lambda im: im[y0:y1, x0:x1]\n"
     "vmax = float(np.percentile(z['truth'][:, :, pk][body], 99.5))\n"
     "# best models at peak enhancement\n"
     "labs = ['truth', 'grasp-k12', 'nik sub16 (2x)', 'nik free (2x)']; ims = [z['truth'], z['grasp'], z['sub16_2x'], z['free_2x']]\n"
     "fig, ax = plt.subplots(1, 4, figsize=(14, 4.0))\n"
     "for a, l, im in zip(ax, labs, ims): a.imshow(crop(im[:, :, pk]), cmap='gray', vmin=0, vmax=vmax); a.set_title(l, fontsize=11); a.axis('off')\n"
     "fig.suptitle('peak enhancement, t={:.0f}s, body-cropped'.format(z['ftimes'][pk]), fontsize=12); plt.tight_layout(); plt.show()\n"
     "# before / after the oversampling fix (sub16)\n"
     "fig, ax = plt.subplots(1, 3, figsize=(10.5, 4.0))\n"
     "for a, l, im in zip(ax, ['sub16 1x (single-fov, aliased)', 'sub16 2x (oversampled)', 'truth'], [z['sub16_1x'][:, :, pk], z['sub16_2x'][:, :, pk], z['truth'][:, :, pk]]):\n"
     "    a.imshow(crop(im), cmap='gray', vmin=0, vmax=vmax); a.set_title(l, fontsize=11); a.axis('off')\n"
     "plt.tight_layout(); plt.show()\n"
     "# metrics, body-masked ruler, best nik (complex seed-avg) vs grasp\n"
     "reb = pd.read_csv('{}/l3_rebaseline.csv'.format('/net/beegfs/users/P101440/DCE_NIK/results/xcat_physical_nomotion_nik_vs_grasp'))\n"
     "nm = reb['name'].astype(str).str.strip(); gv = lambda name, col: float(reb[nm == name][col].iloc[0])\n"
     "fig, ax = plt.subplots(1, 2, figsize=(10, 3.3))\n"
     "for a, mt, col in zip(ax, ['HaarPSI (body-masked)', 'SSIM (body-masked)'], ['haarpsi', 'ssim']):\n"
     "    vals = [gv('GRASP-K12 (ref)', col), gv('sub16 cplx-seedavg', col), gv('free cplx-seedavg', col)]\n"
     "    bars = a.bar(['grasp', 'nik sub16', 'nik free'], vals, color=['#8a94a2', '#0b6e78', '#2f8f5b'])\n"
     "    a.set_title(mt, fontsize=11); a.set_ylim(min(vals)-0.02, max(vals)+0.015)\n"
     "    for b, v in zip(bars, vals): a.text(b.get_x()+b.get_width()/2, v, '%.3f' % v, ha='center', va='bottom', fontsize=9)\n"
     "plt.tight_layout(); plt.show()\n"
     "# contrast curves, aorta / cortex / medulla, best models vs truth\n"
     "t = z['times']; fig, ax = plt.subplots(1, 3, figsize=(14, 3.4))\n"
     "for a, roi in zip(ax, ['aorta', 'cortex', 'medulla']):\n"
     "    a.plot(t, z['cur_truth_'+roi], 'k-', lw=2, label='truth'); a.plot(t, z['cur_grasp_'+roi], lw=1.3, label='grasp')\n"
     "    a.plot(t, z['cur_sub16_'+roi], lw=1.3, label='nik sub16'); a.plot(t, z['cur_free_'+roi], lw=1.3, label='nik free')\n"
     "    a.set_title(roi, fontsize=11); a.set_xlabel('time (s)')\n"
     "ax[0].set_ylabel('signal'); ax[0].legend(fontsize=8, frameon=False); plt.tight_layout(); plt.show()")

md("### seed reproducibility (finding). spread comparable to the grasp gap\n\n"
   "3 seeds: ssim spread 0.023 (sub12) / 0.034 (free); psnr spread 3.4 db; aorta peak spread 0.24 (one sub12 seed overshoots to 1.05). the spread ~ the gap to grasp (0.05 ssim), so nik spatial is not yet reproducible at the comparison precision. grasp is deterministic (a point); nik a seed distribution.")

md("### recons. truth top, methods below, dce phases")
code(f"display(Image('{FP}/fig1_montage.png'))")

md("### contrast curves: two failure modes. nik wins the whole curve + timing (aorta 0.08 vs grasp 0.16) but overshoots the aorta peak ~10% (seed-variable); grasp is peak-exact but ~2.6s late. current curves are in the render-fix-2 panel above (2x render); per-config curve nrmse is in the re-baselined table.")

md("### error maps, recon vs truth")
code(f"display(Image('{FP}/fig2_errormaps.png'))")

md("### pk maps, signal-domain patlak, relative. nik-sub16 best (ktrans near truth); sub12 unreliable (seed); grasp under-reads aorta vp")
code(f"display(Image('{FP}/fig6_pk_maps.png'))\ndisplay(low(pd.read_csv('{PH}/pk_metrics.csv')))")

md("### choosing k fairly: held-out cross-validation, not the oracle\n\n"
   "variance selection (k=5) misses the bolus: it is a 0.004% variance navigator component (pc 10). held-out k-space cv (predict measured unused spokes, truth-blind, same as nik scoring) picks k*=12 and captures the bolus (peak 0.77). so fair grasp = k12.")
code(f"kcv = pd.read_csv('{PH}/grasp_kcv.csv')[['K','val_heldout_NMSE','test_heldout_NMSE','aorta_peak_truth']].round(5)\n"
     "display(low(kcv))\ndisplay(Image('{}/fig_kcv.png'))".format(FP))
md("### navigator eigenspectrum. bolus = low-variance pc, variance selection misses it")
code(f"display(Image('{FP}/fig_nav_spectrum.png'))")
md("### k-sweep frontier, per-spoke navigator. minimum near k=8-12, collapse past k=12")
code(f"ks = pd.read_csv('{PH}/grasp_ksweep_pareto.csv')\n"
     "ks = ks[ks['family']=='GRASP'][['K','SSIM','HaarPSI','curve_aorta','aorta_peak','aorta_fwhm']].round(3)\n"
     "display(low(ks))\ndisplay(Image('{}/fig_ksweep_pareto.png'))".format(FP))
md("### note: the old k x spokes/frame overlay figure was pre render-fix and on an unmasked haarpsi ruler (grasp ~0.91), not comparable to the body-masked table above (grasp 0.87). the current, ruler-consistent nik-vs-grasp comparison is the render-fix-2 panel (metrics + images + curves). the grasp-only k-sweep above still stands for k selection (k*=12).")

# ===================== REAL =====================
md("# 2. real in-vivo, vs cs reference, no ground truth\n\n"
   "consistency, not accuracy. reference: cs / model-free recon. cs-likeness, inter-slice consistency, temporal behavior.")

md("### recons, slice 21. model-free ref, nik, cs")
code(f"display(Image('{RD}/fig4_realdata_montage.png'))")

md("### aorta curve, median, aif-gated roi. roi check below")
code(f"display(Image('{RD}/fig5_realdata_aorta_curve.png'))\ndisplay(Image('{RD}/fig6_roi_verification.png'))")

md("### kidney curves, cortex, medulla, slice 21. model-free ref, nik, cs")
code(f"display(Image('{RD}/fig7_realdata_kidney_curves.png'))")

md("### motion + rois, real cs axial slice 21, over time. fixed contours, anatomy drift")
code(f"display(Image('{RD}/fig8_realdata_motion_rois.gif'))")

md("### temporal denoising. nik advantage, real. cs osc: broadband noise, no respiratory peak. nik: smooth. osc = wobble, respfrac = respiratory power")
code("tS = json.load(open('task_S.json')); sl=[s for s in tS if s in ('18','19','21')]\n"
    "meth=['ref','CS_f100','CS_f25','NIK_full']\n"
    "def avg(m,roi,q):\n"
    "    v=[tS[s][m][roi][q] for s in sl if m in tS[s] and roi in tS[s][m]]\n"
    "    return np.nanmean(v) if v else np.nan\n"
    "rows=[[roi,lab]+[round(avg(m,roi,q),4) for m in meth] for roi in ['cortex','medulla','aorta'] for q,lab in [('osc','oscillation'),('respfrac','resp-band frac')]]\n"
    "display(low(pd.DataFrame(rows, columns=['roi','metric']+meth)))")
code(f"display(Image('{RD}/fig3_denoising.png'))")

md("### pk inter-slice consistency. quantification support. nik cov comparable or better vs cs-fit. cov %, lower better")
code("t2=json.load(open('task2.json'))['consistency']; meth=['NIK_F0','NIK_F2','CS_fit']\n"
    "rows=[[par,roi]+[round(t2[f'{par}_{roi}'].get(m,np.nan),2) for m in meth] for par in ['ktrans','vp'] for roi in ['aorta','cortex','medulla','liver'] if f'{par}_{roi}' in t2]\n"
    "display(low(pd.DataFrame(rows, columns=['param','roi']+meth)))")
code(f"display(Image('{RD}/fig2_pk_cov.png'))")

md("### spoke-fraction frontier. cs-favored, nuanced. nik haarpsi vs cs flat ~0.71, below cs every fraction. cs-likeness, no gt")
code("hs=json.load(open('haarpsi_spoke.json')); pct=sorted({r['pct'] for r in hs.values()}, reverse=True)\n"
    "def g(p,mth): return next((r['mean'] for r in hs.values() if r['pct']==p and r['meth']==mth), np.nan)\n"
    "display(pd.DataFrame([[p,round(g(p,'NIK'),3),round(g(p,'CS'),3)] for p in pct], columns=['spoke %','nik (vs cs-f100)','cs (self)']))")
code(f"display(Image('{RD}/fig1_spoke_frontier.png'))")

# ===================== COIL PLACEMENT (latest) =====================
md("# 3. coil placement: input embedding vs output heads (latest)\n\n"
   "move the coil from a network input (learned embedding, f(k,t,coil)->1 value) to shared-backbone output heads, one per coil (g(k,t)->c values). same fourier features, gabor backbone, temporal basis, normalization, 2x render, scoring; only the coil handling differs. tested on subspace, f0, f2. output-coil is also ~8x cheaper to train (one backbone pass vs c passes).")
md("### phantom, vs truth. output-coil beats input-coil (sub16) on every metric, val-selected\n\n"
   "| metric | input-coil (sub16) | output-coil | grasp |\n|---|---|---|---|\n"
   "| psnr | 36.12 | **37.21** | 38.53 |\n| ssim | 0.922 | **0.928** | 0.952 |\n| haarpsi | 0.886 | **0.901** | 0.871 |\n| cortex curve nrmse | 0.055 | **0.020** | - |\n\n"
   "output-coil wins all, strongest on the tissue curves (~2.5x lower cortex error); haarpsi clears grasp. the shared backbone consolidates the coils and the temporal subspace. images below: output-coil visibly cleaner (less grain), |output-truth| difference map dimmer at the kidneys.")
code(f"display(Image('{FP}/outcoil_compare.png'))")
md("### real slice 21, no ground truth. held-out spoke nmse (matched setup, val spokes)\n\n"
   "| held-out nmse | input-coil | output-coil |\n|---|---|---|\n"
   "| subspace | **0.294** | 0.352 |\n| f2 | **0.508** | 0.536 |\n| f0 | 0.707 | **0.684** |\n\n"
   "parity, no consistent winner. the curves vs the model-free nufft reference (thick grey, unbiased all-data gridding) are the telling test: input-coil tracks the enhancement amplitude (cortex ~3.0, aorta peak ~11, matching model-free); output-coil under-reads (cortex ~2.5, aorta ~7), smoother but low-biased like grasp. so the phantom win does NOT replicate in vivo; against the model-free reference input-coil tracks the dynamics better. f0/f2 (patlak, 3-5 atoms) undershoot the tissue plateaus. figure: rows subspace/f0/f2, cols input|output|grasp|diff, + aorta/cortex/medulla curves.")
code(f"display(Image('{RD}/outcoil_report.png'))")

# ===================== VERDICT =====================
md("# summary\n\n"
   "| domain | finding |\n|---|---|\n"
   "| phantom render bug 1 | +1px even-grid rot180 shift (nik only) inflated the spatial gap ~4x; fixed, guarded by assertion harness |\n"
   "| phantom render bug 2 | single-fov ifft aliasing; 2x oversampling = +1.7 to +5.3 db psnr vs truth, seed spread collapsed. third render bug total, so most of the gap was rendering not modelling |\n"
   "| phantom global image | best nik haarpsi 0.90 beats grasp 0.87 (first metric nik wins); psnr gap 4.3 -> 1.2 db; ssim 0.93 vs 0.95 |\n"
   "| phantom spatial residual | mostly a smooth low-|k| envelope-normalization bias (+4.2 db post-hoc); 6-point envelope sweep = dead end (env 0.75 already optimal, pushing it is a pareto slide) |\n"
   "| phantom reproducibility | nik seed spread (0.023 ssim, 3.4 db) ~ the grasp gap; not yet reproducible at comparison precision |\n"
   "| phantom whole-curve | nik wins (aorta 0.09 vs 0.22; cortex/medulla via free) + exact timing |\n"
   "| phantom aorta peak | nik overshoots ~+10% (real, seed-variable); grasp peak-exact but 2.6s late |\n"
   "| phantom k selection | held-out cv picks k*=12 from data (sharp min); captures bolus |\n"
   "| phantom pk | nik-sub16 edges grasp on tissue ktrans |\n"
   "| real temporal denoising | nik advantage, cs osc noise |\n"
   "| real pk inter-slice consistency | quantification support |\n"
   "| real spoke frontier | cs-favored, no gt |\n"
   "| coil placement (phantom) | output-coil beats input-coil (sub16) on all metrics; best nik config, haarpsi 0.90 clears grasp; ~8x cheaper |\n"
   "| coil placement (real) | held-out nmse parity; vs model-free nufft ref, output-coil under-reads amplitude (low bias like grasp), input-coil tracks it; phantom win does not replicate in vivo |\n\n"
   "headline: most of the apparent nik-vs-grasp spatial gap was rendering, not the model. after fixing three render bugs (support radius, 1px rot180, single-fov aliasing) nik wins haarpsi (0.90 vs 0.87), the psnr gap drops to 1.2 db, ssim stays behind (0.93 vs 0.95), and nik wins whole-curve dynamics + timing. a correctness fix, not a modelling gain; sense-forward and the motion test are next.")

nb = new_notebook(); nb['cells'] = C
nb['metadata']['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
os.chdir(B)
print("executing notebook...")
NotebookClient(nb, timeout=600, kernel_name='python3', resources={'metadata': {'path': B}}).execute()
out = f"{PH}/NIK_vs_CS_combined_report.ipynb"
nbf.write(nb, out)
print("WROTE", out)
