"""post-hoc single-constant calibration for cross-method curve comparison (ADDITIVE, does not touch
existing baseline-normalized plots). each method scaled by ONE constant = median over a large in-body
pre-contrast anchor, so curves are in shared 'body pre-contrast = 1' units (actual signal, not relative).
prints logical checks at every step. out: figures/calib_curves.png"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, os
import scipy.ndimage as ndi
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
RD="results/realdata_nik_vs_cs_figures"; TA=375.0; PRE=40.0
import os as _os
# reference-method plumbing. defaults = grasp-pro (unchanged). grasp v2:
#   CSD=/scratch/rnga/vvpshenov/grasp_v2/results_grasp_v2 CSPRE=gv2 TAG=_gv2
_CSD = _os.environ.get("CSD", "/scratch/rnga/vvpshenov/grasp_pro_py/results_spoke_cs")
_CSPRE = _os.environ.get("CSPRE", "cs"); _TAG = _os.environ.get("TAG", "")
GR=_CSD

ok=lambda c,m: print(("  PASS " if c else "  FAIL ")+m)
z=np.load("step2_slice21.npz"); body=z["body"].astype(bool)                      # [192,192] in-body mask (mf-derived)
mf=np.abs(z["mf"]).transpose(1,2,0).astype(np.float32); tmf=np.asarray(z["tmf"])
ao=np.load("aif_slice21.npz")["ao"].astype(bool); kd=np.load("realkid_slice21.npz"); cx=kd["cortex"].astype(bool); md=kd["medulla"].astype(bool)
core=ndi.binary_erosion(body, iterations=6)                                      # interior region for cross-check (edges = worst streaks/motion)
L=lambda p: np.abs(np.load(p)).astype(np.float32) if os.path.exists(p) else None
# (label, recon, time-axis)
M=[("nik input", L(f"{RD}/outcoil_subspace_input_slice21.npy"), np.linspace(0,TA,342)),
   ("nik output w0-90", L(f"{RD}/outcoil_subspace_output_pw90_slice21.npy"), np.linspace(0,TA,342)),
   ("grasp f80-match", L(f"{GR}/{_CSPRE}_slice21_f80match.npy"), np.linspace(0,TA,122)),
   ("model-free", mf, tmf)]
M=[(l,v,t) for l,v,t in M if v is not None]

print("=== STEP 1: shapes + mask alignment ===")
ok(body.shape==(192,192) and body.dtype==bool, f"body mask 192x192 bool, fraction {body.mean():.2f}")
ok(0.2<body.mean()<0.7, "body fraction in sane range 0.2-0.7")
ok(core.sum()>0.3*body.sum(), f"eroded core non-trivial ({core.sum()}/{body.sum()} px)")
for l,v,t in M:
    ok(v.shape[:2]==(192,192), f"{l}: spatial 192x192 (nt={v.shape[2]}, time-frames={len(t)})")
    ok(v.shape[2]==len(t), f"{l}: recon frames == time axis length")
    pm=v[body].mean(); om=v[~body].mean()
    ok(pm>1.5*om, f"{l}: in-body mean {pm:.3g} >> out-of-body {om:.3g} (mask aligned to anatomy)")

print("=== STEP 2: pre-contrast anchor (median over body, t<40s) ===")
anc={}; anc2={}
for l,v,t in M:
    pre=t<PRE; npre=int(pre.sum())
    ok(npre>=3, f"{l}: {npre} pre-contrast frames (t<{PRE:.0f}s)")
    a1=float(np.median(v[body][:, pre])); a2=float(np.median(v[core][:, pre]))
    ok(np.isfinite(a1) and a1>0, f"{l}: body anchor = {a1:.4g} (>0, finite)")
    anc[l]=a1; anc2[l]=a2

print("=== STEP 3: cross-check (region-independence of the scale) ===")
# within-method region ratio core/body should be ~same across methods if scale is a pure scalar
ratios={l: anc2[l]/anc[l] for l in anc}
rv=np.array(list(ratios.values()))
for l in anc: print(f"    {l:18s} core/body ratio = {ratios[l]:.3f}")
ok(rv.std()/rv.mean()<0.10, f"core/body ratio consistent across methods (CoV {100*rv.std()/rv.mean():.1f}% < 10%)")

print("=== STEP 4: calibrate + prove it is a pure global scale ===")
cal={l: v/anc[l] for l,v,_ in M}                                                 # calibrated recon, body pre-contrast -> 1
med=lambda v,mk: np.median(v[mk],0); n0=lambda c: c/(np.median(c[:8])+1e-30)
for l,v,t in M:
    # calibrated body pre-contrast median must be ~1 by construction
    cb=float(np.median(cal[l][body][:, t<PRE]))
    ok(abs(cb-1.0)<1e-4, f"{l}: calibrated body pre-contrast median = {cb:.5f} (==1)")
    # KEY: baseline-normalizing the calibrated curve must equal baseline-normalizing the raw curve (pure scalar, shape unchanged)
    raw_bn=n0(med(v,ao)); cal_bn=n0(med(cal[l],ao))
    ok(np.allclose(raw_bn, cal_bn, atol=1e-5), f"{l}: baseline-norm(calibrated)==baseline-norm(raw) -> calibration is pure scale, no shape change")

print("=== STEP 5: plot (row1 baseline-norm as before, row2 calibrated actual-signal) ===")
fig,ax=plt.subplots(2,3,figsize=(15,8))
for j,(nm,mk) in enumerate([("aorta",ao),("cortex",cx),("medulla",md)]):
    for l,v,t in M:
        ax[0,j].plot(t, n0(med(v,mk)), lw=1.2, label=l)                          # relative enhancement (existing convention)
        ax[1,j].plot(t, med(cal[l],mk), lw=1.2, label=l)                         # calibrated actual signal (shared body units)
    ax[0,j].set_title(f"{nm}  (baseline-normalized)"); ax[1,j].set_title(f"{nm}  (calibrated, body pre-contrast=1)")
    for r in (0,1): ax[r,j].set_xlabel("time (s)")
ax[0,0].legend(fontsize=7); ax[0,0].set_ylabel("rel. enhancement"); ax[1,0].set_ylabel("signal / body baseline")
fig.suptitle("cross-method curves: relative (top) vs single-constant calibrated actual signal (bottom), real slice 21", fontsize=12)
plt.tight_layout(); fig.savefig(f"{RD}/figures/calib_curves{_TAG}.png", dpi=120)
print("=== STEP 6: aorta peak, baseline-norm vs calibrated (exact) ===")
bnpk={l: float(n0(med(v,ao)).max()) for l,v,t in M}; capk={l: float(med(cal[l],ao).max()) for l,v,t in M}
mfl="model-free"
for l in bnpk: print(f"    {l:18s} baseline-norm peak {bnpk[l]:5.2f} (/mf {bnpk[l]/bnpk[mfl]:.2f})  |  calibrated peak {capk[l]:5.2f} (/mf {capk[l]/capk[mfl]:.2f})")
print("SAVED figures/calib_curves.png"); print("DONE_CALIB")
