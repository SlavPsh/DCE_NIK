"""TASK 1 binning-free ablation. ARM A (continuous) = pk_f0_sl21; ARM B (binned) =
pk_f0_bin{N}_sl21. per arm: vp/Ktrans maps + per-ROI values, agreement (corr + abs bias) with
conventional Patlak-fit of CS, corrected spatial eval, and AIF first-pass fidelity (aorta
recon TTP/FWHM -- the sharpest temporal feature, where binning should hurt most).
THE FIGURE: PK error vs bin width, ARM A as the zero-width point. out: figures + task1.json"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, json, os, sys, scipy.ndimage as ndi
from scipy.signal import savgol_filter
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/net/beegfs/users/P101440/grasp_pro_py"); sys.path.insert(0, ".")
from figpath import fig as fpath
import consolidated as C                                   # reuse slice_ctx, spatial_eval, norm
D = "/net/beegfs/users/P101440/DCE_NIK"; REF = "/net/beegfs/users/P101440/grasp_pro_py/results_ref"; TA = 375.0; Z = 21

# fixed Patlak basis on the 342 render grid (same as training)
az = np.load(f"{D}/aif_slice21.npz"); tg = np.linspace(0, TA, 342)
aif = np.interp(tg, np.asarray(az["tC"]), np.asarray(az["aif_frame"])); aif = aif / (aif.max() + 1e-9)
iaif = np.concatenate([[0], np.cumsum(0.5 * (aif[1:] + aif[:-1]) * np.diff(tg))]); iaif /= (iaif.max() + 1e-9)
Bfix = np.stack([aif, iaif, np.ones_like(aif)], 1); Bpinv = np.linalg.pinv(Bfix)

ctx = C.slice_ctx(Z); rois = ctx["rois"]; BODY = ctx["BODY"]
cs100 = np.abs(np.load(f"{REF}/slice_{Z:02d}.npz")["cs_img"]).astype(np.float32)
cs_amp = np.tensordot(cs100, Bpinv.T, axes=([2], [0])); vp_cs, kt_cs = np.abs(cs_amp[..., 0]), np.abs(cs_amp[..., 1])

def first_pass(curve, t):                                  # TTP + FWHM of the first-pass bolus
    sm = savgol_filter(curve, 11, 3); fp = (t > 20) & (t < 130); ttp = t[fp][np.argmax(sm[fp])]
    pk = sm[fp].max(); half = (sm > 0.5 * pk) & (t < ttp + 60) & (t > 20)
    fw = (t[half].max() - t[half].min()) if half.any() else np.nan
    return float(ttp), float(fw), float(np.std(curve - sm) / (abs(sm).max() + 1e-9))

ARMS = [("A_cont", 0, f"{D}/results_batch/pk_f0_sl21/nik_slice_21_cplx.npy")]
ARMS += [(f"bin{N}", N, f"{D}/results_batch/pk_f0_bin{N}_sl21/nik_slice_21_cplx.npy") for N in (5, 15, 30, 60, 120)]
res = []
for lab, N, path in ARMS:
    if not os.path.exists(path): print("MISSING", lab); continue
    rec = np.load(path); mag = np.abs(rec).astype(np.float32)
    amp = np.tensordot(mag, Bpinv.T, axes=([2], [0])); vp, kt = np.abs(amp[..., 0]), np.abs(amp[..., 1])
    roi_vals = {nm: dict(vp=float(vp[m].mean()), ktrans=float(kt[m].mean())) for nm, m in rois.items() if m.sum() > 0}
    ag_vp = float(np.corrcoef(vp[BODY], vp_cs[BODY])[0, 1]); ag_kt = float(np.corrcoef(kt[BODY], kt_cs[BODY])[0, 1])
    bias_vp = float(vp[BODY].mean() - vp_cs[BODY].mean()); bias_kt = float(kt[BODY].mean() - kt_cs[BODY].mean())
    ao = rois["aorta"]; aoc = C.norm(tg, np.array([mag[..., i][ao].mean() for i in range(342)])) if ao.sum() > 0 else None
    ttp, fw, osc = first_pass(aoc, tg) if aoc is not None else (np.nan, np.nan, np.nan)
    sp = C.spatial_eval(mag, Z)
    res.append(dict(arm=lab, bin=N, roi=roi_vals, agree_vp=ag_vp, agree_kt=ag_kt, bias_vp=bias_vp, bias_kt=bias_kt,
                    aorta_ttp=ttp, aorta_fwhm=fw, aorta_osc=osc, rulerA=sp["rulerA_haarpsi"], rulerB=sp["rulerB_haarpsi"], bgE=sp["bgE"], vp=vp, kt=kt))
json.dump([{k: v for k, v in r.items() if k not in ("vp", "kt")} for r in res], open(f"{D}/task1.json", "w"), indent=1, default=float)

A = next(r for r in res if r["arm"] == "A_cont"); vpA, ktA = A["vp"], A["kt"]
print(f"\n{'arm':>8}{'bin':>5}{'A_ttp':>7}{'A_fwhm':>8}{'A_osc':>7}{'agr_kt':>8}{'bias_kt':>9}{'ktMapCorr_vsA':>14}{'rulerA':>8}")
for r in res:
    mapcorr = float(np.corrcoef(r["kt"][BODY], ktA[BODY])[0, 1])
    print(f"{r['arm']:>8}{r['bin']:>5}{r['aorta_ttp']:>7.0f}{r['aorta_fwhm']:>8.1f}{r['aorta_osc']:>7.3f}{r['agree_kt']:>8.3f}{r['bias_kt']:>9.4f}{mapcorr:>14.3f}{r['rulerA']:>8.3f}")

# ---- THE FIGURE: error vs bin width (Arm A at x=0) ----
bins = [r["bin"] for r in res]; xbin = [b if b > 0 else 0 for b in bins]
fw = [r["aorta_fwhm"] for r in res]; ktmap = [float(np.corrcoef(r["kt"][BODY], ktA[BODY])[0, 1]) for r in res]
ktbias = [abs(r["bias_kt"] - A["bias_kt"]) for r in res]                 # Ktrans bias drift from Arm A
fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
o = np.argsort(xbin); xb = np.array(xbin)[o]
ax[0].plot(xb, np.array(fw)[o], "o-", color="#d11"); ax[0].set_xlabel("bin width (spokes/frame); 0 = ARM A continuous")
ax[0].set_ylabel("aorta first-pass FWHM (s)"); ax[0].grid(alpha=.3); ax[0].set_title("AIF first-pass width vs binning\n(broader = binning smears the bolus)")
ax[1].plot(xb, 1 - np.array(ktmap)[o], "o-", color="#70c"); ax[1].set_xlabel("bin width (spokes/frame)")
ax[1].set_ylabel("1 - corr(Ktrans map, ARM A)"); ax[1].grid(alpha=.3); ax[1].set_title("Ktrans map divergence from continuous")
ax[2].plot(xb, np.array(ktbias)[o], "o-", color="#188"); ax[2].set_xlabel("bin width (spokes/frame)")
ax[2].set_ylabel("|Ktrans body-bias drift vs ARM A|"); ax[2].grid(alpha=.3); ax[2].set_title("Ktrans quantitative drift")
fig.suptitle("TASK 1 binning-free ablation, slice 21 (PK error vs bin width; ARM A = 0)", fontweight="bold")
fig.tight_layout(); p = fpath("task1_binning.png"); fig.savefig(p, dpi=135); print("wrote", p.split("/")[-1])
