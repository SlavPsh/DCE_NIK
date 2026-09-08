"""TASK 2: PK/Patlak replication + inter-slice consistency. per slice (18,19,21), F in {0,2}:
vp/Ktrans maps (project recon onto that slice's fixed Patlak basis), per-ROI values, agreement
with conventional Patlak-fit of that slice's CS recon (corr + bias), spatial f25, and the F=0
cortical-washout figure RECHECKED on raw curves. PRIMARY: inter-slice CoV of Ktrans/vp per
tissue, NIK-PK vs conventional CS-fit. out: task2.json + figures."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, json, os, sys, scipy.ndimage as ndi
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/scratch/rnga/vvpshenov/grasp_pro_py"); sys.path.insert(0, ".")
from figpath import fig as fpath
import consolidated as C
D = "/scratch/rnga/vvpshenov/DCE_NIK"; REF = "/scratch/rnga/vvpshenov/grasp_pro_py/results_ref"; TA = 375.0
import os as _os
_TAG = _os.environ.get("TAG", "")   # "" = grasp-pro (unchanged)

SLICES = [18, 19, 21]; ROIS = ["aorta", "cortex", "medulla", "liver"]

def patlak_basis(Z):                                       # that slice's fixed [AIF, integral AIF, baseline] on 342 grid
    az = np.load(f"{D}/aif_slice{Z}.npz"); tg = np.linspace(0, TA, 342)
    aif = np.interp(tg, np.asarray(az["tC"]), np.asarray(az["aif_frame"])); aif = aif / (aif.max() + 1e-9)
    iaif = np.concatenate([[0], np.cumsum(0.5 * (aif[1:] + aif[:-1]) * np.diff(tg))]); iaif /= (iaif.max() + 1e-9)
    return np.stack([aif, iaif, np.ones_like(aif)], 1), tg

def maps_from(recon_mag, Bpinv):                           # project raw dynamic recon onto fixed Patlak cols -> |vp|,|Ktrans|
    amp = np.tensordot(recon_mag, Bpinv.T, axes=([2], [0])); return np.abs(amp[..., 0]), np.abs(amp[..., 1])

res = {}
for Z in SLICES:
    ctx = C.slice_ctx(Z); rois = ctx["rois"]; BODY = ctx["BODY"]
    cs100 = ctx.get("cs_meth") if ctx.get("cs_meth") is not None else ctx["cs100"]   # method image, roi anatomy stays pro
    B, tg = patlak_basis(Z); Bpinv = np.linalg.pinv(B)
    vp_cs, kt_cs = maps_from(cs100, Bpinv)                  # conventional Patlak fit of CS recon
    slc = {}
    for F in [0, 2]:
        p = f"{D}/results_batch/pk_f{F}_sl{Z}/nik_slice_{Z}_cplx.npy"
        if not os.path.exists(p): slc[F] = dict(missing=True); continue
        mag = np.abs(np.load(p)).astype(np.float32); vp, kt = maps_from(mag, Bpinv)
        roivals = {r: dict(vp=float(vp[rois[r]].mean()), ktrans=float(kt[rois[r]].mean()))
                   for r in ROIS if rois.get(r) is not None and rois[r].sum() > 0}
        ag_vp = float(np.corrcoef(vp[BODY], vp_cs[BODY])[0, 1]); ag_kt = float(np.corrcoef(kt[BODY], kt_cs[BODY])[0, 1])
        bias_kt = float((kt[BODY].mean() - kt_cs[BODY].mean()) / (kt_cs[BODY].mean() + 1e-12) * 100)
        try: sp = C.spatial_eval(mag, Z)
        except Exception as e: sp = dict(err=str(e))
        # F=0 cortical washout on RAW curves (global-affine, not peak-norm)
        washout = None
        if F == 0 and rois.get("cortex") is not None:
            mf = np.load(f"{D}/step2_slice{Z}.npz")["mf"]; tmf = ctx["tmf"]; tN = np.linspace(0, TA, mag.shape[-1])
            names = [r for r in ROIS if rois.get(r) is not None and rois[r].sum() > 0]
            realc = {r: np.array([im[rois[r]].mean() for im in mf]) for r in names}
            nikc = {r: np.interp(tmf, tN, np.array([mag[..., i][rois[r]].mean() for i in range(mag.shape[-1])])) for r in names}
            Ns = np.concatenate([nikc[r] for r in names]); Rs = np.concatenate([realc[r] for r in names])
            a, b = np.linalg.lstsq(np.stack([Ns, np.ones_like(Ns)], 1), Rs, rcond=None)[0]
            lt = (tmf > 100) & (tmf < 210)
            washout = dict(nik_cortex_plateau=float(np.median((a*nikc["cortex"]+b)[lt])),
                           real_cortex_plateau=float(np.median(realc["cortex"][lt])))
        slc[F] = dict(roivals=roivals, agree_vp=ag_vp, agree_kt=ag_kt, bias_kt_pct=bias_kt, spatial=sp, washout=washout,
                      vp=vp, kt=kt)
    slc["cs_fit"] = dict(vp_cs=vp_cs, kt_cs=kt_cs, roivals={r: dict(vp=float(vp_cs[rois[r]].mean()), ktrans=float(kt_cs[rois[r]].mean()))
                                                             for r in ROIS if rois.get(r) is not None and rois[r].sum() > 0})
    res[Z] = slc

# ---- tables ----
print("=== per-slice PK: agreement with conventional CS Patlak-fit, spatial, F0 cortical washout ===")
for Z in SLICES:
    for F in [0, 2]:
        s = res[Z].get(F, {})
        if s.get("missing"): print(f"  sl{Z} F{F}: MISSING"); continue
        sp = s["spatial"]; wo = s["washout"]
        wtxt = f" | F0 cortex washout NIK {wo['nik_cortex_plateau']:.2e} vs real {wo['real_cortex_plateau']:.2e}" if wo else ""
        print(f"  sl{Z} F{F}: agree vp {s['agree_vp']:.2f} Ktrans {s['agree_kt']:.2f} | Ktrans bias {s['bias_kt_pct']:+.0f}% | rulerA {sp.get('rulerA_haarpsi',float('nan')):.3f}{wtxt}")

# ---- PRIMARY: inter-slice consistency (CoV per tissue) NIK-PK vs CS-fit ----
def cov(vals): vals = np.array(vals); return float(np.std(vals) / (np.mean(vals) + 1e-12) * 100)
print("\n=== INTER-SLICE consistency: CoV(%) of per-ROI value across slices 18/19/21 ===")
print(f"{'param':>7} {'tissue':>8} {'NIK_F0':>8} {'NIK_F2':>8} {'CS_fit':>8}")
consist = {}
for param in ["ktrans", "vp"]:
    for tis in ["cortex", "medulla", "aorta", "liver"]:
        row = {}
        for key, F in [("NIK_F0", 0), ("NIK_F2", 2)]:
            vals = [res[Z][F]["roivals"][tis][param] for Z in SLICES if F in res[Z] and not res[Z][F].get("missing") and tis in res[Z][F]["roivals"]]
            row[key] = cov(vals) if len(vals) == len(SLICES) else float("nan")
        vals_cs = [res[Z]["cs_fit"]["roivals"][tis][param] for Z in SLICES if tis in res[Z]["cs_fit"]["roivals"]]
        row["CS_fit"] = cov(vals_cs) if len(vals_cs) == len(SLICES) else float("nan")
        consist[f"{param}_{tis}"] = row
        print(f"{param:>7} {tis:>8} {row['NIK_F0']:>8.0f} {row['NIK_F2']:>8.0f} {row['CS_fit']:>8.0f}")

json.dump({str(Z): {str(F): {k: v for k, v in res[Z].get(F, {}).items() if k not in ('vp','kt')} for F in [0,2] if F in res[Z]} for Z in SLICES} | {"consistency": consist}, open(f"{D}/task2{_TAG}.json", "w"), indent=1, default=float)

# ---- figure: Ktrans maps per slice (F0) + inter-slice CoV bar ----
fig, ax = plt.subplots(2, len(SLICES)+1, figsize=(4*(len(SLICES)+1), 8))
for j, Z in enumerate(SLICES):
    if 0 in res[Z] and not res[Z][0].get("missing"):
        kt = res[Z][0]["kt"]; ax[0, j].imshow(np.rot90(kt), cmap="viridis", vmax=np.percentile(kt, 99)); ax[0, j].set_title(f"sl{Z} NIK-PK F0 Ktrans", fontsize=9)
    ax[0, j].axis("off")
    ktc = res[Z]["cs_fit"]["kt_cs"]; ax[1, j].imshow(np.rot90(ktc), cmap="viridis", vmax=np.percentile(ktc, 99)); ax[1, j].set_title(f"sl{Z} CS-fit Ktrans", fontsize=9); ax[1, j].axis("off")
# CoV bars for Ktrans
tissues = ["cortex", "medulla", "aorta", "liver"]; x = np.arange(len(tissues)); w = 0.27
ax[0, -1].bar(x-w, [consist[f"ktrans_{t}"]["NIK_F0"] for t in tissues], w, label="NIK F0")
ax[0, -1].bar(x, [consist[f"ktrans_{t}"]["NIK_F2"] for t in tissues], w, label="NIK F2")
ax[0, -1].bar(x+w, [consist[f"ktrans_{t}"]["CS_fit"] for t in tissues], w, label="CS-fit")
ax[0, -1].set_xticks(x); ax[0, -1].set_xticklabels(tissues, fontsize=8); ax[0, -1].legend(fontsize=7); ax[0, -1].set_title("Ktrans inter-slice CoV %", fontsize=9); ax[0, -1].grid(alpha=.3, axis="y")
ax[1, -1].bar(x-w, [consist[f"vp_{t}"]["NIK_F0"] for t in tissues], w, label="NIK F0")
ax[1, -1].bar(x, [consist[f"vp_{t}"]["NIK_F2"] for t in tissues], w, label="NIK F2")
ax[1, -1].bar(x+w, [consist[f"vp_{t}"]["CS_fit"] for t in tissues], w, label="CS-fit")
ax[1, -1].set_xticks(x); ax[1, -1].set_xticklabels(tissues, fontsize=8); ax[1, -1].legend(fontsize=7); ax[1, -1].set_title("vp inter-slice CoV %", fontsize=9); ax[1, -1].grid(alpha=.3, axis="y")
fig.suptitle("TASK 2: NIK-PK Ktrans maps vs CS-fit + inter-slice consistency (lower CoV = more consistent)", fontweight="bold")
fig.tight_layout(); p = fpath(f"task2_pk_consistency{_TAG}.png"); fig.savefig(p, dpi=130); print(f"\nwrote {p.split('/')[-1]}")
