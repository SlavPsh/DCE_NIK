"""in-vivo counterpart of xph_v2_vs_nik: grasp v2 at its BEST spokes/frame (NLINE) for lam 0.25 and
0.02, vs the spoke-matched NIK and the model-free nufft reference. slice 21.

NO ground truth in vivo, so the method-neutral model-free nufft recon stands in. it uses a 31-spoke
sliding window (~6.8 s effective temporal resolution), so it anchors images and bolus amplitude but
cannot adjudicate fine temporal detail. consistency, not accuracy.

best NLINE per lam by distance to the ideal corner on the (haarpsi, aorta-nrmse) plane, same rule as
the phantom, so the choice is not a per-metric pick.
out: figures/fig_v2_vs_nik_invivo.png + v2_sweep_invivo/v2_vs_nik_invivo.json
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, json, glob
import numpy as np, torch
sys.path.insert(0, "/scratch/rnga/vvpshenov/DCE_NIK")
from masked_metrics import haarpsi_masked
import consolidated as C
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

B = "/scratch/rnga/vvpshenov/DCE_NIK"; GV = "/scratch/rnga/vvpshenov/grasp_v2/results_grasp_v2"
OUTD = f"{B}/v2_sweep_invivo"; FIG = f"{B}/results/realdata_nik_vs_cs_figures/figures"
os.makedirs(OUTD, exist_ok=True)
TA, NTV = 375.0, 1710
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ctx = C.slice_ctx(21); rois = ctx["rois"]; body = ctx["BODY"]
z = np.load(f"{B}/step2_slice21.npz")
mf = np.abs(z["mf"]).transpose(1, 2, 0).astype(np.float32); tmf = z["tmf"].astype(np.float64)
mt = torch.from_numpy(body.astype(np.float32))[None, None].to(dev)
ROIS = [r for r in ("aorta", "cortex", "medulla") if rois.get(r) is not None and rois[r].sum() > 0]

def ftimes(nt):
    e = np.linspace(0, TA, nt + 1); return 0.5 * (e[:-1] + e[1:])

def refwin(nt):
    e = np.linspace(0, TA, nt + 1)
    out = np.zeros((mf.shape[0], mf.shape[1], nt), np.float32)
    for g in range(nt):
        m = (tmf >= e[g]) & (tmf < e[g + 1])
        if not m.any():
            m = np.zeros_like(tmf, bool); m[np.argmin(np.abs(tmf - 0.5*(e[g]+e[g+1])))] = True
        out[:, :, g] = mf[:, :, m].mean(2)
    return out

def bsub(c): return c - np.median(c[:8])
mf_c = {r: bsub(np.array([mf[..., i][rois[r]].mean() for i in range(mf.shape[-1])])) for r in ROIS}

def score(rec):
    nt = rec.shape[-1]; R = refwin(nt)
    s = np.sum(rec[body]*R[body]) / (np.sum(rec[body]**2) + 1e-12); rec = rec * s
    hs = []
    for t in range(0, nt, max(1, nt//40)):
        vmax = float(np.percentile(R[:, :, t][body], 99.5))
        pt = torch.from_numpy(np.clip(rec[:, :, t]/(vmax+1e-12), 0, 1)[None, None]).float().to(dev)
        rt = torch.from_numpy(np.clip(R[:, :, t]/(vmax+1e-12), 0, 1)[None, None]).float().to(dev)
        hs.append(float(haarpsi_masked(pt, rt, mt, data_range=1.0).cpu()))
    tG = ftimes(nt); out = dict(haarpsi=float(np.mean(hs)))
    for r in ROIS:
        c = np.interp(tmf, tG, np.array([rec[..., i][rois[r]].mean() for i in range(nt)]))
        out[f"{r}_nrmse"] = float(np.linalg.norm(bsub(c)-mf_c[r]) / (np.linalg.norm(mf_c[r])+1e-12))
    ca = np.interp(tmf, tG, np.array([rec[..., i][rois["aorta"]].mean() for i in range(nt)]))
    out["aorta_pk_ratio"] = float(bsub(ca).max() / (mf_c["aorta"].max()+1e-12))
    return out, rec

rows = []
for f in sorted(glob.glob(f"{GV}/gv2_slice21_n*.npy")):
    bn = os.path.basename(f)[:-4]
    NL = int(bn.split("_n")[1][:2]); lam = float(bn.split("_lam")[1]) if "_lam" in bn else 0.25
    m, _ = score(np.abs(np.load(f)).astype(np.float32))
    rows.append(dict(NLINE=NL, lam=lam, dt=NL*TA/NTV, npy=f, **m))
print(f"{'NLINE':>6} {'lam':>6} {'s/fr':>6} {'HaarPSI':>8} " + " ".join(f"{r+'C':>9}" for r in ROIS) + f" {'pk/mf':>7}")
for r in sorted(rows, key=lambda z: (z["lam"], z["NLINE"])):
    print(f"{r['NLINE']:>6} {r['lam']:>6.2f} {r['dt']:>6.2f} {r['haarpsi']:>8.4f} "
          + " ".join(f"{r[x+'_nrmse']:>9.4f}" for x in ROIS) + f" {r['aorta_pk_ratio']:>7.4f}")

sel = {}
for lam in (0.25, 0.02):
    c = [r for r in rows if abs(r["lam"]-lam) < 1e-9]
    if c: sel[lam] = min(c, key=lambda r: (1-r["haarpsi"])**2 + r["aorta_nrmse"]**2)
print("\nbest NLINE per lam (ideal-corner rule):")
for lam, r in sel.items():
    print(f"  lam {lam:<5}: NLINE {r['NLINE']} ({r['dt']:.2f} s/frame), haarpsi {r['haarpsi']:.4f}, aortaC {r['aorta_nrmse']:.4f}")

M = [("model-free ref", mf, tmf)]
nik = np.abs(np.load(f"{B}/results_full_sl21_matched/nik_slice_21.npy")).astype(np.float32)
_, nikS = score(nik); M.append(("NIK matched (1708 spokes)", nikS, ftimes(nik.shape[-1])))
for lam, r in sel.items():
    v = np.abs(np.load(r["npy"])).astype(np.float32); _, vs = score(v)
    M.append((f"GRASP-v2 {r['NLINE']}sp/fr lam{lam:g}", vs, ftimes(v.shape[-1])))

tab = []
for nm, v, t in M:
    row = dict(method=nm, frames=int(v.shape[-1]))
    for r in ROIS:
        c = np.interp(tmf, t, np.array([v[..., i][rois[r]].mean() for i in range(v.shape[-1])]))
        row[f"{r}_nrmse"] = float(np.linalg.norm(bsub(c)-mf_c[r])/(np.linalg.norm(mf_c[r])+1e-12))
    ca = np.interp(tmf, t, np.array([v[..., i][rois["aorta"]].mean() for i in range(v.shape[-1])]))
    row["aorta_pk_ratio"] = float(bsub(ca).max()/(mf_c["aorta"].max()+1e-12))
    row["aorta_ttp"] = float(tmf[np.argmax(bsub(ca))]); tab.append(row)
json.dump(dict(table=tab, sweep=[{k: v for k, v in r.items() if k != "npy"} for r in rows]),
          open(f"{OUTD}/v2_vs_nik_invivo.json", "w"), indent=1)
print(f"\n{'method':30} {'frames':>7} " + " ".join(f"{r+'C':>9}" for r in ROIS) + f" {'pk/mf':>7} {'ttp':>6}")
for r in tab:
    print(f"{r['method']:30} {r['frames']:>7} " + " ".join(f"{r[x+'_nrmse']:>9.4f}" for x in ROIS)
          + f" {r['aorta_pk_ratio']:>7.4f} {r['aorta_ttp']:>6.1f}")

PHASES = [("pre ~20s", 20.0), ("peak ~65s", 65.0), ("late ~250s", 250.0)]
n = len(M); fig = plt.figure(figsize=(3.1*n, 10.6))
gs = fig.add_gridspec(4, n, height_ratios=[1, 1, 1, 1.35], hspace=0.16, wspace=0.04)
vmax = float(np.percentile(mf[body], 99.0))
for pi, (plab, pt) in enumerate(PHASES):
    for mi, (nm, v, t) in enumerate(M):
        ax = fig.add_subplot(gs[pi, mi]); i = int(np.argmin(np.abs(np.asarray(t)-pt)))
        ax.imshow(v[:, :, i], cmap="gray", vmin=0, vmax=vmax); ax.axis("off")
        if pi == 0: ax.set_title(nm, fontsize=8)
        if mi == 0: ax.text(-0.08, 0.5, plab, transform=ax.transAxes, rotation=90, va="center", fontsize=9)
COLS = ["k", "#7c3aed", "#c0392b", "#e67e22"]
for ri, r in enumerate(ROIS):
    ax = fig.add_subplot(gs[3, ri*n//len(ROIS):(ri+1)*n//len(ROIS)])
    for mi, (nm, v, t) in enumerate(M):
        c = np.interp(tmf, t, np.array([v[..., i][rois[r]].mean() for i in range(v.shape[-1])]))
        ax.plot(tmf, bsub(c), color=COLS[mi % 4], lw=2.2 if mi == 0 else 1.5, ls="-" if mi == 0 else "--", label=nm)
    ax.set_title(f"{r} enhancement", fontsize=9); ax.set_xlabel("time (s)", fontsize=8)
    ax.grid(alpha=.3); ax.tick_params(labelsize=7)
    if ri == 0: ax.legend(fontsize=6, loc="upper right")
fig.suptitle("real in-vivo slice 21, NO ground truth. reference = model-free nufft (31-spoke sliding window,\n"
             "~6.8 s effective temporal resolution). grasp v2 at its BEST spokes/frame for each lam. "
             "NIK and grasp both on all acquired spokes. consistency, not accuracy.", fontsize=9)
fig.savefig(f"{FIG}/fig_v2_vs_nik_invivo.png", dpi=125, bbox_inches="tight")
print(f"\nSAVED {FIG}/fig_v2_vs_nik_invivo.png")
print("INVIVO_V2_VS_NIK_DONE")
