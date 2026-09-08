"""A2 multi-ROI temporal comparison, slice 13. three groups (low-resid / high-resid-organ /
high-resid-motion), separated by residual CHARACTER not magnitude. model-free NUFFT curves
preserve raw character. out: figures/A2_multiroi.png + printed table."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, scipy.ndimage as ndi, finufft, json, sys
from scipy.signal import savgol_filter
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/scratch/rnga/vvpshenov/grasp_pro_py")
from figpath import fig as fpath
D = "/scratch/rnga/vvpshenov/DCE_NIK"; REF = "/scratch/rnga/vvpshenov/grasp_pro_py/results_ref"
CSD = "/scratch/rnga/vvpshenov/grasp_pro_py/results_spoke_cs"; TA = 375.0
Phi5 = np.load(f"{D}/navK5_Phi5.npy")

# --- recons ---
full = np.abs(np.load(f"{D}/results_spoke_full_f100/nik_slice_13.npy")).astype(np.float32)  # [x,y,342]
r16 = np.abs(np.load(f"{D}/results_spoke_nik_f100/nik_slice_13.npy")).astype(np.float32)
cs = np.abs(np.load(f"{CSD}/cs_slice13_f100.npy")).astype(np.float32)                       # [x,y,122]
nt = full.shape[-1]; tN = np.linspace(0, TA, nt); tC = np.linspace(0, TA, cs.shape[-1])
m = full.mean(-1); body = m > np.quantile(m, 0.55)
base = full[..., tN < 50].mean(-1); enh = full - base[..., None]
ttp = tN[enh.argmax(-1)]; amp = enh.max(-1) / (base + 1e-6)

# --- structural residual map (smoothed voxel curves onto K=5) ---
P = Phi5; PtPi = np.linalg.inv(P.T @ P)
def resid_curve(c):
    c = np.atleast_2d(c); proj = (c @ P) @ PtPi @ P.T
    return np.linalg.norm(c - proj, axis=1) / (np.linalg.norm(c - c.mean(1, keepdims=True), axis=1) + 1e-9)
rs = resid_curve(savgol_filter(full[body], 21, 3, axis=1)); rmap = np.zeros_like(m); rmap[body] = rs
r_p = lambda mask: float((rs < np.median(resid_curve(savgol_filter(full[mask], 21, 3, axis=1)))).mean() * 100)

# --- ROIs (3 groups) ---
aorta = np.load(f"{D}/aorta_roi.npy"); liver = np.load(f"{D}/liver_roi.npy")
interior = ndi.binary_erosion(body, iterations=6)
periphery = ndi.binary_dilation(body, iterations=1) & ~ndi.binary_erosion(body, iterations=5)          # motion/edge control
liver_core = ndi.binary_erosion(liver, iterations=2)
# late-enhancing interior organ (bowel/mesentery), portal-phase vessel, highest-resid interior blob
late = interior & (ttp > 250) & (amp > np.quantile(amp[body], 0.6)); late = ndi.binary_opening(late, iterations=2)
portal = interior & (ttp > 60) & (ttp < 130) & (amp > np.quantile(amp[body], 0.85)); portal = ndi.binary_opening(portal, iterations=1)
hi_int = interior & (rmap > np.percentile(rs, 85))                                # organ-kinetics candidate
def largest(mask, n=1):
    l, k = ndi.label(mask)
    if not k: return mask
    sizes = ndi.sum(np.ones_like(l), l, range(1, k + 1)); keep = 1 + np.argsort(-sizes)[:n]
    return np.isin(l, keep)
ROIS = {"aorta": (aorta, "LOW"), "liver core": (largest(liver_core), "LOW"),
        "late-enh organ": (largest(late), "HIGH-organ?"), "portal vessel": (largest(portal), "HIGH-organ?"),
        "hi-resid interior": (largest(hi_int, 2), "HIGH-organ?"), "periphery rim": (periphery, "MOTION")}
ROIS = {k: v for k, v in ROIS.items() if v[0].sum() >= 15}

# --- model-free sliding-window NUFFT curves per ROI ---
sh = np.load(f"{REF}/shared.npz"); traj = np.asarray(sh["traj_norm"]).astype(np.complex64)
vt = np.asarray(sh["view_time"]).ravel().astype(np.float64); nx = int(sh["nx"]); bas = int(sh["bas"])
sl = np.load(f"{REF}/slice_13.npz"); kdata = np.asarray(sl["kdata_radial"]).astype(np.complex64)
b1 = np.asarray(sl["b1"]).astype(np.complex64); ncc = kdata.shape[2]; den = np.sum(np.abs(b1) ** 2, 2) + 1e-12
meta = json.load(open(f"{D}/results_nufft/meta.json")); SIGN = meta["sign"]; order = np.argsort(vt)
def win_img(idx):
    tr = traj[:, idx]; w = np.maximum(np.abs(tr), 1 / nx / 4)
    x = (SIGN * 2 * np.pi * tr.real).ravel().astype(np.float64); y = (SIGN * 2 * np.pi * tr.imag).ravel().astype(np.float64)
    acc = sum(finufft.nufft2d1(x, y, (kdata[:, idx, c] * w).astype(np.complex128).ravel(), (nx, nx), isign=1, eps=1e-4) * np.conj(b1[:, :, c]) for c in range(ncc))
    s = (nx - bas) // 2; return np.abs(acc / den)[s:s + bas, s:s + bas]
W = 41; wins = [(order[a:a + W]) for a in range(0, len(order) - W + 1, 10)]
tmf = np.array([vt[idx].mean() * TA for idx in wins]); mf_imgs = np.stack([win_img(idx) for idx in wins])
print("built model-free windows", mf_imgs.shape, flush=True)

def roic(v, t, mask): return np.array([v[..., i][mask].mean() for i in range(v.shape[-1])])
def nrm(t, c):
    b = c[t < 50].mean() if (t < 50).any() else c[0]; pk = c[(t > 20) & (t < 200)].max() if ((t > 20) & (t < 200)).any() else c.max()
    return (c - b) / (pk - b + 1e-9)
def stats(t, n, fpwin=200):
    sm = savgol_filter(n, min(11, len(n) - (1 - len(n) % 2)), 3) if len(n) > 11 else n
    fp = (t > 20) & (t < fpwin);
    if not fp.any(): return dict(ttp=np.nan, fwhm=np.nan, up=np.nan, osc=np.nan)
    ttp = t[fp][np.argmax(sm[fp])]; half = (sm > 0.5) & (t < ttp + 60)
    fw = (t[half].max() - t[half].min()) if half.any() else np.nan
    up = np.gradient(sm, t)[fp].max(); osc = float(np.std(n - sm) / (np.abs(sm).max() + 1e-9))
    return dict(ttp=float(ttp), fwhm=float(fw), up=float(up), osc=osc)

rows = []
COL = {"aorta": "#d11", "liver core": "#181", "late-enh organ": "#18c", "portal vessel": "#a1c", "hi-resid interior": "#e80", "periphery rim": "#555"}
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, (name, (mask, grp)) in zip(axes.ravel(), ROIS.items()):
    tm, cm = tmf, np.array([im[mask].mean() for im in mf_imgs])                 # model-free
    nmf = nrm(tm, cm); mf_sm = savgol_filter(nmf, 11, 3)
    # residual of the model-free ROI-mean, and its CHARACTER (raw-smooth gap + low-freq frac)
    rr_raw = float(resid_curve(np.interp(tN, tm, cm)[None])[0]); rr_sm = float(resid_curve(np.interp(tN, tm, savgol_filter(cm, 11, 3))[None])[0])
    fftmag = np.abs(np.fft.rfft(nmf - mf_sm)); character = "oscillatory/motion" if (rr_raw - rr_sm) > 0.15 else "smooth/kinetics"
    cN, cR, cCr = nrm(tN, roic(full, tN, mask)), nrm(tN, roic(r16, tN, mask)), nrm(tC, roic(cs, tC, mask))
    sN, sR, sC, sM = stats(tN, cN), stats(tN, cR), stats(tC, cCr), stats(tm, nmf)
    div = float(np.sqrt(np.mean((np.interp(tC, tN, cR) - cCr) ** 2)))            # NIK R16 vs CS divergence
    rows.append(dict(roi=name, grp=grp, nvox=int(mask.sum()), resid_sm=rr_sm, resid_raw=rr_raw, gap=rr_raw - rr_sm,
                     character=character, div_R16_CS=div, ttp_cs=sC["ttp"], ttp_r16=sR["ttp"], ttp_full=sN["ttp"],
                     fwhm_cs=sC["fwhm"], fwhm_r16=sR["fwhm"], osc_r16=sR["osc"], osc_full=sN["osc"]))
    ax.plot(tm, nmf, color="0.4", lw=1, alpha=.7, label="model-free")
    ax.plot(tC, cCr, color="#08a", lw=2, label="CS")
    ax.plot(tN, cR, color="#70c", lw=1.6, label="NIK R16")
    ax.plot(tN, cN, color="#e62", lw=1.3, alpha=.8, label="NIK full")
    ax.set_title(f"{name} [{grp}]\nresid {rr_sm:.2f}, {character}, div {div:.2f}", fontsize=9)
    ax.set_xlim(0, 260); ax.grid(alpha=.3); ax.legend(fontsize=6.5)
fig.suptitle("A2 multi-ROI temporal comparison, slice 13 (model-free / CS / NIK)", fontweight="bold")
fig.tight_layout(); p = fpath("A2_multiroi.png"); fig.savefig(p, dpi=135)
print(f"\n{'ROI':18}{'group':13}{'nvox':>5}{'resid_sm':>9}{'gap':>6}{'character':>20}{'div R16-CS':>11}{'osc R16':>8}")
for r in rows:
    print(f"{r['roi']:18}{r['grp']:13}{r['nvox']:5d}{r['resid_sm']:9.2f}{r['gap']:6.2f}{r['character']:>20}{r['div_R16_CS']:11.3f}{r['osc_r16']:8.3f}")
json.dump(rows, open(f"{D}/A2_rois.json", "w"), indent=1, default=float)
print(f"\nwrote {p.split('/')[-1]}")
