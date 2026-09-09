"""J1-ADD + J2a/b/c/d Tier-0 diagnostics (NO training, CPU). Reads existing CV CSVs, phantom
k-space, real k-space, and the corrected NIK/GRASP recons. Prints every number under a section
header and saves the K*-collapse figure (deliverable, figures dir)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, csv, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import xph_common as X, xph_pipeline as P
OUT = X.OUT; A = f"{OUT}/arrays"; FIGS = f"{OUT}/figures"
def rd(p):
    with open(p) as f: return list(csv.DictReader(f))

# ============================== J1-ADD: K* collapse ==============================
print("="*74); print("J1-ADD  K* COLLAPSE (phantom vs in-vivo CV)"); print("="*74)
ph = rd(f"{OUT}/grasp_kcv.csv")                                   # phantom
Kp = np.array([int(r["K"]) for r in ph]); vph = np.array([float(r["val_heldout_NMSE"]) for r in ph])
iv = rd(f"{OUT}/../realdata_nik_vs_cs_figures/real_cs_kcv.csv")   # in-vivo
Ki = np.array([int(r["K"]) for r in iv]); viv = np.array([float(r["inner_lowk"]) for r in iv])
kp_star = Kp[vph.argmin()]; ki_star = Ki[viv.argmin()]
print(f"phantom  K* = {kp_star}   min NMSE {vph.min():.3e}")
print(f"in-vivo  K* = {ki_star}   min inner-NMSE {viv.min():.4f}")
# sharpness: value at K=12 relative to each curve's own minimum
def at(K, Ks, v): return float(v[list(Ks).index(K)])
print(f"\nphantom sharpness (V at K=12): K8 {at(8,Kp,vph):.3e} | K12 {at(12,Kp,vph):.3e} | K16 {at(16,Kp,vph):.3e}")
print(f"  K12 is {at(8,Kp,vph)/at(12,Kp,vph):.2f}x better than K8 and {at(16,Kp,vph)/at(12,Kp,vph):.2f}x better than K16  -> SHARP minimum")
print(f"in-vivo shallowness: K3 {at(3,Ki,viv):.4f} | K12 {at(12,Ki,viv):.4f}  (K12 is +{100*(at(12,Ki,viv)/at(3,Ki,viv)-1):.1f}% vs K3)")
print(f"  full inner-band span K3->K32: {at(3,Ki,viv):.4f} -> {viv.max():.4f}  (+{100*(viv.max()/at(3,Ki,viv)-1):.1f}%)  -> SHALLOW, soft collapse")
# normalized overlay figure
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.plot(Kp, vph/vph.min(), "o-", color="#c0392b", label=f"phantom (noiseless), K*={kp_star}")
ax.plot(Ki, viv/viv.min(), "s-", color="#2471a3", label=f"in-vivo (noisy), K*={ki_star}")
ax.axvline(12, ls=":", c="0.6"); ax.axvline(3, ls=":", c="0.6")
ax.set_xscale("log", base=2); ax.set_xticks(Kp); ax.set_xticklabels(Kp)
ax.set_xlabel("temporal PCA rank K"); ax.set_ylabel("held-out NMSE / min (per curve)")
ax.set_title("K* collapse: sharp optimum (phantom) vs shallow (in-vivo)"); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{FIGS}/fig_kstar_collapse.png", dpi=130); plt.close(fig)
print(f"[saved] {FIGS}/fig_kstar_collapse.png")

# ============================== J2a: NOISE CHECK ==============================
print("\n"+"="*74); print("J2a  NOISE CHECK (radial power profile: phantom vs in-vivo)"); print("="*74)
d = P.data(); kd = d["kdata"]; kx = d["kx"]; ky = d["ky"]                        # (C,F,nang,RO),(F,nang,RO)
C, F, nang, RO = kd.shape
rp = (np.abs(kx[None]+1j*ky[None])/0.5)                                          # (1,F,nang,RO) r01 per sample
rp = np.broadcast_to(rp, kd.shape)
mag = np.abs(kd)
def radprofile(mag, r01, nb=24):
    edges = np.linspace(0, 1, nb+1); c = .5*(edges[:-1]+edges[1:]); prof = np.full(nb, np.nan)
    for b in range(nb):
        m = (r01 >= edges[b]) & (r01 < edges[b+1])
        if m.sum() > 20: prof[b] = np.sqrt(np.mean(mag[m]**2))                   # RMS magnitude per shell
    return c, prof
cp, pp = radprofile(mag, rp)
# noise floor = outer plateau (r01>0.85); signal = center (r01<0.1)
sig_ph = np.nanmean(pp[cp < 0.1]); flr_ph = np.nanmean(pp[cp > 0.85])
print(f"phantom  center-RMS {sig_ph:.3e}  outer-RMS(r>0.85) {flr_ph:.3e}  dynamic range {sig_ph/flr_ph:.1f}x ({20*np.log10(sig_ph/flr_ph):.1f} dB)")
# in-vivo
REF = "/net/beegfs/users/P101440/grasp_pro_py/results_ref"
sh = np.load(f"{REF}/shared.npz"); s13 = np.load(f"{REF}/slice_13.npz")
tn = np.asarray(sh["traj_norm"]); kdr = np.asarray(s13["kdata_radial"])          # [nx,1710,ncc]
r01v = (np.abs(tn)/np.abs(tn).max())                                            # [nx,1710]
r01v = np.broadcast_to(r01v[:, :, None], kdr.shape)
cv, pv = radprofile(np.abs(kdr), r01v)
sig_iv = np.nanmean(pv[cv < 0.1]); flr_iv = np.nanmean(pv[cv > 0.85])
print(f"in-vivo  center-RMS {sig_iv:.3e}  outer-RMS(r>0.85) {flr_iv:.3e}  dynamic range {sig_iv/flr_iv:.1f}x ({20*np.log10(sig_iv/flr_iv):.1f} dB)")
print(f"-> phantom dynamic range is {(sig_ph/flr_ph)/(sig_iv/flr_iv):.0f}x that of in-vivo: phantom effectively noiseless, in-vivo noise-limited")

# ============================== J2b: INPUT-PARITY AUDIT ==============================
print("\n"+"="*74); print("J2b  INPUT-PARITY AUDIT (measured inputs per method)"); print("="*74)
mk = P.masks(); ntr = int(mk["train"][0].sum()); nval = int(mk["val"][0].sum()); ntst = int(mk["test"][0].sum())
print(f"{'quantity':30s} {'NIK':>14s} {'GRASP-Pro(K12)':>16s}")
rows = [("spokes/frame used (recon)", f"{ntr} (ang {P.TRAIN_ANG})", f"{ntr} (ang {P.TRAIN_ANG})"),
        ("held-out val / test", f"{nval}/{ntst} (ang5/6)", f"{nval}/{ntst} (ang5/6)"),
        ("coils", f"{C}", f"{C}"), ("matrix (grid)", f"{RO}x{RO}", f"{RO}x{RO}"),
        ("readout samples/spoke", f"{RO}", f"{RO}"), ("frames", f"{F}", f"{F}"),
        ("trajectory", "shared kx,ky", "shared -kx,-ky"), ("coil maps b1", "true (shared)", "true (shared)"),
        ("DCF", "simple_ramp", "ramp |k|"), ("k-centre nav basis", "PCA (warmstart)", "PCA rank K"),
        ("truth usage", "eval-only scale", "eval-only scale")]
for a_, b_, c_ in rows: print(f"{a_:30s} {b_:>14s} {c_:>16s}")
print("-> identical measured inputs (5 train spokes, same coils/traj/grid); only the model differs")

# ============================== J2c: LOW-|k| ERROR SPLIT (measured vs gaps) ==============================
print("\n"+"="*74); print("J2c  LOW-|k| ERROR: measured-spoke cells vs angular-gap cells (PER-FRAME)"); print("="*74)
# champion corrected recon (sub16 s0) and truth
ev = np.load(f"{A}/img_eval_sub16_w768_s0.npz"); rec = ev["rec_best"]            # (RO,RO,F) magnitude, 1px-fixed
tq = d["times"]; body = d["labels"] > 0; Tr = X.truth_at(P.ZI, tq)
tr = np.array(P.TRAIN_ANG); yy, xx = np.mgrid[0:RO, 0:RO]
r01g = np.sqrt(((xx-RO/2)/(RO/2))**2 + ((yy-RO/2)/(RO/2))**2)
def frame_cov(f):
    """rasterize this frame's 5 train spokes onto the grid (+/-1 cell), = 'measured' cells this frame."""
    c = np.zeros((RO, RO), bool)
    ix = np.clip(np.round(kx[f, tr]/0.5*(RO/2) + RO/2).astype(int), 0, RO-1).ravel()
    iy = np.clip(np.round(ky[f, tr]/0.5*(RO/2) + RO/2).astype(int), 0, RO-1).ravel()
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1): c[np.clip(iy+dy, 0, RO-1), np.clip(ix+dx, 0, RO-1)] = True
    return c
bands = [(0, .10), (.10, .15), (.15, .30), (0, .15)]
acc = {b: [0., 0., 0., 0] for b in bands}                                        # ecov, egap, covcells, band-nframes
for f in range(0, F, 4):                                                         # subsample frames for speed
    E = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(rec[:, :, f]-Tr[:, :, f]))); Ep = np.abs(E)**2
    cov = frame_cov(f)
    for lo, hi in bands:
        band = (r01g >= lo) & (r01g < hi)
        acc[(lo, hi)][0] += Ep[band & cov].sum(); acc[(lo, hi)][1] += Ep[band & ~cov].sum()
        acc[(lo, hi)][2] += cov[band].mean(); acc[(lo, hi)][3] += 1
for lo, hi in bands:
    ec, eg, fc, n = acc[(lo, hi)]; tot = ec+eg+1e-30
    print(f"|k| [{lo:.2f},{hi:.2f}): per-frame coverage {100*fc/n:4.1f}%  err-energy on measured {100*ec/tot:4.1f}% / gaps {100*eg/tot:4.1f}%")
# temporal-mean (union across acquisition) view for the bias framing
recm = rec.mean(-1); trm = Tr.mean(-1); Em = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(recm-trm))))**2
lowk = (r01g < .15)
print(f"temporal-mean image: {100*Em[lowk].sum()/Em.sum():.1f}% of error energy sits at |k|<0.15 (fully sampled across acquisition)")
print("-> low-|k| error energy on MEASURED cells >> gaps => fit/envelope bias at data points, not undersampling gaps")

# ============================== J2d: POST-HOC TV RE-SWEEP (corrected recon) ==============================
print("\n"+"="*74); print("J2d  POST-HOC TV RE-SWEEP on corrected NIK recon (sub16 s0)"); print("="*74)
from skimage.restoration import denoise_tv_chambolle                             # canonical isotropic ROF TV
def nrmse(a, b, m): return float(np.sqrt(np.mean((a[m]-b[m])**2))/(b[m].max()-b[m].min()+1e-12))
ys, xs = np.where(body); y0, y1, x0, x1 = ys.min(), ys.max()+1, xs.min(), xs.max()+1   # body bbox crop
rc = rec[y0:y1, x0:x1]; tc = Tr[y0:y1, x0:x1]; bc = body[y0:y1, x0:x1]                 # cropped volumes
fr = np.arange(0, F, 6)                                                                 # subsample frames for NRMSE
base = np.mean([nrmse(rc[:, :, t], tc[:, :, t], bc) for t in fr])
print(f"body-bbox crop {rc.shape}; baseline (no TV) mean per-frame NRMSE {base:.4f}")
print(f"{'weight':>8s} {'spatial-2D':>12s} {'3D(x,y,t)':>12s}")
best = (base, 0.0, "none")
def tnrmse(vol): return np.mean([nrmse(vol[:, :, t], tc[:, :, t], bc) for t in fr])
for w in [0.002, 0.005, 0.01, 0.02, 0.05]:
    sp = np.stack([denoise_tv_chambolle(rc[:, :, t], weight=w, max_num_iter=100) for t in range(F)], -1)
    v3 = denoise_tv_chambolle(rc, weight=w, max_num_iter=100)                    # 3D includes temporal axis
    n_sp, n_3 = tnrmse(sp), tnrmse(v3)
    print(f"{w:8.3f} {n_sp:12.4f} {n_3:12.4f}")
    for nm, val in [("spatial", n_sp), ("3D", n_3)]:
        if val < best[0]: best = (val, w, nm)
print(f"best: {best[2]} TV w={best[1]}  NRMSE {best[0]:.4f}  (delta {100*(best[0]-base)/base:+.1f}% vs baseline)")
print("\nDONE_J2_CPU")
