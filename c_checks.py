"""C1: why SSIM (small gap) and PSNR/HaarPSI (large gap) disagree -> decompose NIK error into global
scale / per-frame amplitude / smooth spatial bias / structural residual; PSNR after each; error energy
by |k| band and body-vs-background. C2: measure ROI-median 1px shift sensitivity on the ACTUAL recons
(not truth) and reconcile the 44x B4 miss. C3: signed peak/TTP/FWHM per variant + aorta ROI erode/dilate
sweep to test partial-volume vs real NIK overshoot bias. Corrected recons only. STOP after."""
import warnings; warnings.filterwarnings("ignore")
import glob, numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter, shift as ndshift, binary_dilation, binary_erosion
import xph_pipeline as P, xph_common as X
A = f"{X.OUT}/arrays"
d = P.data(); tq = d["times"]; body = d["labels"] > 0; R = X.rois(P.ZI, d["labels"]); Tr = X.truth_at(P.ZI, tq); RO = Tr.shape[0]
rv = float(Tr[body].max()-Tr[body].min()); fov = np.zeros((RO, RO), bool)
yy, xx = np.meshgrid(np.arange(RO)-RO//2, np.arange(RO)-RO//2, indexing="ij"); fov[np.sqrt(xx**2+yy**2) < RO//2] = True; bg = fov & ~body
_pre = tq < 18
def bsub(c): return c - np.median(c[_pre])
def ls(v): return v * float((v[body]*Tr[body]).sum()/((v[body]**2).sum()+1e-12))
def mnr(a, b): return float(np.sqrt(np.mean((a[body]-b[body])**2))/(rv+1e-12))
def psnr_body(a, b, m=None):
    m = body if m is None else m; return float(20*np.log10(rv/(np.sqrt(np.mean((a[m]-b[m])**2))+1e-12)))
def per_frame(rec, fn): return np.array([fn(rec[:, :, t], Tr[:, :, t]) for t in range(len(tq))])

FILES = {"NIK-F0": "nik_eval_w768_ks2.5_s*", "NIK-sub5": "img_eval_sub5_w768_s*", "NIK-sub12": "img_eval_sub12_w768_s*",
         "NIK-sub16": "img_eval_sub16_w768_s*", "NIK-free": "img_eval_free_w768_s*"}
def load_repr(v):
    fs = sorted(glob.glob(f"{A}/{FILES[v]}.npz")); recs = [ls(np.abs(np.load(f)["rec_best"]).astype(np.float64)) for f in fs]
    nr = [per_frame(r, mnr).mean() for r in recs]; return recs[int(np.argsort(nr)[len(nr)//2])]
REC = {v: load_repr(v) for v in FILES}; REC["GRASP-K12"] = ls(np.abs(np.load(f"{A}/grasp_recon.npz")["rec"]).astype(np.float64))

# ================= C1 =================
print("===== C1: SSIM-vs-PSNR divergence, error decomposition =====")
print(f"{'variant':10s} {'PSNRglob':>9s} {'PSNRpfLS':>9s} {'PSNRaffine':>11s} {'PSNRnobias':>11s} | {'Elo%':>6s} {'Emid%':>6s} {'Ehi%':>6s} | {'Ebody%':>7s} {'Ebg%':>6s}")
for v in ["NIK-sub12", "NIK-sub16", "NIK-free", "GRASP-K12"]:
    rec = REC[v]
    pg = float(per_frame(rec, psnr_body).mean())                                   # global-LS (already applied)
    # per-frame LS scale
    pf = np.stack([rec[:, :, t]*float((rec[body, t] if False else (rec[:, :, t][body]*Tr[:, :, t][body]).sum()/((rec[:, :, t][body]**2).sum()+1e-12))) for t in range(len(tq))], -1)
    ppf = float(per_frame(pf, psnr_body).mean())
    # per-frame affine a*rec+b
    aff = np.empty_like(rec)
    for t in range(len(tq)):
        x = rec[:, :, t][body]; y = Tr[:, :, t][body]; Aa = np.vstack([x, np.ones_like(x)]).T; ab, *_ = np.linalg.lstsq(Aa, y, rcond=None)
        aff[:, :, t] = ab[0]*rec[:, :, t]+ab[1]
    paf = float(per_frame(aff, psnr_body).mean())
    # smooth spatial bias removal (per frame, large-sigma gaussian of the error)
    nob = np.empty_like(rec)
    for t in range(len(tq)):
        e = rec[:, :, t]-Tr[:, :, t]; bias = gaussian_filter(e, sigma=8); nob[:, :, t] = rec[:, :, t]-bias
    pnb = float(per_frame(nob, psnr_body).mean())
    # error energy by |k| band (temporal-mean error)
    E = (rec-Tr).mean(2); Ef = np.abs(np.fft.fftshift(np.fft.fft2(E)))**2
    rr = np.sqrt(xx**2+yy**2)/(RO//2); lo = Ef[rr < 0.2].sum(); mid = Ef[(rr >= 0.2) & (rr < 0.6)].sum(); hi = Ef[rr >= 0.6].sum(); tot = lo+mid+hi+1e-30
    # error energy body vs background
    Eb = float((( (rec-Tr)**2)[body]).sum()); Ebg = float((((rec-Tr)**2)[bg]).sum()); et = Eb+Ebg+1e-30
    print(f"{v:10s} {pg:9.2f} {ppf:9.2f} {paf:11.2f} {pnb:11.2f} | {100*lo/tot:6.1f} {100*mid/tot:6.1f} {100*hi/tot:6.1f} | {100*Eb/et:7.1f} {100*Ebg/et:6.1f}")

# ================= C2 =================
print("\n===== C2: ROI-median 1px shift sensitivity, RECON vs TRUTH (reconcile B4 44x) =====")
def curve(vol, m): return bsub(np.median(vol[m], 0))
def rolln(vol, s): return np.stack([ndshift(vol[:, :, t], (s, s), order=1) for t in range(vol.shape[2])], -1)
print(f"{'ROI':8s} {'B4(truth 1px)':>13s} {'recon 1px sens':>15s} {'observed old->new':>18s}")
for nm in ["aorta", "cortex", "medulla"]:
    m = R[nm]; ct = curve(Tr, m); cts = curve(rolln(Tr, 1), m); b4 = float(np.linalg.norm(cts-ct)/(np.linalg.norm(ct)+1e-12))
    rec = REC["NIK-sub16"]; cn = curve(rec, m); co = curve(rolln(rec, -1), m)        # old = roll(new,-1)
    rsens = float(np.linalg.norm(co-cn)/(np.linalg.norm(cn)+1e-12))
    print(f"{nm:8s} {b4:13.4f} {rsens:15.4f} {'(= recon sens by constr.)':>18s}")

# ================= C3 =================
print("\n===== C3: signed first-pass peak/TTP/FWHM per variant (truth peak 0.772, TTP 27.4, FWHM 11.5) =====")
def fp(c):
    b = c[:20].mean(); n = c-b; pk = n.max(); ttp = tq[n.argmax()]; idx = np.where((n >= pk/2) & (tq < ttp+50))[0]
    return float(c.max()), float(ttp), (float(tq[idx[-1]]-tq[idx[0]]) if idx.size > 1 else 0.0)
tp, tt, tf = fp(curve(Tr, R["aorta"]))
print(f"{'variant':10s} {'peak':>6s} {'dPeak':>7s} {'TTP':>6s} {'dTTP':>6s} {'FWHM':>6s} {'dFWHM':>6s}")
for v in ["NIK-F0", "NIK-sub5", "NIK-sub12", "NIK-sub16", "NIK-free", "GRASP-K12"]:
    pk, ttp, fw = fp(curve(REC[v], R["aorta"])); print(f"{v:10s} {pk:6.3f} {pk-tp:+7.3f} {ttp:6.1f} {ttp-tt:+6.1f} {fw:6.1f} {fw-tf:+6.1f}")
print("\naorta ROI erode/dilate sweep (peak amplitude; partial-volume test):")
print(f"{'variant':10s} {'erode1':>7s} {'orig(33)':>9s} {'dil1':>7s} {'dil2':>7s} {'dil3':>7s}")
a0 = R["aorta"]
masks = {"erode1": binary_erosion(a0), "orig": a0, "dil1": binary_dilation(a0), "dil2": binary_dilation(a0, iterations=2), "dil3": binary_dilation(a0, iterations=3)}
print(f"{'truth':10s}" + "".join(f" {fp(curve(Tr, masks[k]))[0]:7.3f}" if k != 'orig' else f" {fp(curve(Tr, masks[k]))[0]:9.3f}" for k in ["erode1", "orig", "dil1", "dil2", "dil3"]))
for v in ["NIK-F0", "NIK-free", "GRASP-K12"]:
    print(f"{v:10s}" + "".join(f" {fp(curve(REC[v], masks[k]))[0]:7.3f}" if k != 'orig' else f" {fp(curve(REC[v], masks[k]))[0]:9.3f}" for k in ["erode1", "orig", "dil1", "dil2", "dil3"]))
print(f"(aorta mask sizes: erode1 {int(masks['erode1'].sum())}, orig {int(a0.sum())}, dil1 {int(masks['dil1'].sum())}, dil2 {int(masks['dil2'].sum())}, dil3 {int(masks['dil3'].sum())})")
print("\nC_CHECKS_DONE")
