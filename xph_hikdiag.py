"""STEP 1 diagnostic (no sweep). Two deliverables, then STOP:
(A) held-out-spoke (angles 5+6) NMSE RESOLVED PER |k| ANNULUS, for NIK and GRASP-K12. Global NMSE is
    energy-dominated by k-centre and can hide a high-|k| failure; per-annulus exposes it.
(B) radial k-space power spectrum of each recon vs XCAT ground truth (does NIK attenuate high spatial
    frequency = blur, or match truth).
NIK = sub12 (best NIK: best SSIM + best held-out fit) and free (pure continuous, secondary reference).
Frozen models; nothing tuned."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, cupy as cp, cufinufft
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import xph_pipeline as P, xph_common as X, xph_grasp_nufft as GN
OUT = X.OUT; FIG = f"{OUT}/figures"; A = f"{OUT}/arrays"; dev = torch.device("cuda")
d = P.data(); kx = d["kx"]; ky = d["ky"]; kdata = d["kdata"]; b1 = d["b1"]; C, F, nang, RO = kdata.shape
tq = d["times"]; body = d["labels"] > 0; Tr = X.truth_at(P.ZI, tq)
EDGES = np.linspace(0.0, 1.0, 17)                       # 16 |k| annuli, r=|k|/0.5 in [0,1]
CTR = 0.5 * (EDGES[:-1] + EDGES[1:])

# ---------- (A) held-out per-annulus NMSE ----------
held = P.masks(); held_mask = held["val"] | held["test"]         # angles 5 and 6

@torch.no_grad()
def nik_annuli(tag, model_type, rank):
    ev = np.load(f"{A}/img_eval_{tag}.npz"); bstep = int(ev["best_step"])
    _, _, _, _, nz, dims = P.build_train(dev); Cc = dims[3]
    model = P.make_model_g(model_type, 768, P.FIX["k_sigma"], 0, Cc, dev, rank=rank, warmstart=False); model.eval()
    import glob
    ck = [p for p in glob.glob(f"{OUT}/checkpoints/{tag}/ck_*.pt") if int(p.split("ck_")[-1].split(".")[0]) == bstep][0]
    model.load_state_dict(torch.load(ck, map_location=dev, weights_only=False)["state_dict"])
    X_, Y_, T_, C_, R, _ = P._dataset(held_mask, dev); n = X_.shape[0]; num = np.zeros(16); den = np.zeros(16); gn = 0.0; gd = 0.0
    for i in range(0, n, 200000):
        pr = nz.denormalize(X_[i:i+200000], model(X_[i:i+200000], T_[i:i+200000], C_[i:i+200000]))
        yh = (pr[:, 0] + 1j * pr[:, 1]).cpu().numpy(); yt = (Y_[i:i+200000, 0] + 1j * Y_[i:i+200000, 1]).cpu().numpy()
        e = np.abs(yh - yt) ** 2; p = np.abs(yt) ** 2; gn += e.sum(); gd += p.sum(); b = np.clip(np.digitize(R[i:i+200000], EDGES) - 1, 0, 15)
        for k in range(16): m = b == k; num[k] += e[m].sum(); den[k] += p[m].sum()
    print(f"  {tag}: global held-out NMSE {gn/gd:.3e}", flush=True); return num / (den + 1e-30), gn / gd

def grasp_annuli(k=12):
    dyn, _ = GN.reconstruct(k=k); dg = cp.asarray(dyn, cp.complex128)
    b1c = cp.asarray(b1, cp.complex128); b1n = b1c / (cp.sqrt(cp.max(cp.sum(cp.abs(b1c) ** 2, -1))) + 1e-12)
    xx, yy = np.meshgrid(np.arange(RO) - RO // 2, np.arange(RO) - RO // 2, indexing="ij")
    rrmask = cp.asarray(np.sqrt(xx ** 2 + yy ** 2) / (RO // 2) > 1.0)
    plan2 = cufinufft.Plan(2, (RO, RO), C, eps=1e-5, isign=-1, dtype="complex128")
    res = []; mea = []; rad = []
    for a in (P.VAL_ANG[0], P.TEST_ANG[0]):
        for t in range(F):
            cimg = dg[:, :, t][:, :, None] * b1n; cimg[rrmask] = 0; cimg = cp.ascontiguousarray(cp.transpose(cimg, (2, 0, 1)))
            plan2.setpts(cp.asarray(GN.SIGN * 2 * np.pi * (-kx[t, a]).ravel(), cp.float64), cp.asarray(GN.SIGN * 2 * np.pi * (-ky[t, a]).ravel(), cp.float64))
            pk = plan2.execute(cimg)                                  # [C,RO]
            res.append(cp.asnumpy(pk)); mea.append(kdata[:, t, a, :]); rad.append(np.abs(kx[t, a] + 1j * ky[t, a]) / 0.5)
    pred = np.stack(res, 0); meas = np.stack(mea, 0); r = np.stack(rad, 0)[:, None, :] * np.ones((1, C, 1))  # [N,C,RO]
    s = complex(np.sum(np.conj(pred) * meas) / (np.sum(np.abs(pred) ** 2) + 1e-30))                          # global gauge scale
    e = np.abs(s * pred - meas) ** 2; p = np.abs(meas) ** 2; gnmse = e.sum() / p.sum()
    b = np.clip(np.digitize(r.ravel(), EDGES) - 1, 0, 15); num = np.zeros(16); den = np.zeros(16); ef = e.ravel(); pf = p.ravel()
    for k in range(16): m = b == k; num[k] = ef[m].sum(); den[k] = pf[m].sum()
    print(f"  GRASP-K12: global held-out NMSE {gnmse:.3e} (gauge s={abs(s):.3e})", flush=True); return num / (den + 1e-30), float(gnmse)

nik12_ann, nik12_g = nik_annuli("sub12_w768_s1", "wire_ff_subspace", 12)
free_ann, free_g = nik_annuli("free_w768_s0", "wire_ff", 5)
grasp_ann, grasp_g = grasp_annuli(12)

# ---------- (B) radial power spectrum of recon vs truth ----------
def radial_pspec(vol, step=4):
    cy = vol.shape[0] // 2; xx, yy = np.meshgrid(np.arange(vol.shape[0]) - cy, np.arange(vol.shape[1]) - cy, indexing="ij")
    rr = np.sqrt(xx ** 2 + yy ** 2); nb = 60; bins = np.linspace(0, cy, nb + 1); idx = np.clip(np.digitize(rr.ravel(), bins) - 1, 0, nb - 1)
    cnt = np.bincount(idx, minlength=nb); acc = np.zeros(nb); nf = 0
    for t in range(0, vol.shape[2], step):
        pw = (np.abs(np.fft.fftshift(np.fft.fft2(vol[:, :, t]))) ** 2).ravel(); acc += np.bincount(idx, weights=pw, minlength=nb); nf += 1
    return 0.5 * (bins[:-1] + bins[1:]) / cy, acc / (cnt + 1e-9) / nf                  # normalized spatial freq [0,1], mean power

def scaled(v): return v * float((v[body] * Tr[body]).sum() / ((v[body] ** 2).sum() + 1e-12))
rec12 = scaled(np.abs(np.load(f"{A}/img_eval_sub12_w768_s1.npz")["rec_best"]).astype(np.float64))
recfr = scaled(np.abs(np.load(f"{A}/img_eval_free_w768_s0.npz")["rec_best"]).astype(np.float64))
recg = scaled(np.abs(np.load(f"{A}/grasp_recon.npz")["rec"]).astype(np.float64))
fq, ptru = radial_pspec(Tr.astype(np.float64)); _, p12 = radial_pspec(rec12); _, pfr = radial_pspec(recfr); _, pg = radial_pspec(recg)

# ---------- figures + tables ----------
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].semilogy(CTR, grasp_ann, "-o", label=f"GRASP-K12 (global {grasp_g:.2e})", color="C0")
ax[0].semilogy(CTR, nik12_ann, "-s", label=f"NIK-sub12 (global {nik12_g:.2e})", color="C3")
ax[0].semilogy(CTR, free_ann, "-^", label=f"NIK-free (global {free_g:.2e})", color="C1")
ax[0].set_xlabel("|k| / |k|max (radial annulus)"); ax[0].set_ylabel("held-out-spoke NMSE (per annulus)")
ax[0].set_title("(A) held-out data fit vs spatial frequency"); ax[0].grid(alpha=0.2, which="both"); ax[0].legend(fontsize=8)
ax[1].semilogy(fq, ptru, "k-", lw=2.5, label="XCAT truth")
ax[1].semilogy(fq, pg, "-", label="GRASP-K12", color="C0"); ax[1].semilogy(fq, p12, "-", label="NIK-sub12", color="C3"); ax[1].semilogy(fq, pfr, "-", label="NIK-free", color="C1")
ax[1].set_xlabel("spatial frequency / Nyquist"); ax[1].set_ylabel("radial power (mean over frames)")
ax[1].set_title("(B) image radial power spectrum vs truth"); ax[1].grid(alpha=0.2, which="both"); ax[1].legend(fontsize=8)
fig.suptitle("STEP 1 diagnostic: high-|k| held-out fit and image power spectrum. z15 phantom")
fig.tight_layout(); fig.savefig(f"{FIG}/fig_hikdiag.png", dpi=130); plt.close(fig)

print("\n(A) held-out-spoke NMSE per |k| annulus (r=|k|/kmax):")
print("  annulus:  " + "  ".join(f"{c:.2f}" for c in CTR))
for nm, v in [("GRASP-K12", grasp_ann), ("NIK-sub12", nik12_ann), ("NIK-free", free_ann)]:
    print(f"  {nm:10s}" + "  ".join(f"{x:.1e}" for x in v))
print("\n(B) radial power ratio recon/truth at high spatial freq (>0.6 Nyquist):")
hi = fq > 0.6
for nm, pp in [("GRASP-K12", pg), ("NIK-sub12", p12), ("NIK-free", pfr)]:
    print(f"  {nm:10s} mean power/truth = {np.mean(pp[hi]/(ptru[hi]+1e-30)):.3f}")
np.savez(f"{A}/hikdiag.npz", edges=EDGES, ctr=CTR, grasp_ann=grasp_ann, nik12_ann=nik12_ann, free_ann=free_ann,
         freq=fq, p_truth=ptru, p_grasp=pg, p_nik12=p12, p_free=pfr)
print("SAVED", f"{FIG}/fig_hikdiag.png", "+ hikdiag.npz"); print("HIKDIAG_DONE")
