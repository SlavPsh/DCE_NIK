"""Key evidence figure: the temporal-PCA navigator eigenspectrum shows the arterial bolus is a
sub-percent-variance component, so any variance-threshold subspace selection (the truth-blind way to
pick K) discards it. Truth aorta curve is used ONLY to locate which PC carries the bolus (annotation),
never to pick K. Panels: cumulative variance vs K (both navigators, thresholds + resulting K); scree
with the bolus-carrying PC highlighted."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import xph_pipeline as P, xph_common as X
FIG = f"{X.OUT}/figures"
d = P.data(); kdata = d["kdata"]; tq = d["times"]; C, F, nang, RO = kdata.shape; tr = np.array(P.TRAIN_ANG); c0 = RO//2
R = X.rois(P.ZI, d["labels"]); Tr = X.truth_at(P.ZI, tq)
_pre = tq < 18
def _bsub(c): return c - np.median(c[_pre])
aorta = _bsub(np.median(Tr[R["aorta"]], 0)); aorta = aorta/ (np.linalg.norm(aorta)+1e-9)

def eig(nav):
    w, V = np.linalg.eigh(np.cov(nav, rowvar=False)); o = np.argsort(-w)
    return w[o], V[:, o]                                    # eigenvalues desc, temporal eigenvectors [F,F]
navs = {"raw (per-spoke, C x 25)": np.abs(kdata[:, :, tr, c0-2:c0+3]).transpose(0,2,3,1).reshape(-1, F),
        "avg (per-frame, C x 5)":  np.stack([np.abs(kdata[:, fr, tr, c0-2:c0+3]).mean(1).reshape(-1) for fr in range(F)], 1)}
info = {}
for nm, nav in navs.items():
    w, V = eig(nav); frac = w/w.sum(); cum = np.cumsum(frac)
    # locate bolus PC: eigenvector whose |temporal loading| best matches the truth aorta enhancement
    corr = [abs(np.corrcoef(np.abs(V[:, k])-np.abs(V[:, k]).mean(), aorta-aorta.mean())[0,1]) for k in range(min(30, F))]
    bpc = int(np.nanargmax(corr))
    info[nm] = dict(frac=frac, cum=cum, bpc=bpc, bfrac=float(frac[bpc]))
    print(f"[{nm}] bolus PC index {bpc} (var frac {frac[bpc]:.2e}); K@99% {int(np.searchsorted(cum,0.99)+1)} K@99.9% {int(np.searchsorted(cum,0.999)+1)}", flush=True)

fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
col = {"raw (per-spoke, C x 25)": "C0", "avg (per-frame, C x 5)": "C1"}
for nm, I in info.items():
    kk = np.arange(1, len(I["cum"])+1)
    ax[0].plot(kk[:32], I["cum"][:32], "-o", ms=3, color=col[nm], label=nm)
    ax[0].scatter([I["bpc"]+1], [I["cum"][I["bpc"]]], color=col[nm], marker="*", s=160, zorder=5)
    ax[1].semilogy(kk[:32], I["frac"][:32], "-o", ms=3, color=col[nm], label=nm)
    ax[1].scatter([I["bpc"]+1], [I["frac"][I["bpc"]]], color=col[nm], marker="*", s=160, zorder=5)
for th in (0.99, 0.999): ax[0].axhline(th, color="k", ls=":", lw=0.8, alpha=0.5)
ax[0].axvline(5, color="grey", ls="--", lw=1, alpha=0.7); ax[0].axvline(12, color="red", ls="--", lw=1, alpha=0.7)
ax[0].text(5, 0.4, "K=5 fair", rotation=90, fontsize=8, color="grey", va="bottom")
ax[0].text(12, 0.4, "K=12 oracle", rotation=90, fontsize=8, color="red", va="bottom")
ax[0].set_xlabel("subspace rank K"); ax[0].set_ylabel("cumulative navigator variance"); ax[0].set_ylim(0.9, 1.001)
ax[0].set_title("variance selection stops at K=1-5; bolus PC (star) is far down the tail"); ax[0].legend(fontsize=8, loc="lower right")
ax[1].axvline(5, color="grey", ls="--", lw=1, alpha=0.7); ax[1].axvline(12, color="red", ls="--", lw=1, alpha=0.7)
ax[1].set_xlabel("PC index"); ax[1].set_ylabel("variance fraction (log)"); ax[1].set_title("scree: the bolus PC carries <0.1% variance")
ax[1].legend(fontsize=8)
fig.suptitle("why K=5 misses the aorta bolus: it is a low-variance temporal component. z15 phantom")
fig.tight_layout(); fig.savefig(f"{FIG}/fig_nav_spectrum.png", dpi=130); plt.close(fig)
print("SAVED", f"{FIG}/fig_nav_spectrum.png"); print("NAVFIG_DONE")
