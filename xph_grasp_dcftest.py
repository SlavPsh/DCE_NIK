"""A/B test: GRASP dcf-preconditioned vs the current (no-dcf) recon, same 15 iters. tests the blur = under-convergence hypothesis."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import xph_grasp as GP, xph_pipeline as P, xph_common as X
d = P.data(); tq = d["times"]; body = d["labels"] > 0; Tr = X.truth_at(P.ZI, tq); A = f"{P.OUT}/arrays"
old = np.load(f"{A}/grasp_recon.npz")["rec"]                       # current blurry recon (already truth-scaled)
dyn, Phi, _ = GP.reconstruct(iters=GP.ITERS, use_dcf=True)         # NEW: dcf-preconditioned, same 15 iters
rec = np.abs(dyn); s = np.sum(rec[body] * Tr[body]) / (np.sum(rec[body] ** 2) + 1e-12); rec = rec * s
tm = Tr.mean(2)
def nrmse(a): return float(np.sqrt(np.mean((a.mean(2)[body] - tm[body]) ** 2)) / (tm[body].max() - tm[body].min()))
def gradE(im): return float((np.diff(im, axis=0) ** 2).sum() + (np.diff(im, axis=1) ** 2).sum())   # edge energy = sharpness
def hf(im):
    F2 = np.fft.fftshift(np.abs(np.fft.fft2(im * body))); n = im.shape[0]
    yy, xx = np.mgrid[:n, :n]; r = np.sqrt((yy - n / 2) ** 2 + (xx - n / 2) ** 2)
    return float(F2[r > n * 0.15].sum() / (F2.sum() + 1e-9))
print("%-22s | imgNRMSE | edgeE(sharpness) | HF-frac" % "variant")
print("%-22s |    -     | %.3e        | %.4f" % ("TRUTH", gradE(tm), hf(tm)))
print("%-22s | %.4f   | %.3e        | %.4f" % ("GRASP no-dcf (current)", nrmse(old), gradE(old.mean(2)), hf(old.mean(2))))
print("%-22s | %.4f   | %.3e        | %.4f" % ("GRASP dcf-precond", nrmse(rec), gradE(rec.mean(2)), hf(rec.mean(2))))
ip = np.argmin(abs(tq - 40)); vmax = np.percentile(Tr[body], 99)
fig, ax = plt.subplots(1, 3, figsize=(14, 5))
for a, im, ttl in [(ax[0], Tr[:, :, ip], "truth ~40s"), (ax[1], old[:, :, ip], "GRASP no-dcf (current, blurry)"), (ax[2], rec[:, :, ip], "GRASP dcf-precond (15 iters)")]:
    a.imshow(im, cmap="gray", vmax=vmax); a.set_title(ttl); a.axis("off")
fig.tight_layout(); o = f"{P.OUT}/figures/grasp_dcf_test.png"; fig.savefig(o, dpi=120); print("SAVED", o)
np.savez(f"{A}/grasp_dcf_recon.npz", rec=rec.astype(np.float32), Phi=Phi)
print("GRASP_DCF_TEST_DONE")
