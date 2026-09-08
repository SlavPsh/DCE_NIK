import warnings; warnings.filterwarnings("ignore")
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import xph_grasp_nufft as GN, xph_pipeline as P, xph_common as X
d = P.data(); tq = d["times"]; body = d["labels"] > 0; Tr = X.truth_at(P.ZI, tq); A = f"{P.OUT}/arrays"
dyn, Phi = GN.reconstruct(); rec0 = np.abs(dyn); tm = Tr.mean(2)
def corr(a, b): a = a[body].ravel() - a[body].mean(); b = b[body].ravel() - b[body].mean(); return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
def orient(v, nm): return {"id": v, "rot180": v[::-1, ::-1], "fliplr": v[:, ::-1], "flipud": v[::-1], "T": np.transpose(v, (1, 0, 2)), "rot90": np.rot90(v), "rot270": np.rot90(v, 3)}[nm]
best = max(["id", "rot180", "fliplr", "flipud", "T", "rot90", "rot270"], key=lambda nm: corr(orient(rec0, nm).mean(2), tm))
rec = orient(rec0, best); print("orientation best:", best, "corr %.3f" % corr(rec.mean(2), tm))
s = np.sum(rec[body] * Tr[body]) / (np.sum(rec[body] ** 2) + 1e-12); rec = rec * s
old = np.load(f"{A}/grasp_recon.npz")["rec"]
def gradE(im): return float((np.diff(im, axis=0) ** 2).sum() + (np.diff(im, axis=1) ** 2).sum())
def nrmse(a): return float(np.sqrt(np.mean((a.mean(2)[body] - tm[body]) ** 2)) / (tm[body].max() - tm[body].min()))
print("edgeE(sharpness): truth %.3e | old task4-FISTA %.3e | new nufft+pca+nlcg %.3e" % (gradE(tm), gradE(old.mean(2)), gradE(rec.mean(2))))
print("imgNRMSE: old %.4f | new %.4f" % (nrmse(old), nrmse(rec)))
ip = np.argmin(abs(tq - 40)); vmax = np.percentile(Tr[body], 99)
fig, ax = plt.subplots(1, 3, figsize=(14, 5))
for a, im, ttl in [(ax[0], Tr[:, :, ip], "truth ~40s"), (ax[1], old[:, :, ip], "old task4-FISTA (blurry)"), (ax[2], rec[:, :, ip], "new NUFFT+PCA+NLCG (true coils)")]:
    a.imshow(im, cmap="gray", vmax=vmax); a.set_title(ttl); a.axis("off")
fig.tight_layout(); o = f"{P.OUT}/figures/grasp_nufft_test.png"; fig.savefig(o, dpi=120); print("SAVED", o)
np.savez(f"{A}/grasp_nufft_recon.npz", rec=rec.astype(np.float32), Phi=Phi, orient=best, scale=float(s))
print("GRASP_NUFFT_DONE")
