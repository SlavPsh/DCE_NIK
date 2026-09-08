"""Truth-blind K selection for GRASP-Pro by held-out k-space cross-validation (NOT the oracle).
For each K: reconstruct from the training spokes (angles 0-4), then forward-project the recon onto the
MEASURED-BUT-UNUSED spokes (val angle 5, test angle 6) and score prediction NMSE. K* = argmin val-NMSE.
Uses only acquired data, never the truth image. Parallels how NIK is scored on the same held-out spokes.
Also prints the truth-derived aorta metrics per K so we can see whether the data-chosen K captures the
bolus (annotation only; not used for selection)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, cupy as cp, cufinufft
import xph_grasp_nufft as GN, xph_pipeline as P, xph_common as X
KS = [3, 5, 8, 12, 16, 24, 32]; EPS = 1e-5; SIGN = GN.SIGN
d = P.data(); kx = d["kx"]; ky = d["ky"]; kdata = d["kdata"]; b1 = d["b1"]
C, F, nang, RO = kdata.shape; c0 = RO // 2
val_a = P.VAL_ANG[0]; test_a = P.TEST_ANG[0]
tq = d["times"]; body = d["labels"] > 0; R = X.rois(P.ZI, d["labels"]); Tr = X.truth_at(P.ZI, tq)
_pre = tq < 18
def _bsub(c): return c - np.median(c[_pre])
truec_a = _bsub(np.median(Tr[R["aorta"]], 0))

# normalized coils, exactly as the operator uses them
b1c = cp.asarray(b1, cp.complex128); nrm = cp.sqrt(cp.max(cp.sum(cp.abs(b1c) ** 2, -1))); b1n = b1c / (nrm + 1e-12)
xx, yy = np.meshgrid(np.arange(RO) - RO // 2, np.arange(RO) - RO // 2, indexing="ij")
rrmask = cp.asarray((np.sqrt(xx ** 2 + yy ** 2) / (RO // 2)) > 1.0)
plan2 = cufinufft.Plan(2, (RO, RO), C, eps=EPS, isign=-1, dtype="complex128")

def predict_angle(dyn, a):
    """forward-project complex recon dyn [RO,RO,F] onto measured spokes at angle a -> pred,meas [F,C,RO]."""
    dg = cp.asarray(dyn, cp.complex128); preds = cp.empty((F, C, RO), cp.complex128); meas = cp.asarray(kdata[:, :, a, :].transpose(1, 0, 2), cp.complex128)
    for t in range(F):
        cimg = dg[:, :, t][:, :, None] * b1n; cimg[rrmask] = 0
        cimg = cp.ascontiguousarray(cp.transpose(cimg, (2, 0, 1)))
        cox = cp.asarray(SIGN * 2 * np.pi * (-kx[t, a]).ravel(), cp.float64); coy = cp.asarray(SIGN * 2 * np.pi * (-ky[t, a]).ravel(), cp.float64)
        plan2.setpts(cox, coy); preds[t] = plan2.execute(cimg)
    return preds, meas

def cv_nmse(pred, meas, s):
    return float(cp.sum(cp.abs(s * pred - meas) ** 2) / cp.sum(cp.abs(meas) ** 2))

rows = []
for K in KS:
    dyn, Phi = GN.reconstruct(k=K)                                   # complex recon, native operator frame
    pv, mv = predict_angle(dyn, val_a)                              # global complex gauge scale from val
    s = complex(cp.sum(cp.conj(pv) * mv) / (cp.sum(cp.abs(pv) ** 2) + 1e-30))
    nv = cv_nmse(pv, mv, s); pt, mt = predict_angle(dyn, test_a); nt = cv_nmse(pt, mt, s)
    # truth aorta (annotation only)
    rec = np.abs(dyn); rec = rec * float((rec[body] * Tr[body]).sum() / ((rec[body] ** 2).sum() + 1e-12))
    ac = _bsub(np.median(rec[R["aorta"]], 0)); peak = float(ac.max()); cnr = float(np.linalg.norm(ac - truec_a) / np.linalg.norm(truec_a))
    rows.append((K, nv, nt, peak, cnr)); print(f"[K={K:2d}] val-NMSE {nv:.4e} test-NMSE {nt:.4e} | (truth) aorta peak {peak:.3f} curve {cnr:.3f}", flush=True)

Kstar = min(rows, key=lambda r: r[1])[0]
print(f"\nCV-SELECTED K* = {Kstar} (argmin val-NMSE, truth-blind)")
import csv
with open(f"{X.OUT}/grasp_kcv.csv", "w", newline="") as fp:
    w = csv.writer(fp); w.writerow(["K", "val_heldout_NMSE", "test_heldout_NMSE", "aorta_peak_truth", "aorta_curve_truth"])
    for r in rows: w.writerow([r[0], f"{r[1]:.6e}", f"{r[2]:.6e}", f"{r[3]:.4f}", f"{r[4]:.4f}"])
print("SAVED grasp_kcv.csv"); print("KCV_DONE")
