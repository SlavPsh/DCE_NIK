"""Run GRASP-Pro on the physical no-motion XCAT (SLURM, full A100). Reconstruct, rot180-align to
truth, ONE truth-derived global scale, save dynamic + curves + orientation check + config."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, xph_grasp as GP, xph_pipeline as P, xph_common as X
dev = "cuda"
dyn, Phi, coeff = GP.reconstruct(iters=GP.ITERS, ls=GP.SPATIAL_TV, lt=GP.TEMPORAL_TV)
d = P.data(); tq = d["times"]; body = d["labels"] > 0; R = X.rois(P.ZI, d["labels"]); Tr = X.truth_at(P.ZI, tq)
rot = lambda im: im[::-1, ::-1]
rec = np.abs(dyn)  # GRASP is identity-oriented (cufinufft+negated-trajs), unlike NIK rot180
# orientation sanity (should confirm rot180 like NIK)
oc = {nm: float(np.corrcoef(fn(np.abs(dyn).mean(2))[body].ravel(), Tr.mean(2)[body].ravel())[0, 1])
      for nm, fn in [("identity", lambda x: x), ("rot180", rot), ("fliplr", np.fliplr)]}
print("grasp orient corr", {k: round(v, 3) for k, v in oc.items()}, flush=True)
s = np.sum(rec[body]*Tr[body])/(np.sum(rec[body]**2)+1e-12); rec = rec*s          # ONE global truth-derived scale
def mn(a, b, m): return float(np.sqrt(np.mean((a[m]-b[m])**2))/(b[m].max()-b[m].min()+1e-12))
PH = dict(precontrast=(0, 18), first_pass=(18, 45), cortical=(45, 90), late=(90, 200))
per = np.array([mn(rec[:, :, t], Tr[:, :, t], body) for t in range(len(tq))])
phases = {k: float(per[(tq >= lo) & (tq < hi)].mean()) for k, (lo, hi) in PH.items()}
curves = {nm: (rec[R[nm]].mean(0), Tr[R[nm]].mean(0)) for nm in ["aorta", "cortex", "medulla"]}
cur = {nm: float(np.linalg.norm(rc-tt)/(np.linalg.norm(tt)+1e-12)) for nm, (rc, tt) in curves.items()}
np.savez(f"{P.OUT}/arrays/grasp_recon.npz", rec=rec.astype(np.float32), Phi=Phi, per_frame=per,
         phases=np.array(list(phases.values())), phase_names=np.array(list(PH.keys())),
         cur_nrmse=np.array([cur[k] for k in ("aorta", "cortex", "medulla")]), scale=float(s), orient=str(oc),
         K=GP.K_PCA, spatial_tv=GP.SPATIAL_TV, temporal_tv=GP.TEMPORAL_TV, iters=GP.ITERS,
         aorta_curve_rec=curves["aorta"][0], aorta_curve_true=curves["aorta"][1],
         cortex_curve_rec=curves["cortex"][0], cortex_curve_true=curves["cortex"][1],
         medulla_curve_rec=curves["medulla"][0], medulla_curve_true=curves["medulla"][1])
print(f"GRASP done: imgNRMSE {per.mean():.3f} | aortaCurve {cur['aorta']:.3f} cortex {cur['cortex']:.3f} medulla {cur['medulla']:.3f}", flush=True)
print("GRASP_DONE", flush=True)
