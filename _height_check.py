import warnings; warnings.filterwarnings("ignore")
import numpy as np, glob, re
import xph_pipeline as P, xph_common as X
A = f"{X.OUT}/arrays"
d = P.data(); tq = d["times"]; body = d["labels"] > 0; R = X.rois(P.ZI, d["labels"]); Tr = X.truth_at(P.ZI, tq)
M = {"NIK-F0": np.load(f"{A}/nik_eval_w768_ks2.5_s0.npz")["rec_best"],
     "NIK-sub16": np.load(f"{A}/img_eval_sub16_w768_s0.npz")["rec_best"],
     "NIK-free": np.load(f"{A}/img_eval_free_w768_s0.npz")["rec_best"],
     "GRASP-Pro": np.load(f"{A}/grasp_recon.npz")["rec"]}
import h5py
rc = np.abs(np.array(h5py.File(X.SIM, "r")["results"]["images"]["Recon"]["img"])[:, P.ZI]).astype(np.float32)
cs = np.stack([X._embed(rc[i], rc.shape[1]) for i in range(rc.shape[0])], -1)
if np.corrcoef(cs.mean(2).ravel(), Tr.mean(2).ravel())[0,1] < np.corrcoef(cs[::-1,::-1].mean(2).ravel(), Tr.mean(2).ravel())[0,1]: cs = cs[::-1,::-1]
M["CS-file"] = cs
pre = tq < 18; late = tq > 120
def base_plat(curve): return np.median(curve[pre]), np.median(curve[late])
print("truth aorta base/plateau: %.3f / %.3f   cortex: %.3f / %.3f" % (
    *base_plat(np.median(Tr[R['aorta']],0)), *base_plat(np.median(Tr[R['cortex']],0))))
print("%-10s | %-22s | %-22s | %-22s" % ("method","aorta base/plat (globalscale)","cortex base/plat","noise-floor b (affine)"))
for m, V in M.items():
    sc = (V[body]*Tr[body]).sum()/((V[body]**2).sum()+1e-9); Vs = V*sc               # global LS scale (aggregate)
    ab, ap = base_plat(np.median(Vs[R['aorta']],0)); cb, cp = base_plat(np.median(Vs[R['cortex']],0))
    # affine a*V+b fit over body,all frames -> b = additive floor (magnitude bias)
    x = V[body].ravel().astype(np.float64); y = Tr[body].ravel().astype(np.float64)
    aa = np.polyfit(x, y, 1); b_floor = aa[1]                                          # intercept in truth units
    print("%-10s | %.3f / %.3f            | %.3f / %.3f            | %+.4f" % (m, ab, ap, cb, cp, b_floor))
