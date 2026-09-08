"""NIK offline eval for ONE run: per-checkpoint train/val k-space NMSE -> select best by VAL NMSE
only (no truth). At best+final: test kNMSE+shells, Path C dynamic, truth image/curve metrics
(rot180-aligned, ONE truth-derived global scale). usage: python xph_eval.py --tag w512_ks2.5_s0"""
import warnings; warnings.filterwarnings("ignore")
import argparse, glob, numpy as np, torch
import xph_pipeline as P, xph_common as X
# DCE phases (s): aorta peak ~28, cortex ~39
PHASES = dict(precontrast=(0, 18), first_pass=(18, 45), cortical=(45, 90), late=(90, 200))

def masked_nrmse(a, b, m): return float(np.sqrt(np.mean((a[m]-b[m])**2))/(b[m].max()-b[m].min()+1e-12))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parts = a.tag.split("_"); width = int(parts[0][1:]); ks = float(parts[1][2:]); seed = int(parts[2][1:])
    _, _, _, _, nz, dims = P.build_train(dev); C = dims[3]
    model = P.make_model(width, ks, seed, C, dev); model.eval()
    rundir = f"{P.OUT}/checkpoints/{a.tag}"
    cks = sorted(glob.glob(f"{rundir}/ck_*.pt"), key=lambda p: int(p.split("ck_")[-1].split(".")[0]))
    mk = P.masks(); steps, trn, val = [], [], []
    for p in cks:
        ck = torch.load(p, map_location=dev, weights_only=False); model.load_state_dict(ck["state_dict"])
        steps.append(ck["step"]); trn.append(P.kspace_nmse(model, nz, mk["train"], dev)["nmse"]); val.append(P.kspace_nmse(model, nz, mk["val"], dev)["nmse"])
        print(f"  {a.tag} step {ck['step']:5d} train {trn[-1]:.3e} val {val[-1]:.3e}", flush=True)
    steps = np.array(steps); val = np.array(val); trn = np.array(trn); best = int(np.argmin(val)); final = len(steps)-1
    d = P.data(); tq = d["times"]; body = d["labels"] > 0; R = X.rois(P.ZI, d["labels"]); Tr = X.truth_at(P.ZI, tq)
    rot = lambda im: im[::-1, ::-1]
    def full_eval(i):
        ck = torch.load(cks[i], map_location=dev, weights_only=False); model.load_state_dict(ck["state_dict"])
        tk = P.kspace_nmse(model, nz, mk["test"], dev)
        dyn, thC = P.reconstruct_pathC(model, nz, tq, dev); rec = np.abs(np.stack([rot(dyn[:, :, t]) for t in range(dyn.shape[2])], -1))
        s = np.sum(rec[body]*Tr[body])/(np.sum(rec[body]**2)+1e-12); rec = rec*s          # ONE global truth-derived scale
        per = np.array([masked_nrmse(rec[:, :, t], Tr[:, :, t], body) for t in range(len(tq))])
        ph = {k: float(per[(tq >= lo) & (tq < hi)].mean()) for k, (lo, hi) in PHASES.items()}
        curves = {nm: (rec[R[nm]].mean(0), Tr[R[nm]].mean(0)) for nm in ["aorta", "cortex", "medulla"]}
        cur_nrmse = {nm: float(np.linalg.norm(rc-tt)/(np.linalg.norm(tt)+1e-12)) for nm, (rc, tt) in curves.items()}
        return dict(test=tk, per_frame=per, phases=ph, cur_nrmse=cur_nrmse, curves=curves, rec=rec, thC=thC, scale=float(s))
    eb, ef = full_eval(best), full_eval(final)
    np.savez(f"{P.OUT}/arrays/nik_eval_{a.tag}.npz", tag=a.tag, width=width, k_sigma=ks, seed=seed, steps=steps,
             train_nmse=trn, val_nmse=val, best_idx=best, best_step=int(steps[best]), final_step=int(steps[final]),
             test_best=[eb["test"][k] for k in ("nmse", "inner", "mid", "outer")], test_final=[ef["test"][k] for k in ("nmse", "inner", "mid", "outer")],
             img_nrmse_mean_best=float(eb["per_frame"].mean()), img_nrmse_mean_final=float(ef["per_frame"].mean()),
             per_frame_best=eb["per_frame"], phases_best=np.array(list(eb["phases"].values())), phase_names=np.array(list(PHASES.keys())),
             cur_nrmse_best=np.array([eb["cur_nrmse"][k] for k in ("aorta", "cortex", "medulla")]),
             rec_best=eb["rec"].astype(np.float32), thC_best=eb["thC"], scale_best=eb["scale"],
             aorta_curve_rec=eb["curves"]["aorta"][0], aorta_curve_true=eb["curves"]["aorta"][1],
             cortex_curve_rec=eb["curves"]["cortex"][0], cortex_curve_true=eb["curves"]["cortex"][1],
             medulla_curve_rec=eb["curves"]["medulla"][0], medulla_curve_true=eb["curves"]["medulla"][1])
    print(f"DONE {a.tag}: best_step {steps[best]} val {val[best]:.3e} test {eb['test']['nmse']:.3e} imgNRMSE {eb['per_frame'].mean():.3f} aortaCurve {eb['cur_nrmse']['aorta']:.3f}", flush=True)

if __name__ == "__main__": main()
