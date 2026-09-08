"""Image-reconstruction NIK eval (subspace/free): per-checkpoint val kNMSE selection, then test +
truth image/curve metrics at best+final via reconstruct_g (rot180-aligned, ONE truth-derived scale).
usage: python xph_img_eval.py --tag sub5_w768_s0 --model wire_ff_subspace --rank 5 --width 768 --warmstart"""
import warnings; warnings.filterwarnings("ignore")
import argparse, glob, numpy as np, torch
import xph_pipeline as P, xph_common as X
PH = dict(precontrast=(0, 18), first_pass=(18, 45), cortical=(45, 90), late=(90, 200))
def mnr(a, b, m): return float(np.sqrt(np.mean((a[m]-b[m])**2))/(b[m].max()-b[m].min()+1e-12))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--rank", type=int, default=5); ap.add_argument("--width", type=int, default=768); ap.add_argument("--warmstart", action="store_true")
    a = ap.parse_args(); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, _, _, nz, dims = P.build_train(dev); C = dims[3]
    model = P.make_model_g(a.model, a.width, P.FIX["k_sigma"], 0, C, dev, rank=a.rank, warmstart=False); model.eval()
    rundir = f"{P.OUT}/checkpoints/{a.tag}"; cks = sorted(glob.glob(f"{rundir}/ck_*.pt"), key=lambda p: int(p.split("ck_")[-1].split(".")[0]))
    mk = P.masks(); steps, trn, val = [], [], []
    for p in cks:
        ck = torch.load(p, map_location=dev, weights_only=False); model.load_state_dict(ck["state_dict"])
        steps.append(ck["step"]); trn.append(P.kspace_nmse(model, nz, mk["train"], dev)["nmse"]); val.append(P.kspace_nmse(model, nz, mk["val"], dev)["nmse"])
        print(f"  {a.tag} step {ck['step']:5d} train {trn[-1]:.3e} val {val[-1]:.3e}", flush=True)
    steps = np.array(steps); val = np.array(val); best = int(np.argmin(val)); final = len(steps)-1
    d = P.data(); tq = d["times"]; body = d["labels"] > 0; R = X.rois(P.ZI, d["labels"]); Tr = X.truth_at(P.ZI, tq); rot = lambda im: im[::-1, ::-1]
    def full(i):
        ck = torch.load(cks[i], map_location=dev, weights_only=False); model.load_state_dict(ck["state_dict"])
        tk = P.kspace_nmse(model, nz, mk["test"], dev)
        dyn = P.reconstruct_g(model, nz, tq, dev); rec = np.abs(np.stack([rot(dyn[:, :, t]) for t in range(dyn.shape[2])], -1))
        s = np.sum(rec[body]*Tr[body])/(np.sum(rec[body]**2)+1e-12); rec = rec*s
        per = np.array([mnr(rec[:, :, t], Tr[:, :, t], body) for t in range(len(tq))])
        cur = {nm: float(np.linalg.norm(rec[R[nm]].mean(0)-Tr[R[nm]].mean(0))/(np.linalg.norm(Tr[R[nm]].mean(0))+1e-12)) for nm in ["aorta", "cortex", "medulla"]}
        return dict(test=[tk[k] for k in ("nmse", "inner", "mid", "outer")], per=per, rec=rec, cur=cur,
                    curves={nm: (rec[R[nm]].mean(0), Tr[R[nm]].mean(0)) for nm in ["aorta", "cortex", "medulla"]})
    eb = full(best); ef = full(final)
    np.savez(f"{P.OUT}/arrays/img_eval_{a.tag}.npz", tag=a.tag, model=a.model, rank=a.rank, width=a.width, steps=steps,
             train_nmse=np.array(trn), val_nmse=val, best_idx=best, best_step=int(steps[best]),
             test_best=eb["test"], test_final=ef["test"], img_nrmse_mean_best=float(eb["per"].mean()), per_frame_best=eb["per"],
             cur_nrmse_best=np.array([eb["cur"][k] for k in ("aorta", "cortex", "medulla")]), rec_best=eb["rec"].astype(np.float32),
             aorta_curve_rec=eb["curves"]["aorta"][0], aorta_curve_true=eb["curves"]["aorta"][1],
             cortex_curve_rec=eb["curves"]["cortex"][0], cortex_curve_true=eb["curves"]["cortex"][1],
             medulla_curve_rec=eb["curves"]["medulla"][0], medulla_curve_true=eb["curves"]["medulla"][1])
    print(f"DONE {a.tag}: best_step {steps[best]} val {val[best]:.3e} test {eb['test'][0]:.3e} imgNRMSE {eb['per'].mean():.4f} aortaCurve {eb['cur']['aorta']:.4f}", flush=True)

if __name__ == "__main__": main()
