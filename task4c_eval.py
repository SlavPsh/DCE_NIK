"""TASK 4C offline eval for ONE run (width,seed). For every checkpoint: measured-domain k-space
NMSE on TRAIN and VALIDATION (+radial shells), canonical Path C coefficient extraction, and
retrospective truth metrics (NOT used for selection). Selects best checkpoint by VALIDATION NMSE
only; evaluates the untouched TEST spokes for best + final. Saves everything for aggregation.
usage: python task4c_eval.py --width 512 --seed 0"""
import warnings; warnings.filterwarnings("ignore")
import argparse, os, glob, numpy as np, torch
import task4c_common as K

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--width", type=int, required=True); ap.add_argument("--seed", type=int, required=True); a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    S, M = K.load()
    _, _, _, _, nz, dims = K.build_train(S, dev)                      # deterministic normalizer (train spokes)
    C = dims[3]; model = K.make_model(a.width, a.seed, C, dev); model.eval()
    rundir = f"{K.OUT}/checkpoints/w{a.width}_s{a.seed}"
    cks = sorted(glob.glob(f"{rundir}/ck_*.pt"), key=lambda p: int(p.split("ck_")[-1].split(".")[0]))
    assert cks, f"no checkpoints in {rundir}"
    steps, trn, val, vsh, tru = [], [], [], [], []
    thetas = []
    for p in cks:
        ck = torch.load(p, map_location=dev, weights_only=False); model.load_state_dict(ck["state_dict"]); step = ck["step"]
        tr = K.kspace_nmse(model, nz, S, M["train"], dev); va = K.kspace_nmse(model, nz, S, M["val"], dev)
        thC = K.extract_pathC(model, nz, S, dev); tm = K.truth_metrics(thC, S)
        steps.append(step); trn.append(tr["nmse"]); val.append(va["nmse"])
        vsh.append([va["nmse_inner"], va["nmse_mid"], va["nmse_outer"]]); tru.append([tm["R_aorta"], tm["aorta_curve_nrmse"], tm["intAIF_nrmse"], tm["FP_energy"], tm["scale_abs"], tm["scale_phase_deg"]])
        thetas.append(thC)
        print(f"  w{a.width} s{a.seed} step {step:5d}: train {tr['nmse']:.3e} val {va['nmse']:.3e} | R_aorta {tm['R_aorta']:.3f} FP {tm['FP_energy']:.3f}", flush=True)
    steps = np.array(steps); val = np.array(val); trn = np.array(trn); vsh = np.array(vsh); tru = np.array(tru)
    best = int(np.argmin(val)); final = len(steps) - 1                # selection: validation NMSE ONLY
    # TEST evaluation (untouched) at best + final
    def test_at(i):
        ck = torch.load(cks[i], map_location=dev, weights_only=False); model.load_state_dict(ck["state_dict"])
        return K.kspace_nmse(model, nz, S, M["test"], dev)
    tb, tf = test_at(best), test_at(final)
    np.savez(f"{K.OUT}/arrays/eval_w{a.width}_s{a.seed}.npz",
             width=a.width, seed=a.seed, steps=steps, train_nmse=trn, val_nmse=val, val_shells=vsh,
             truth=tru, truth_cols=np.array(["R_aorta","aorta_curve","intAIF","FP","scale_abs","scale_phase"]),
             best_idx=best, best_step=int(steps[best]), final_step=int(steps[final]),
             test_best=[tb["nmse"], tb["nmse_inner"], tb["nmse_mid"], tb["nmse_outer"]],
             test_final=[tf["nmse"], tf["nmse_inner"], tf["nmse_mid"], tf["nmse_outer"]],
             thetaC_best=thetas[best], thetaC_final=thetas[final], params=np.array([K.param_counts(model)[k] for k in ("total","spatial","coil","temporal")]))
    print(f"DONE w{a.width} s{a.seed}: best_step={steps[best]} val={val[best]:.3e} test_best={tb['nmse']:.3e} test_final={tf['nmse']:.3e} R_aorta_best={tru[best,0]:.3f}", flush=True)

if __name__ == "__main__": main()
