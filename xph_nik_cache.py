"""render the best NIK phantom configs ONCE onto the fine time grid and cache them, so the
spatial-vs-temporal frontier can score NIK on exactly the same window rulers as grasp v2.
seed-averaged in the COMPLEX domain, same as l3_rebaseline's 'cplx-seedavg' rows (the best rows).
out: arrays/nik_fine_<tag>.npy  [RO,RO,F] magnitude, truth-frame, unscaled
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, glob
import numpy as np, torch
sys.path.insert(0, "/scratch/rnga/vvpshenov/DCE_NIK")
import xph_pipeline as P, xph_common as X

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_, _, _, _, nz, dims = P.build_train(dev); C = dims[3]
d = P.data(); tq = d["times"]; F = len(tq)
A = f"{P.OUT}/arrays"
rot = lambda im: np.roll(im[::-1, ::-1], (1, 1), axis=(0, 1))

def load(kind, s, st, rank):
    tag = (f"sub{rank}_w768_s{s}" if kind == "sub" else (f"free_w768_s{s}" if kind == "free" else f"w768_ks2.5_s{s}"))
    p = f"{P.OUT}/checkpoints/{tag}/ck_{st}.pt"
    if not os.path.exists(p):
        p = sorted(glob.glob(f"{P.OUT}/checkpoints/{tag}/ck_*.pt"))[-1]
    if kind == "F0":
        m = P.make_model(768, P.FIX["k_sigma"], s, C, dev)
    else:
        m = P.make_model_g("wire_ff_subspace" if kind == "sub" else "wire_ff", 768, P.FIX["k_sigma"], 0, C, dev,
                           rank=rank, warmstart=False)
    m.load_state_dict(torch.load(p, map_location=dev, weights_only=False)["state_dict"]); m.eval()
    return m, kind

@torch.no_grad()
def render_tf(mk):
    m, kind = mk
    dyn = (P.reconstruct_pathC(m, nz, tq, dev)[0] if kind == "F0" else P.reconstruct_g(m, nz, tq, dev))
    return np.stack([rot(dyn[:, :, t]) for t in range(F)], -1)

CFG = [("sub16", "sub", 16, [(0, 24000), (1, 40000)]),
       ("free",  "free", 0, [(0, 40000), (1, 34000), (2, 36000)])]

if __name__ == "__main__":
    for name, kind, rank, seeds in CFG:
        out = f"{A}/nik_fine_{name}.npy"
        if os.path.exists(out):
            print(f"{name}: cached"); continue
        tfs = [render_tf(load(kind, s, st, rank)) for s, st in seeds]
        cplx_avg = np.abs(np.mean(tfs, 0))                       # complex seed-average, then magnitude
        np.save(out, cplx_avg.astype(np.float32))
        print(f"SAVED {out} {cplx_avg.shape} (n={len(seeds)} seeds, complex-averaged)", flush=True)
    print("NIK_CACHE_DONE")
