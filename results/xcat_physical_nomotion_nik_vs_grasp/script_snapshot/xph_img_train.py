"""Image-reconstruction NIK trainer (subspace rank-R warmstart, or free wire_ff) on the physical
no-motion XCAT. Same recipe as F0 lane; only the temporal model differs. Saves >=20 checkpoints.
usage: python xph_img_train.py --model wire_ff_subspace --rank 5 --warmstart --width 768 --seed 0"""
import warnings; warnings.filterwarnings("ignore")
import argparse, os, time, torch
import xph_pipeline as P
from nik_focal_loss import composable_kspace_loss
SHORT = {"wire_ff_subspace": "sub", "wire_ff": "free"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["wire_ff_subspace", "wire_ff"])
    ap.add_argument("--rank", type=int, default=5); ap.add_argument("--warmstart", action="store_true")
    ap.add_argument("--width", type=int, default=768); ap.add_argument("--k-sigma", type=float, default=P.FIX["k_sigma"])
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--steps", type=int, default=P.STEPS); ap.add_argument("--ckpt-every", type=int, default=2000)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_, Yn, T_, C_, nz, dims = P.build_train(dev); C = dims[3]
    model = P.make_model_g(a.model, a.width, a.k_sigma, a.seed, C, dev, rank=a.rank, warmstart=a.warmstart); model.train()
    tag = f"{SHORT[a.model]}{a.rank if a.model=='wire_ff_subspace' else ''}_w{a.width}_s{a.seed}"
    rundir = f"{P.OUT}/checkpoints/{tag}"; os.makedirs(rundir, exist_ok=True)
    pc = P.param_counts(model); print(f"{tag}: model {a.model} rank {a.rank} warmstart {a.warmstart} | params {pc}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=P.LR, weight_decay=P.WD); N = X_.shape[0]; t0 = time.time()
    def save(step):
        torch.save(dict(state_dict={k: v.cpu() for k, v in model.state_dict().items()}, model=a.model, rank=a.rank,
                        warmstart=a.warmstart, width=a.width, k_sigma=a.k_sigma, seed=a.seed, step=step, ncc=C), f"{rundir}/ck_{step:05d}.pt")
    save(0)
    for step in range(1, a.steps + 1):
        idx = torch.randint(0, N, (P.BATCH,), device=dev); opt.zero_grad(set_to_none=True)
        loss = composable_kspace_loss(model(X_[idx], T_[idx], C_[idx]), Yn[idx], dcf=torch.ones(P.BATCH, device=dev),
                                      use_dcf=False, dcf_power=0.0, use_focal=False, focal_warmup_progress=1.0)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % a.ckpt_every == 0: save(step)
        if step % 5000 == 0 or step == a.steps: print(f"  step {step} loss {float(loss):.4e} ({time.time()-t0:.0f}s)", flush=True)
    if a.steps % a.ckpt_every != 0: save(a.steps)
    print(f"DONE {tag}", flush=True)

if __name__ == "__main__": main()
