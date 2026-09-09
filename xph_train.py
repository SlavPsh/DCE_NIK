"""NIK-F0 trainer for the physical no-motion XCAT. Saves >=20 checkpoints; NO in-job val/test/truth
eval (selection is offline via val k-space NMSE). Only --hidden-width, --k-sigma, --seed vary.
usage: python xph_train.py --hidden-width 512 --k-sigma 2.5 --seed 0 --steps 40000 --ckpt-every 2000"""
import warnings; warnings.filterwarnings("ignore")
import argparse, os, time, torch
import xph_pipeline as P
import nik_wandb as W
from nik_focal_loss import composable_kspace_loss

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden-width", type=int, default=512)
    ap.add_argument("--k-sigma", type=float, default=P.FIX["k_sigma"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=P.STEPS)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--model", default="wire_ff_patlak", choices=["wire_ff_patlak", "wire_ff_tofts"])
    ap.add_argument("--basis-file", default=None, help="wire_ff_tofts: basis npz (default P.TOFTS_BASIS)")
    ap.add_argument("--val-every", type=int, default=2000, help="log VAL k-space NMSE (selection ruler) every N steps")
    ap.add_argument("--no-wandb", action="store_true", help="skip wandb (project dce_nik, offline fallback)")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_, Yn, T_, C_, nz, dims = P.build_train(dev); C = dims[3]
    if a.model == "wire_ff_tofts": model = P.make_model_tofts(a.hidden_width, a.k_sigma, a.seed, C, dev, a.basis_file)
    else: model = P.make_model(a.hidden_width, a.k_sigma, a.seed, C, dev)
    model.train(); pc = P.param_counts(model)
    tag = f"w{a.hidden_width}_ks{a.k_sigma:g}_s{a.seed}" + (f"_tofts{model.rank}" if a.model == "wire_ff_tofts" else "")
    torch.cuda.reset_peak_memory_stats() if dev.type == "cuda" else None
    rundir = f"{P.OUT}/checkpoints/{tag}"; os.makedirs(rundir, exist_ok=True)
    print(f"{tag}: train {X_.shape[0]} samples | params {pc}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=P.LR, weight_decay=P.WD); N = X_.shape[0]; t0 = time.time()
    sim = os.environ.get("XPH_SIM", "nomotion")
    run = W.Run(f"xph_{sim}_{tag}", config=dict(vars(a), sim=sim, params=pc), group="xph_train", tags=[a.model],
                local_json=f"{P.OUT}/wandb_runs/{tag}.json", enabled=not a.no_wandb)
    best_v, best_step = float("inf"), 0
    def save(step):
        sd = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save(dict(state_dict=sd, hidden_width=a.hidden_width, k_sigma=a.k_sigma, seed=a.seed, step=step, ncc=C, params=pc,
                        model=a.model, basis_file=(a.basis_file or P.TOFTS_BASIS) if a.model == "wire_ff_tofts" else None,
                        wall_s=time.time() - t0, peak_gpu_mb=(torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else 0.0)),
                   f"{rundir}/ck_{step:05d}.pt")
    save(0)
    for step in range(1, a.steps + 1):
        idx = torch.randint(0, N, (P.BATCH,), device=dev); opt.zero_grad(set_to_none=True)
        loss = composable_kspace_loss(model(X_[idx], T_[idx], C_[idx]), Yn[idx], dcf=torch.ones(P.BATCH, device=dev),
                                      use_dcf=False, dcf_power=0.0, use_focal=False, focal_warmup_progress=1.0)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % a.ckpt_every == 0: save(step)
        if step % a.val_every == 0:
            model.eval(); v = P.kspace_nmse(model, nz, P.masks()["val"], dev); model.train()
            print(f"  step {step} VAL nmse {v}", flush=True)
            v0 = float(v["nmse"]); shells = {f"val_knmse_{k}": float(v[k]) for k in ("inner", "mid", "outer")}
            if v0 < best_v: best_v, best_step = v0, step
            run.log(dict(loss=float(loss), val_knmse=v0, best_val_knmse=best_v, best_step=best_step, wall_s=time.time() - t0,
                         peak_gpu_mb=W.peak_gpu_mb(), **shells), step=step)
        if step % 5000 == 0 or step == a.steps:
            print(f"  step {step} loss {float(loss):.4e} ({time.time()-t0:.0f}s)", flush=True)
            run.log(dict(loss=float(loss), wall_s=time.time() - t0), step=step)
    if a.steps % a.ckpt_every != 0: save(a.steps)
    print(f"DONE {tag}: {len(os.listdir(rundir))} checkpoints | wall {time.time()-t0:.0f}s | params {pc} | peak_gpu_MB {torch.cuda.max_memory_allocated()/2**20 if dev.type=='cuda' else 0:.0f}", flush=True)
    run.finish(best_val_knmse=best_v, best_step=best_step, params=pc, peak_gpu_mb=W.peak_gpu_mb(), wall_s=time.time() - t0,
               n_ckpt=len(os.listdir(rundir)))

if __name__ == "__main__": main()
