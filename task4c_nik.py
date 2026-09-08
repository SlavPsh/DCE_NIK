"""TASK 4C trainer: NIK-F0 on the Task-4 f25 TRAIN spokes. Only --hidden-width and --seed vary;
all else frozen to Task-4. Saves >=20 checkpoints over training. NO validation/test/truth eval in
the job (checkpoint selection + truth are done offline in task4c_eval.py -> no leakage).
Default (no --hidden-width) reproduces the Task-4 width. usage:
  python task4c_nik.py --hidden-width 512 --seed 0 --steps 40000 --ckpt-every 2000"""
import warnings; warnings.filterwarnings("ignore")
import argparse, os, time, torch
import task4c_common as K
from nik_focal_loss import composable_kspace_loss

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden-width", type=int, default=K.CURRENT_W)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=K.STEPS)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=K.BATCH)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    S, _ = K.load()
    X, Yn, T, Ct, nz, dims = K.build_train(S, dev)
    F, NA, RO, C, Tt = dims
    model = K.make_model(a.hidden_width, a.seed, C, dev); model.train()
    pc = K.param_counts(model)
    rundir = f"{K.OUT}/checkpoints/w{a.hidden_width}_s{a.seed}"; os.makedirs(rundir, exist_ok=True)
    print(f"width={a.hidden_width} seed={a.seed} | train samples={X.shape[0]} | params tot={pc['total']} spatial={pc['spatial']} coil={pc['coil']} temporal={pc['temporal']} | samples/param={X.shape[0]/pc['total']:.3f}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=K.LR, weight_decay=K.WD)
    Ntr = X.shape[0]; t0 = time.time()
    def save(step):
        sd = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save(dict(state_dict=sd, hidden_width=a.hidden_width, seed=a.seed, step=step, ncc=C,
                        params=pc, model="wire_ff_patlak", patlak_free=0), f"{rundir}/ck_{step:05d}.pt")
    save(0)                                                            # step-0 (init) checkpoint
    for step in range(1, a.steps + 1):
        idx = torch.randint(0, Ntr, (a.batch,), device=dev); opt.zero_grad(set_to_none=True)
        loss = composable_kspace_loss(model(X[idx], T[idx], Ct[idx]), Yn[idx], dcf=torch.ones(a.batch, device=dev),
                                      use_dcf=False, dcf_power=0.0, use_focal=False, focal_warmup_progress=1.0)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % a.ckpt_every == 0: save(step)
        if step % 5000 == 0 or step == a.steps:
            print(f"  step {step} loss {float(loss):.4e} ({time.time()-t0:.0f}s)", flush=True)
    if a.steps % a.ckpt_every != 0: save(a.steps)                     # ensure final saved
    print(f"DONE w{a.hidden_width} s{a.seed}: {len(os.listdir(rundir))} checkpoints -> {rundir}", flush=True)

if __name__ == "__main__": main()
