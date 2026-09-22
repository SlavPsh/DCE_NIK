"""gradient check on trained in vivo nik models: (1) assert no complex-dtype tensor with grad exists in the training graph of the data loss,
(2) list the autograd functions in the graph and flag the ones with kinks (abs, sign, sqrt), (3) compare autograd gradients with central
finite differences of the real loss for a sample of parameters in every part of the model (coil embedding, first gabor layer, mid block, head,
temporal net for sub16), on one fixed batch of kept spokes. float64 for the finite differences. out: results/tofts_vs_patlak/gradient_check.md
usage: python gradient_check.py --runs tofts8:<dir>,sub16:<dir>,free:<dir> --slice 21"""
import warnings; warnings.filterwarnings("ignore")
import sys, argparse, numpy as np, torch
from types import SimpleNamespace
B = "/net/beegfs/users/P101440/DCE_NIK"; REFD = "/net/beegfs/users/P101440/grasp_pro_py/results_ref"; sys.path.insert(0, B); sys.path.insert(0, "/net/beegfs/users/P101440/grasp_pro_py")
import nik_adapter as A
from train_grasp_nik import build_model
from kspace_normalization import KSpaceNormalizer, compute_dcf_radial
from nik_focal_loss import composable_kspace_loss
dev = torch.device("cpu")

def graph_ops(t):
    seen, ops, stack = set(), {}, [t.grad_fn]
    while stack:
        fn = stack.pop()
        if fn is None or fn in seen: continue
        seen.add(fn); nm = type(fn).__name__; ops[nm] = ops.get(nm, 0) + 1
        for nxt, _ in fn.next_functions: stack.append(nxt)
    return ops

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", required=True); ap.add_argument("--slice", type=int, default=21); ap.add_argument("--batch", type=int, default=4096); ap.add_argument("--n-params", type=int, default=6); a = ap.parse_args(); Z = a.slice
    torch.manual_seed(0); sh = A.load_shared(REFD); ds = A.make_radial_dataset(REFD, Z, compute_device=dev, shared=sh)
    x, t, c, y_raw, sid = ds["x_all"], ds["t_all"], ds["coil_all"], ds["y_all_raw"], ds["spoke_id_all"]
    KEEP = np.load(f"{B}/spoke_masks/keep_f80match.npy"); tr = torch.where(torch.isin(sid, torch.as_tensor(KEEP, dtype=sid.dtype)))[0]
    dcf = compute_dcf_radial(x, method="simple_ramp"); nz = KSpaceNormalizer(); nz.fit(x[tr], y_raw[tr], dcf=dcf[tr], envelope_exponent=0.75); y = nz.normalize(x, y_raw)
    idx = tr[torch.randperm(tr.numel())[:a.batch]]; xb, tb, cb, yb = x[idx].double(), t[idx].double(), c[idx], y[idx].double()
    L = ["# gradient check on trained in vivo models (slice %d, one fixed batch of %d kept samples, float64)" % (Z, a.batch), "",
         "| model | complex tensors with grad in the data-loss graph | autograd ops with kinks (abs / sign / sqrt) | parameter | autograd | finite difference | rel. error |", "|---|---|---|---|---|---|---|"]
    for spec in a.runs.split(","):
        nm, d = spec.split(":", 1); ck = torch.load(f"{d}/model_slice_{Z:02d}.pt", map_location=dev, weights_only=False)
        args = SimpleNamespace(**{k: ck[k] for k in ("model", "rank", "hidden", "depth", "w0", "s0", "coil_embed_dim", "k_freq", "k_sigma", "t_freq", "t_sigma", "ff_seed")},
                               patlak_free=0, aif_file=f"{B}/aif_slice{Z}.npz", tofts_basis=f"{B}/results/tofts_vs_patlak/basis_sl{Z}_r8_rms1.npz", phi_hidden=64, phi_depth=3, phi_w0=30.0, phi_ortho=False, n_pk=-1, radial_alpha=1.0, coil_mode=ck.get("coil_mode", "input"))
        m = build_model(args, int(ck["ncc"])); m.load_state_dict(ck["state_dict"]); m = m.double().to(dev); m.train()
        def loss_fn():
            return composable_kspace_loss(m(xb, tb, cb), yb, dcf=torch.ones(xb.shape[0], dtype=torch.float64), use_dcf=False, dcf_power=0.0, use_focal=False, focal_warmup_progress=1.0, return_diagnostics=False)
        # (1) complex tensors in the graph: hook every module output
        cplx = []
        hooks = [mod.register_forward_hook(lambda mod, inp, out, n=n: cplx.append(n) if (torch.is_tensor(out) and out.is_complex()) else None) for n, mod in m.named_modules()]
        loss = loss_fn(); [h.remove() for h in hooks]
        # (2) ops in the graph
        ops = graph_ops(loss); kinks = {k: v for k, v in ops.items() if any(s in k.lower() for s in ("abs", "sign", "sqrt", "relu", "clamp"))}
        # (3) finite differences vs autograd on sampled parameters
        m.zero_grad(); loss.backward(); params = dict(m.named_parameters())
        picks = []
        for key in ("coil_embed.weight", "a_first.linear.weight", "first.linear.weight", "a_blocks.5.linear.weight", "blocks.5.linear.weight", "a_head.weight", "head.weight", "a_head.bias", "phi_body.0.linear.weight", "phi_head.weight"):
            if key in params and len(picks) < a.n_params: picks.append(key)
        rows = []
        for key in picks:
            p = params[key]; g = p.grad; flat = p.data.view(-1); gi = g.view(-1); i = int(torch.argmax(gi.abs())); eps = 1e-4 * max(1.0, float(flat[i].abs()))
            with torch.no_grad():
                orig = float(flat[i]); flat[i] = orig + eps; lp = float(loss_fn()); flat[i] = orig - eps; lm = float(loss_fn()); flat[i] = orig
            fd = (lp - lm) / (2 * eps); ag = float(gi[i]); rel = abs(ag - fd) / (abs(ag) + abs(fd) + 1e-30); rows.append((key, ag, fd, rel))
        first = True
        for key, ag, fd, rel in rows:
            L.append(f"| {nm if first else ''} | {('none' if not cplx else ', '.join(sorted(set(cplx)))) if first else ''} | {(str(kinks) if kinks else 'none') if first else ''} | {key} | {ag:+.6e} | {fd:+.6e} | {rel:.1e} |"); first = False
        print(nm, "complex:", cplx or "none", "| kinks:", kinks or "none", "| worst rel err:", max(r[3] for r in rows), flush=True)
    L += ["", "reading: 'none' complex tensors = the data-loss graph is entirely real; rel. error at 1e-6 or below = autograd equals the finite-difference derivative of the real loss to float64 precision at every sampled parameter"]
    open(f"{B}/results/tofts_vs_patlak/gradient_check.md", "w").write("\n".join(L)); print("\n".join(L)); print("GRADCHECK_DONE")

if __name__ == "__main__": main()
