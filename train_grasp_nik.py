#!/usr/bin/env python
"""WIRE-NIK trainer for the grasp-pro precomputed twix data (CS-vs-NIK comparison).

parallel to train_multicoil_cart.py: SAME model (WIRE_KXY_COIL_T_REIM), loss
(composable_kspace_loss), DCF + KSpaceNormalizer, Adam + plateau scheduler, and
spoke-based heldout split. ONLY differences: data comes from nik_adapter (precompute
results_ref/) instead of the XCAT h5, and the image is formed with the grasp coil maps
b1 via nik_output_recon.recon_nik_cart, so it is apples-to-apples with cs_recon.npy.

per z-slice: build samples -> normalize -> train -> query cartesian (support-masked)
-> coil-combine+crop -> save nik image. assembles nik_recon.npy [bas,bas,nslices,nt].

run in torch29:
  micromamba run -n torch29 python train_grasp_nik.py --slices 12 --steps 8000
  micromamba run -n torch29 python train_grasp_nik.py --slices all --steps 8000 --save-dir results_nik
"""
import argparse, os, sys, time, logging
import numpy as np
import torch
import torch.nn.functional as F
torch.set_float32_matmul_precision("high")

sys.path.insert(0, '/scratch/rnga/vvpshenov/grasp_pro_py')   # nik_output_recon
import nik_adapter as A
from nik_model import WIRE_KXY_COIL_T_REIM
from kspace_normalization import compute_dcf_radial, compute_radius, KSpaceNormalizer
from nik_focal_loss import composable_kspace_loss
from nik_output_recon import recon_nik_cart


def parse_slices(s, nz):
    if s in (None, 'all'):
        return list(range(nz))
    if ':' in s:
        a, b = s.split(':'); return list(range(int(a), int(b)))
    return [int(x) for x in s.split(',')]


def train_one_slice(out_dir, slc, sh, args, device):
    """returns nik image [bas,bas,nt] for this slice."""
    ds = A.make_radial_dataset(out_dir, slc, compute_device=device, shared=sh)
    x, t, c, y_raw = ds['x_all'], ds['t_all'], ds['coil_all'], ds['y_all_raw']
    spoke_id, b1 = ds['spoke_id_all'], ds['b1']
    ncc, bas = ds['meta']['ncc'], ds['meta']['bas']

    # spoke-based train/heldout split (matches train_multicoil_cart)
    uniq = torch.unique(spoke_id)
    n_train = max(1, int(uniq.numel() * args.subsample_frac))
    g = torch.Generator(device=device).manual_seed(args.seed)
    perm = uniq[torch.randperm(uniq.numel(), generator=g, device=device)]
    train_mask = torch.isin(spoke_id, perm[:n_train])
    train_idx = torch.where(train_mask)[0]
    heldout_idx = torch.where(~train_mask)[0]
    has_heldout = heldout_idx.numel() > 0

    # DCF (geometry only) + normalizer fit on train spokes
    dcf = compute_dcf_radial(x, method=args.dcf_method) if args.use_dcf else torch.ones(
        x.shape[0], device=device)
    normalizer = KSpaceNormalizer()
    normalizer.fit(x[train_idx], y_raw[train_idx], dcf=dcf[train_idx])
    y = normalizer.normalize(x, y_raw)

    xtr, ttr, ctr, ytr, wtr = (x[train_idx], t[train_idx], c[train_idx],
                               y[train_idx], dcf[train_idx])
    if has_heldout:
        xhe, the, che, yhe = x[heldout_idx], t[heldout_idx], c[heldout_idx], y[heldout_idx]
    N_train = xtr.shape[0]

    model = WIRE_KXY_COIL_T_REIM(n_coils=ncc, coil_embed_dim=args.coil_embed_dim,
                                 hidden=args.hidden, depth=args.depth,
                                 w0=args.w0, s0=args.s0).to(device)
    if args.compile and device.type == 'cuda':
        try:
            model = torch.compile(model)
        except Exception as e:
            logging.warning(f'compile failed: {e}')
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=args.scheduler_patience, min_lr=1e-6)

    best_heldout, best_state = float('inf'), None
    model.train()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, N_train, (args.batch_size,), device=device)
        opt.zero_grad(set_to_none=True)
        y_pred = model(xtr[idx], ttr[idx], ctr[idx])
        loss = composable_kspace_loss(
            y_pred, ytr[idx], dcf=wtr[idx], use_dcf=args.use_dcf, dcf_power=args.dcf_power,
            use_focal=args.use_focal, focal_warmup_progress=min(1.0, step / 1000.0),
            return_diagnostics=False)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if step % args.eval_every == 0 or step == args.steps:
            if has_heldout:
                model.eval()
                with torch.no_grad():
                    hp = torch.cat([model(xhe[i:i+262144], the[i:i+262144], che[i:i+262144])
                                    for i in range(0, xhe.shape[0], 262144)], 0)
                    hl = float(F.mse_loss(hp, yhe).item())
                model.train()
                sched.step(hl)
                if step >= args.warmup_steps and hl < best_heldout:
                    best_heldout = hl
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if step % args.console_every == 0 or step == args.steps:
                    print(f'    step {step:6d}  train {float(loss):.3e}  heldout {hl:.3e}', flush=True)
            elif step % args.console_every == 0 or step == args.steps:
                print(f'    step {step:6d}  train {float(loss):.3e}', flush=True)

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f'    restored best (heldout {best_heldout:.3e})', flush=True)
    model.eval()

    cart = A.reconstruct_cartesian(model, normalizer, out_dir, device=device.type,
                                   shared=sh, support_radius=args.support_radius, verbose=False)
    return recon_nik_cart(cart, b1, bas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default='/scratch/rnga/vvpshenov/grasp_pro_py/results_ref')
    ap.add_argument('--save-dir', default='/scratch/rnga/vvpshenov/grasp_pro_py/results_nik')
    ap.add_argument('--slices', default='13', help='"13" | "0:27" | "all" | "3,13,20"')
    # model (config/multicoil_cart_eval.toml defaults)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--depth', type=int, default=6)   # matches sim runs (see note: needs enough steps/warmup)
    ap.add_argument('--w0', type=float, default=32.0)
    ap.add_argument('--s0', type=float, default=18.0)
    ap.add_argument('--coil-embed-dim', type=int, default=8)
    # training
    ap.add_argument('--steps', type=int, default=8000)
    ap.add_argument('--batch-size', type=int, default=4096)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight-decay', type=float, default=0.003)
    ap.add_argument('--grad-clip', type=float, default=1.0)
    ap.add_argument('--warmup-steps', type=int, default=1600)
    ap.add_argument('--eval-every', type=int, default=100)
    ap.add_argument('--console-every', type=int, default=1000)
    ap.add_argument('--scheduler-patience', type=int, default=50)
    ap.add_argument('--subsample-frac', type=float, default=0.9,
                    help='fraction of spokes for train; rest are heldout for model selection')
    ap.add_argument('--seed', type=int, default=0)
    # loss / norm
    ap.add_argument('--use-dcf', type=int, default=1)
    ap.add_argument('--dcf-method', default='simple_ramp')
    ap.add_argument('--dcf-power', type=float, default=0.5,
                    help='0=config default; 0.5-1.0 fixes high-|k| under-fit (project finding)')
    ap.add_argument('--use-focal', type=int, default=0)
    ap.add_argument('--support-radius', type=float, default=0.5)
    ap.add_argument('--no-compile', dest='compile', action='store_false')
    args = ap.parse_args()
    args.use_dcf = bool(args.use_dcf); args.use_focal = bool(args.use_focal)

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}', flush=True)
    os.makedirs(args.save_dir, exist_ok=True)
    sh = A.load_shared(args.out_dir)
    nz, nt, bas = int(sh['nzz']), int(sh['nt']), int(sh['bas'])
    sl = parse_slices(args.slices, nz)

    t0 = time.time()
    vol = np.full((bas, bas, nz, nt), np.nan, np.float32)
    for slc in sl:
        ts = time.time()
        img = train_one_slice(args.out_dir, slc, sh, args, device)
        vol[:, :, slc, :] = img
        np.save(os.path.join(args.save_dir, f'nik_slice_{slc:02d}.npy'), img)
        line = f'slice {slc:02d}  nik {img.shape}  ({time.time()-ts:.0f}s)'
        try:                                                   # scale-matched check vs per-slice CS
            csimg = A.load_slice(args.out_dir, slc)['cs_img']
            a, b = img.ravel(), csimg.ravel()
            s = float((a @ b) / (a @ a + 1e-12))               # LS scale a->b
            nrmse = float(np.linalg.norm(s*a - b) / (np.linalg.norm(b) + 1e-12))
            corr = float(np.corrcoef(a, b)[0, 1])
            line += f'  vs CS: corr {corr:.3f}  nrmse {nrmse:.3f}'
        except Exception:
            pass
        print(line, flush=True)

    if len(sl) == nz:
        np.save(os.path.join(args.save_dir, 'nik_recon.npy'), vol)
        print(f'saved nik_recon {vol.shape}', flush=True)
    print(f'done {time.time()-t0:.0f}s -> {args.save_dir}', flush=True)


if __name__ == '__main__':
    main()
