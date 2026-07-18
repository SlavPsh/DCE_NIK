#!/usr/bin/env python
"""WIRE-NIK trainer for the grasp-pro precomputed twix data (CS-vs-NIK comparison).

DEFAULTS = the winning real-data recipe from the nik-autoresearch E1-E8 ablation:
  model    WIRE + Fourier features + RESIDUAL skips, depth 12, hidden 512, w0 62, s0 15
  FF       separate encoders: (kx,ky) k_freq 256/sigma 2.5 | t  t_freq 32/sigma 1.5
  norm     dcf_power 0.0 (DCF OFF), envelope_exponent 0.75
  train    40k steps, batch 65536, lr 1e-5, wd 3e-3, 70/30 spoke split, best-heldout
  -> slice 13: held-out 0.323, contrast swing 51%, nav-corr 0.885  (CS-70 swing 37.5%)

Key ablation facts baked into the defaults:
  - depth 12 WITH residual skips is the spatial lever (0.420 -> 0.337). Skips alone do nothing.
  - DCF OFF: it de-weights the low-|k| baseline that carries the contrast -> harms dynamics.
  - t_sigma 1.5 is the temporal bandwidth sweet spot; >3 collapses dynamics (nav-corr -> 0.70).
  - held-out MSE alone is misleading: the lowest-MSE models had the WORST dynamics.

per z-slice: build samples -> normalize -> train -> query cartesian (support-masked)
-> coil-combine+crop -> save nik image. assembles nik_recon.npy [bas,bas,nslices,nt].

run in torch29:
  micromamba run -n torch29 python train_grasp_nik.py --slices 13
  micromamba run -n torch29 python train_grasp_nik.py --slices all --save-dir results_nik
  # old small baseline for comparison:
  micromamba run -n torch29 python train_grasp_nik.py --model wire --hidden 64 --depth 6 --steps 8000
"""
import argparse, os, sys, time, logging
import numpy as np
import torch
import torch.nn.functional as F
torch.set_float32_matmul_precision("high")

sys.path.insert(0, '/scratch/rnga/vvpshenov/grasp_pro_py')   # nik_output_recon
import nik_adapter as A
from nik_model import (WIRE_KXY_COIL_T_REIM, WIRE_FF_KXY_COIL_T_REIM,
                       WIRE_FF_RES_KXY_COIL_T_REIM, WIRE_FF_SUBSPACE_KXY_COIL_T_REIM,
                       WIRE_FF_RES_RADIAL_KXY_COIL_T_REIM, warmstart_phi)
from kspace_normalization import compute_dcf_radial, compute_radius, KSpaceNormalizer
from nik_focal_loss import composable_kspace_loss
from nik_output_recon import recon_nik_cart


def compute_pca_phi(out_dir, slc, sh, rank, n_frames=100):
    """K=rank temporal PCA basis from the k-center navigator (same construction as grasp's
    front-end / build_phi), plus the frame times in the model's t convention (2*view_time-1).
    returns (frame_t[F] float32, Phi[F,rank] complex64) for warmstart_phi (Option 3 init)."""
    sl = np.load(os.path.join(out_dir, f'slice_{slc:02d}.npz'))
    krad = np.asarray(sl['kdata_radial'])                        # [nx, nspokes, nc]
    vt = np.asarray(sh['view_time']).ravel()
    nx, nsp, nc = krad.shape
    F = int(min(n_frames, nsp // 5))                             # keep >=5 spokes/frame
    nline = nsp // F; use = F * nline; c0 = nx // 2
    nav = np.abs(krad[c0 - 2:c0 + 3, :use, :]).reshape(5, nline, F, nc, order='F').mean(1)  # (5,F,nc)
    ds = nav.transpose(0, 2, 1).reshape(5 * nc, F, order='F')    # (5nc, F)
    w, PC = np.linalg.eigh(np.cov(ds, rowvar=False))
    Phi = PC[:, np.argsort(-w)][:, :rank].astype(np.complex64)   # (F, rank), real modes (imag 0)
    ft = np.array([vt[j * nline:(j + 1) * nline].mean() for j in range(F)], dtype=np.float32)
    return (2.0 * ft - 1.0).astype(np.float32), Phi


def build_model(args, ncc):
    """winning recipe = wire_ff_res (E1-E8 ablation). wire/wire_ff kept for comparison.
    wire_ff_subspace = experimental factorized low-rank model (rank = temporal-DoF knob)."""
    if args.model == 'wire':
        return WIRE_KXY_COIL_T_REIM(n_coils=ncc, coil_embed_dim=args.coil_embed_dim,
                                    hidden=args.hidden, depth=args.depth, w0=args.w0, s0=args.s0)
    if args.model == 'wire_ff_res_radial':
        return WIRE_FF_RES_RADIAL_KXY_COIL_T_REIM(
            n_coils=ncc, coil_embed_dim=args.coil_embed_dim, hidden=args.hidden, depth=args.depth,
            w0=args.w0, s0=args.s0, k_freq=args.k_freq, k_sigma=args.k_sigma, t_freq=args.t_freq,
            t_sigma=args.t_sigma, ff_seed=args.ff_seed, radial_alpha=args.radial_alpha)
    if args.model == 'wire_ff_subspace':
        return WIRE_FF_SUBSPACE_KXY_COIL_T_REIM(
            n_coils=ncc, coil_embed_dim=args.coil_embed_dim, rank=args.rank, hidden=args.hidden,
            depth=args.depth, w0=args.w0, s0=args.s0, k_freq=args.k_freq, k_sigma=args.k_sigma,
            t_freq=args.t_freq, t_sigma=args.t_sigma, ff_seed=args.ff_seed, residual=True,
            phi_hidden=args.phi_hidden, phi_depth=args.phi_depth, phi_w0=args.phi_w0)
    cls = WIRE_FF_RES_KXY_COIL_T_REIM if args.model == 'wire_ff_res' else WIRE_FF_KXY_COIL_T_REIM
    return cls(n_coils=ncc, coil_embed_dim=args.coil_embed_dim, hidden=args.hidden,
               depth=args.depth, w0=args.w0, s0=args.s0, k_freq=args.k_freq, k_sigma=args.k_sigma,
               t_freq=args.t_freq, t_sigma=args.t_sigma, ff_seed=args.ff_seed)


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
    normalizer.fit(x[train_idx], y_raw[train_idx], dcf=dcf[train_idx],
                   envelope_exponent=args.envelope_exponent)
    y = normalizer.normalize(x, y_raw)

    xtr, ttr, ctr, ytr, wtr = (x[train_idx], t[train_idx], c[train_idx],
                               y[train_idx], dcf[train_idx])
    if has_heldout:
        xhe, the, che, yhe = x[heldout_idx], t[heldout_idx], c[heldout_idx], y[heldout_idx]
    N_train = xtr.shape[0]

    torch.manual_seed(args.seed)
    model = build_model(args, ncc).to(device)
    if args.model == 'wire_ff_subspace' and args.warmstart:       # Option-3 init: Phi <- k-center PCA basis
        ft, phi = compute_pca_phi(out_dir, slc, sh, args.rank)
        err = warmstart_phi(model, ft, phi, steps=args.warmstart_steps, lr=1e-3, device=device.type)
        print(f'    warmstarted Phi from PCA basis (rank {args.rank}, {len(ft)} frames, fit MSE {err:.3e})', flush=True)
    if args.compile and device.type == 'cuda':
        try:
            model = torch.compile(model)
        except Exception as e:
            logging.warning(f'compile failed: {e}')
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=args.scheduler_patience, min_lr=args.scheduler_min_lr)

    best_heldout, best_state = float('inf'), None
    ckpt_path = os.path.join(args.save_dir, f'ckpt_slice_{slc:02d}.pt')
    start_step = 1
    if args.resume and os.path.exists(ckpt_path):                       # continue a timed-out run
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched'])
        best_heldout, best_state, start_step = ck['best_heldout'], ck['best_state'], ck['step'] + 1
        print(f'    resumed slice {slc:02d} from step {ck["step"]} (best heldout {best_heldout:.3e})', flush=True)

    def save_ckpt(step):                                                # atomic: tmp then rename
        tmp = ckpt_path + '.tmp'
        torch.save(dict(step=step, model=model.state_dict(), opt=opt.state_dict(),
                        sched=sched.state_dict(), best_heldout=best_heldout, best_state=best_state), tmp)
        os.replace(tmp, ckpt_path)

    model.train()
    for step in range(start_step, args.steps + 1):
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
                save_ckpt(step)                                        # survive wall-clock timeout
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
    img = recon_nik_cart(cart, b1, bas)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)                          # slice done -> drop its checkpoint
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default='/scratch/rnga/vvpshenov/grasp_pro_py/results_ref')
    ap.add_argument('--save-dir', default='/scratch/rnga/vvpshenov/grasp_pro_py/results_nik')
    ap.add_argument('--slices', default='13', help='"13" | "0:27" | "all" | "3,13,20"')
    # ---- WINNING RECIPE (nik-autoresearch E1-E8 ablation, slice 13) ----
    # WIRE + Fourier features + residual skips, depth 12, hidden 512.
    # held-out 0.323, contrast swing 51%, nav-corr 0.885 (CS-70 swing = 37.5%).
    # model
    ap.add_argument('--model', default='wire_ff_res',
                    choices=['wire', 'wire_ff', 'wire_ff_res', 'wire_ff_subspace', 'wire_ff_res_radial'])
    ap.add_argument('--radial-alpha', type=float, default=1.0,
                    help='(wire_ff_res_radial) |k|-dependent FF warp strength; 0 = no warp')
    # factorized low-rank model (--model wire_ff_subspace). rank = temporal-DoF knob (sweep > 5).
    ap.add_argument('--rank', type=int, default=12, help='subspace rank R (temporal DoF)')
    ap.add_argument('--phi-hidden', type=int, default=64, help='temporal-basis net width')
    ap.add_argument('--phi-depth', type=int, default=3, help='temporal-basis net depth')
    ap.add_argument('--phi-w0', type=float, default=30.0, help='temporal-basis SIREN w0')
    ap.add_argument('--warmstart', dest='warmstart', action='store_true', default=True,
                    help='(subspace) init Phi from k-center PCA basis -- stabilizes the bilinear fit')
    ap.add_argument('--no-warmstart', dest='warmstart', action='store_false')
    ap.add_argument('--warmstart-steps', type=int, default=800)
    ap.add_argument('--hidden', type=int, default=512)
    ap.add_argument('--depth', type=int, default=12)     # d12 = sweet spot (d16 gains ~0, loses swing)
    ap.add_argument('--w0', type=float, default=62.0)
    ap.add_argument('--s0', type=float, default=15.0)
    ap.add_argument('--coil-embed-dim', type=int, default=8)
    # fourier features: SEPARATE encoders for (kx,ky) and t -- t needs its own bandwidth
    ap.add_argument('--k-freq', type=int, default=256)
    ap.add_argument('--k-sigma', type=float, default=2.5)
    ap.add_argument('--t-freq', type=int, default=32)    # E8: 32 > 8 at the right bandwidth
    ap.add_argument('--t-sigma', type=float, default=1.5)  # E8: bandwidth sweet spot. >3 kills dynamics
    ap.add_argument('--ff-seed', type=int, default=0)
    # training
    ap.add_argument('--steps', type=int, default=40000)
    ap.add_argument('--batch-size', type=int, default=65536)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--weight-decay', type=float, default=0.003)
    ap.add_argument('--grad-clip', type=float, default=1.0)
    ap.add_argument('--warmup-steps', type=int, default=2000)
    ap.add_argument('--eval-every', type=int, default=1000)
    ap.add_argument('--console-every', type=int, default=1000)
    ap.add_argument('--scheduler-patience', type=int, default=30)
    ap.add_argument('--scheduler-min-lr', type=float, default=1e-7)
    ap.add_argument('--subsample-frac', type=float, default=0.7,
                    help='fraction of spokes for train; rest are heldout for model selection')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--resume', action='store_true',
                    help='resume each slice from save_dir/ckpt_slice_XX.pt if present (survives timeouts)')
    # loss / norm
    ap.add_argument('--use-dcf', type=int, default=1)
    ap.add_argument('--dcf-method', default='simple_ramp')
    ap.add_argument('--dcf-power', type=float, default=0.0,
                    help='0.0 = DCF OFF. DCF de-weights the low-|k| baseline that carries '
                         'contrast -> harms dynamics. (Overturns the earlier 0.5 finding.)')
    ap.add_argument('--envelope-exponent', type=float, default=0.75,
                    help='0.75 = soft whitening. 1.0 (full) amplifies noisy high-|k| periphery.')
    ap.add_argument('--use-focal', type=int, default=0)
    ap.add_argument('--support-radius', type=float, default=1.0)
    ap.add_argument('--no-compile', dest='compile', action='store_false')
    args = ap.parse_args()
    args.use_dcf = bool(args.use_dcf); args.use_focal = bool(args.use_focal)

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}', flush=True)
    rank_str = f' rank{args.rank}' if args.model == 'wire_ff_subspace' else ''
    if args.model == 'wire_ff_res_radial': rank_str = f' radial_alpha={args.radial_alpha:g}'
    print(f'model={args.model}{rank_str} h{args.hidden} d{args.depth} w0{args.w0:g} | '
          f'FF k{args.k_freq}/{args.k_sigma:g} t{args.t_freq}/{args.t_sigma:g} | '
          f'dcf_pow {args.dcf_power:g} env {args.envelope_exponent:g} | '
          f'{args.steps} steps, batch {args.batch_size}, split {args.subsample_frac:g}', flush=True)
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
