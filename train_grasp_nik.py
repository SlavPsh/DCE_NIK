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
import nik_wandb as W

sys.path.insert(0, '/net/beegfs/users/P101440/grasp_pro_py')   # nik_output_recon
import nik_adapter as A
from nik_model import (WIRE_KXY_COIL_T_REIM, WIRE_FF_KXY_COIL_T_REIM,
                       WIRE_FF_RES_KXY_COIL_T_REIM, WIRE_FF_SUBSPACE_KXY_COIL_T_REIM,
                       WIRE_FF_RES_RADIAL_KXY_COIL_T_REIM, WIRE_FF_PK_KXY_COIL_T_REIM,
                       WIRE_FF_PATLAK_KXY_COIL_T_REIM, WIRE_FF_TOFTS_KXY_COIL_T_REIM, warmstart_phi)
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


def phi_tv_penalty(model, n=256, delta=0.1, device='cuda'):
    """edge-preserving (Huber) total-variation on the temporal atoms Phi_r(t), sampled on
    a dense t-grid in [-1,1]. Huber on the per-step |dPhi| (normalized by each atom's RMS
    scale so delta is scale-free): small wiggles (noise) ~quadratic, large jumps (bolus
    rise) ~linear -> suppresses noise but KEEPS the sharp edge. This is temporal TV (what
    CS uses), NOT an L2 curvature penalty (which would clip the peak). Regularizing the
    shared atoms regularizes every voxel curve (a linear combo of them)."""
    tg = torch.linspace(-1.0, 1.0, int(n), device=device)
    P = model.basis(tg)                                       # (n, R, 2)
    d = P[1:] - P[:-1]                                        # (n-1, R, 2) first difference
    mag = torch.sqrt((d ** 2).sum(-1) + 1e-12)               # (n-1, R) complex-isotropic |dPhi|
    rms = torch.sqrt((P ** 2).sum(-1).mean(0) + 1e-12)       # (R,) per-atom scale
    m = mag / rms.unsqueeze(0)                                # scale-free per-step variation
    huber = torch.where(m <= delta, 0.5 * m * m / delta, m - 0.5 * delta)
    return huber.mean()



def kspace_tv21_penalty(model, x_pool, c_pool, n_k=1024, n_t=64, delta=0.1, device='cuda'):
    """edge-preserving temporal prior applied DIRECTLY to the k-space output, for models with no
    temporal-atom basis (wire_ff_res). phi_tv_penalty only works on subspace models.

    l2,1 mixed norm: L2 across the sampled k-locations/coils at each time step, Huber (~L1) across
    time. the inner L2 is Parseval-equivalent to an L2 across image space, so this is the k-space
    form of "few time points where the image changes" -- piecewise-smooth in t, sharp bolus allowed.
    a plain per-sample L1 in k-space would NOT be equivalent (it would promote few k-locations
    changing, which is not the wanted prior).

    k-locations are drawn from the MEASURED pool, so the sampling density (radial -> denser at low
    |k|) weights the penalty toward the low frequencies where the bolus contrast lives.
    """
    i = torch.randint(0, x_pool.shape[0], (n_k,), device=device)
    xk, cc = x_pool[i], c_pool[i]
    tg = torch.linspace(-1.0, 1.0, int(n_t), device=device)
    X = xk.repeat_interleave(n_t, 0)
    C = cc.repeat_interleave(n_t, 0)
    T = tg.repeat(n_k)
    y = model(X, T, C).view(n_k, n_t, -1)                     # (n_k, n_t, 2) Re/Im
    d = y[:, 1:] - y[:, :-1]                                  # first difference along t
    g = torch.sqrt((d ** 2).sum(-1).sum(0) + 1e-12)           # (n_t-1,) L2 over k and Re/Im
    rms = torch.sqrt((y ** 2).sum(-1).mean() + 1e-12) * (n_k ** 0.5)
    m = g / (rms + 1e-12)                                     # scale-free per-step variation
    huber = torch.where(m <= delta, 0.5 * m * m / delta, m - 0.5 * delta)
    return huber.mean()


def build_model(args, ncc):
    """winning recipe = wire_ff_res (E1-E8 ablation). wire/wire_ff kept for comparison.
    wire_ff_subspace = factorized low-rank (rank = temporal-DoF knob); wire_ff_pk = same
    but gamma-variate (physical bolus) temporal atoms + baseline + soft free residual."""
    if args.model == 'wire':
        return WIRE_KXY_COIL_T_REIM(n_coils=ncc, coil_embed_dim=args.coil_embed_dim,
                                    hidden=args.hidden, depth=args.depth, w0=args.w0, s0=args.s0)
    if args.model == 'wire_ff_pk':
        return WIRE_FF_PK_KXY_COIL_T_REIM(
            n_coils=ncc, coil_embed_dim=args.coil_embed_dim, rank=args.rank, hidden=args.hidden,
            depth=args.depth, w0=args.w0, s0=args.s0, k_freq=args.k_freq, k_sigma=args.k_sigma,
            t_freq=args.t_freq, t_sigma=args.t_sigma, ff_seed=args.ff_seed, residual=True,
            phi_hidden=args.phi_hidden, phi_depth=args.phi_depth, phi_w0=args.phi_w0,
            n_pk=(None if args.n_pk < 0 else args.n_pk))
    if args.model == 'wire_ff_patlak':
        az = np.load(args.aif_file)
        TA = float(az['tC'][-1]); tgrid = 2.0 * (np.asarray(az['tC']) / TA) - 1.0   # frame secs -> model t [-1,1]
        aif = np.asarray(az['aif_frame'], dtype=np.float64); aif = aif / (aif.max() + 1e-9)  # peak-norm
        iaif = np.cumsum(0.5 * (aif[1:] + aif[:-1]) * np.diff(np.asarray(az['tC'])))          # trapz integral
        iaif = np.concatenate([[0.0], iaif]); iaif = iaif / (iaif.max() + 1e-9)               # max-norm
        return WIRE_FF_PATLAK_KXY_COIL_T_REIM(
            n_coils=ncc, aif_tgrid=tgrid, aif_vals=aif, iaif_vals=iaif, n_free=args.patlak_free,
            coil_embed_dim=args.coil_embed_dim, hidden=args.hidden, depth=args.depth, w0=args.w0,
            s0=args.s0, k_freq=args.k_freq, k_sigma=args.k_sigma, t_freq=args.t_freq,
            t_sigma=args.t_sigma, ff_seed=args.ff_seed, residual=True,
            phi_hidden=args.phi_hidden, phi_depth=args.phi_depth, phi_w0=args.phi_w0)
    if args.model == 'wire_ff_tofts':
        bz = np.load(args.tofts_basis, allow_pickle=True)      # built offline by nik_tofts_basis.py (fixed, no learned rate)
        return WIRE_FF_TOFTS_KXY_COIL_T_REIM(
            n_coils=ncc, atom_tgrid=bz['tgrid_model'], atoms=bz['atoms'], R_patlak=bz['R_patlak'],
            coil_embed_dim=args.coil_embed_dim, hidden=args.hidden, depth=args.depth, w0=args.w0,
            s0=args.s0, k_freq=args.k_freq, k_sigma=args.k_sigma, t_freq=args.t_freq,
            t_sigma=args.t_sigma, ff_seed=args.ff_seed, residual=True)
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
            phi_hidden=args.phi_hidden, phi_depth=args.phi_depth, phi_w0=args.phi_w0,
            ortho=args.phi_ortho)
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
    if args.spoke_keep_file:                                   # spoke-reduction: identical acquired set as CS
        kept = torch.as_tensor(np.load(args.spoke_keep_file), device=device, dtype=uniq.dtype)
        train_mask = torch.isin(spoke_id, kept)               # train on the acquired spokes ONLY
        train_idx = torch.where(train_mask)[0]
        if getattr(args, 'spoke_heldout_file', None):
            # explicit VAL spoke set (disjoint from kept); everything else stays untouched (TEST)
            hv = torch.as_tensor(np.load(args.spoke_heldout_file), device=device, dtype=uniq.dtype)
            assert not bool(torch.isin(hv, kept).any()), 'heldout spokes overlap kept spokes'
            heldout_idx = torch.where(torch.isin(spoke_id, hv))[0]
            has_heldout = heldout_idx.numel() > 0
            print(f'    explicit heldout: {int(hv.numel())} VAL spokes', flush=True)
        elif getattr(args, 'keep_heldout', False):
            # train on kept spokes ONLY, but use the NON-kept spokes for early stopping. they are
            # never fit, only scored, so the training input still matches CS/GRASP exactly.
            heldout_idx = torch.where(~train_mask)[0]
            has_heldout = heldout_idx.numel() > 0
        else:
            heldout_idx = torch.empty(0, dtype=torch.long, device=device)   # NO heldout -> non-acquired spokes never seen
            has_heldout = False
        print(f'    spoke-reduction: {int(torch.isin(uniq, kept).sum())} acquired spokes, '
              f'{train_idx.numel()} samples, heldout={"yes" if has_heldout else "NO"} (same input as CS)', flush=True)
    else:
        n_train = max(1, int(uniq.numel() * args.subsample_frac))
        g = torch.Generator(device=device).manual_seed(args.seed)
        perm = uniq[torch.randperm(uniq.numel(), generator=g, device=device)]
        train_mask = torch.isin(spoke_id, perm[:n_train])
        train_idx = torch.where(train_mask)[0]
        heldout_idx = torch.where(~train_mask)[0]
        has_heldout = heldout_idx.numel() > 0

    # ARM B (Task 1 ablation): bin acquired spokes by time into groups of --bin-spokes and
    # replace each spoke's timestamp with its bin-centre (prior-art frame binning). t stays
    # continuous (Arm A) when bin_spokes<=0. render is unaffected (always continuous frame_t).
    if getattr(args, 'bin_spokes', 0) and args.bin_spokes > 0:
        usp, inv = torch.unique(spoke_id, return_inverse=True)
        sp_time = torch.zeros(usp.numel(), device=device, dtype=t.dtype); sp_time[inv] = t
        acq_u = torch.isin(usp, kept) if args.spoke_keep_file else torch.ones_like(usp, dtype=torch.bool)
        at = sp_time[acq_u]; order = torch.argsort(at)
        binidx = torch.empty_like(order); binidx[order] = torch.arange(at.numel(), device=device) // int(args.bin_spokes)
        nb = int(binidx.max().item()) + 1
        bsum = torch.zeros(nb, device=device, dtype=t.dtype).scatter_add_(0, binidx, at)
        bcnt = torch.zeros(nb, device=device, dtype=t.dtype).scatter_add_(0, binidx, torch.ones_like(at))
        new_sp = sp_time.clone(); new_sp[acq_u] = (bsum / bcnt)[binidx]
        t = new_sp[inv]
        print(f'    ARM B binning: {int(at.numel())} acquired spokes -> {nb} bins of {args.bin_spokes} spokes/frame '
              f'(t quantized to bin-centres)', flush=True)

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
    t_slice = time.time()
    yhe_energy = float(yhe.pow(2).mean()) if has_heldout else 1.0
    grp = os.path.basename(os.path.normpath(args.save_dir))
    run = W.Run(f'gnik_{grp}_s{slc:02d}', config=dict(vars(args), slice=slc), group=grp, tags=[args.model],
                local_json=os.path.join(args.save_dir, 'wandb_runs', f'slice_{slc:02d}.json'), enabled=not args.no_wandb)

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
    pen_model = getattr(model, '_orig_mod', model)                # uncompiled handle for basis()/TV
    tv_on = args.phi_tv_weight > 0 and hasattr(pen_model, 'basis')   # atoms exist only for subspace models
    ktv_on = args.ktv21_weight > 0                                   # k-space output: works for ANY model
    if args.phi_tv_weight > 0 and not tv_on:
        print('    [phi-tv] requested but model has no basis(); ignored', flush=True)

    # SPIRiT k-space coil-consistency prior. query a cartesian patch (all coils), denormalize
    # to raw k-space (G is calibrated raw), apply G, penalize ||(G-I)x||^2. no render.
    spirit_on = args.spirit_weight > 0
    if spirit_on:
        from spirit import spirit_penalty as _spirit_pen
        _sp = np.load(args.spirit_kernel)
        _G = _sp['G']
        Gr = torch.from_numpy(_G.real.astype(np.float32)).to(device)
        Gi = torch.from_numpy(_G.imag.astype(np.float32)).to(device)
        nxg = int(sh['nx'])
        _grid = A.cartesian_grid(nxg).reshape(nxg, nxg, 2).astype(np.float32)   # [-1,1], model coords
        _ft = (2.0 * sh['frame_time'] - 1.0).astype(np.float32)
        P = int(args.spirit_patch)
        print(f'    [spirit] kernel {_G.shape} patch {P}x{P} weight {args.spirit_weight:g}', flush=True)

        def spirit_term():
            gy = int(torch.randint(0, nxg - P, (1,)).item()); gx = int(torch.randint(0, nxg - P, (1,)).item())
            coords = torch.from_numpy(_grid[gy:gy + P, gx:gx + P].reshape(P * P, 2)).to(device)
            f = int(torch.randint(0, len(_ft), (1,)).item())
            tf = torch.full((P * P,), float(_ft[f]), device=device)
            outs = []
            for c in range(ncc):
                cc = torch.full((P * P,), c, device=device, dtype=torch.long)
                raw = normalizer.denormalize(coords, pen_model(coords, tf, cc))   # [P*P,2] raw
                outs.append(raw.view(P, P, 2))
            x = torch.stack(outs, 0).permute(0, 3, 1, 2)                          # [C,2,P,P]
            return _spirit_pen(x, Gr, Gi)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=args.scheduler_patience, min_lr=args.scheduler_min_lr)

    best_heldout, best_state, best_step = float('inf'), None, 0
    ckpt_path = os.path.join(args.save_dir, f'ckpt_slice_{slc:02d}.pt')
    start_step = 1
    if args.resume and os.path.exists(ckpt_path):                       # continue a timed-out run
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched'])
        best_heldout, best_state, start_step = ck['best_heldout'], ck['best_state'], ck['step'] + 1
        best_step = ck.get('best_step', 0)
        print(f'    resumed slice {slc:02d} from step {ck["step"]} (best heldout {best_heldout:.3e})', flush=True)

    def save_ckpt(step):                                                # atomic: tmp then rename
        tmp = ckpt_path + '.tmp'
        torch.save(dict(step=step, model=model.state_dict(), opt=opt.state_dict(),
                        sched=sched.state_dict(), best_heldout=best_heldout, best_state=best_state, best_step=best_step), tmp)
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
        if tv_on:                                                # P0 temporal-TV (edge-preserving) prior
            loss = loss + args.phi_tv_weight * phi_tv_penalty(
                pen_model, n=args.phi_tv_grid, delta=args.phi_tv_delta, device=device.type)
        if ktv_on:                                               # l2,1 temporal prior on k-space output
            loss = loss + args.ktv21_weight * kspace_tv21_penalty(
                model, xtr, ctr, n_k=args.ktv21_nk, n_t=args.ktv21_nt,
                delta=args.ktv21_delta, device=device.type)
        if spirit_on:                                            # A0 k-space coil-consistency prior
            loss = loss + args.spirit_weight * spirit_term()
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        if args.phi_ortho and (step <= 20 or (step <= 300 and step % 20 == 0)   # QR stability, ALL 40k
                               or step % 1000 == 0 or step == args.steps):
            print(f'    [ortho] step {step:6d}  loss {float(loss):.3e}  gradnorm {float(gnorm):.2e}  '
                  f'rawGramCond {getattr(pen_model, "last_gram_cond", 0.0):.2e}', flush=True)

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
                    best_step = step
                save_ckpt(step)                                        # survive wall-clock timeout
                run.log(dict(loss=float(loss), heldout_mse=hl, val_knmse=hl / yhe_energy, best_heldout_mse=best_heldout,
                             best_step=best_step, lr=opt.param_groups[0]['lr'], wall_s=time.time() - t_slice,
                             peak_gpu_mb=W.peak_gpu_mb()), step=step)
                if step % args.console_every == 0 or step == args.steps:
                    print(f'    step {step:6d}  train {float(loss):.3e}  heldout {hl:.3e}', flush=True)
            elif step % args.console_every == 0 or step == args.steps:
                print(f'    step {step:6d}  train {float(loss):.3e}', flush=True)
                run.log(dict(loss=float(loss), wall_s=time.time() - t_slice, peak_gpu_mb=W.peak_gpu_mb()), step=step)

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f'    restored best (heldout {best_heldout:.3e})', flush=True)
    model.eval()

    cart = A.reconstruct_cartesian(model, normalizer, out_dir, device=device.type,
                                   shared=sh, support_radius=args.support_radius, verbose=False)
    img, img_cplx = recon_nik_cart(cart, b1, bas, return_complex=True)
    # DEFAULT persistence: model weights + complex recon, so diagnostics are never blocked
    sd = getattr(model, '_orig_mod', model).state_dict()
    torch.save(dict(state_dict={k: v.cpu() for k, v in sd.items()},
                    model=args.model, rank=args.rank, hidden=args.hidden, depth=args.depth,
                    w0=args.w0, s0=args.s0, coil_embed_dim=args.coil_embed_dim,
                    k_freq=args.k_freq, k_sigma=args.k_sigma, t_freq=args.t_freq,
                    t_sigma=args.t_sigma, ff_seed=args.ff_seed, ncc=ncc, slice=slc),
               os.path.join(args.save_dir, f'model_slice_{slc:02d}.pt'))
    np.save(os.path.join(args.save_dir, f'nik_slice_{slc:02d}_cplx.npy'), img_cplx)
    if device.type == 'cuda':
        print(f'    RESOURCES slice {slc:02d}: peak_gpu_MB {torch.cuda.max_memory_allocated()/2**20:.0f} '
              f'params {sum(p.numel() for p in model.parameters())} best_heldout {best_heldout:.4e}', flush=True)
    run.finish(best_heldout_mse=best_heldout, best_step=best_step, params=sum(p.numel() for p in model.parameters()),
               peak_gpu_mb=W.peak_gpu_mb(), wall_s=time.time() - t_slice)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)                          # resume ckpt only; final weights saved above
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default='/net/beegfs/users/P101440/grasp_pro_py/results_ref')
    ap.add_argument('--save-dir', default='/net/beegfs/users/P101440/grasp_pro_py/results_nik')
    ap.add_argument('--slices', default='13', help='"13" | "0:27" | "all" | "3,13,20"')
    # ---- WINNING RECIPE (nik-autoresearch E1-E8 ablation, slice 13) ----
    # WIRE + Fourier features + residual skips, depth 12, hidden 512.
    # held-out 0.323, contrast swing 51%, nav-corr 0.885 (CS-70 swing = 37.5%).
    # model
    ap.add_argument('--model', default='wire_ff_res',
                    choices=['wire', 'wire_ff', 'wire_ff_res', 'wire_ff_subspace',
                             'wire_ff_res_radial', 'wire_ff_pk', 'wire_ff_patlak', 'wire_ff_tofts'])
    ap.add_argument('--aif-file', default='/net/beegfs/users/P101440/DCE_NIK/aif_slice21.npz')
    ap.add_argument('--tofts-basis', default=None, help='basis npz from nik_tofts_basis.py (model wire_ff_tofts)')
    ap.add_argument('--patlak-free', type=int, default=0, help='F free SIREN atoms appended to the fixed Patlak basis')
    ap.add_argument('--bin-spokes', type=int, default=0, help='Task1 ARM B: quantize acquired-spoke times to bin-centres, N spokes/bin (0=continuous ARM A)')
    # PK / gamma-variate temporal atoms (--model wire_ff_pk). n_pk gamma bolus atoms +
    # 1 baseline + (rank-n_pk-1) free residual atoms. -1 = auto (~2/3 gamma). 0 free = hard prior.
    ap.add_argument('--n-pk', type=int, default=-1, help='# gamma bolus atoms (-1 auto); rest = baseline + free residual')
    # temporal TV (Huber, edge-preserving) prior on the Phi atoms -- the P0 temporal denoiser.
    # 0 = OFF (default). Sweep the weight; delta is the scale-free edge threshold (per-step, /atom-RMS).
    ap.add_argument('--ktv21-weight', type=float, default=0.0,
                    help='l2,1 temporal prior on the k-space output (works for non-subspace models); 0 = off')
    ap.add_argument('--ktv21-nk', type=int, default=1024, help='k-locations sampled per step')
    ap.add_argument('--ktv21-nt', type=int, default=64, help='t-grid size for the k-space TV')
    ap.add_argument('--ktv21-delta', type=float, default=0.1, help='Huber edge threshold')
    ap.add_argument('--phi-tv-weight', type=float, default=0.0, help='temporal-TV (Huber) weight on Phi atoms; 0 = off')
    ap.add_argument('--phi-tv-delta', type=float, default=0.1, help='Huber edge threshold (scale-free per-step |dPhi|)')
    ap.add_argument('--phi-tv-grid', type=int, default=256, help='dense t-grid size for the TV penalty')
    # A0 SPIRiT k-space coil-consistency prior. 0 = off.
    ap.add_argument('--spirit-weight', type=float, default=0.0, help='SPIRiT ||(G-I)x||^2 weight; 0 = off')
    ap.add_argument('--spirit-kernel', default='/net/beegfs/users/P101440/DCE_NIK/spirit_kernel.npz')
    ap.add_argument('--spirit-patch', type=int, default=24, help='cartesian patch size for the SPIRiT penalty')
    ap.add_argument('--radial-alpha', type=float, default=1.0,
                    help='(wire_ff_res_radial) |k|-dependent FF warp strength; 0 = no warp')
    # factorized low-rank model (--model wire_ff_subspace). rank = temporal-DoF knob (sweep > 5).
    ap.add_argument('--rank', type=int, default=12, help='subspace rank R (temporal DoF)')
    ap.add_argument('--phi-ortho', action='store_true', help='hard QR orthonormalize Phi (fix the gauge)')
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
    ap.add_argument('--spoke-keep-file', default=None,
                    help='.npy of view indices to train on (spoke-reduction; identical acquired set as CS, no heldout)')
    ap.add_argument('--spoke-heldout-file', default=None,
                    help='.npy of view indices used ONLY for early stopping (VAL); disjoint from keep file')
    ap.add_argument('--keep-heldout', action='store_true',
                    help='with --spoke-keep-file, score the NON-kept spokes as heldout for early stopping')
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
    ap.add_argument('--no-wandb', action='store_true', help='skip wandb (project dce_nik, offline fallback)')
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
