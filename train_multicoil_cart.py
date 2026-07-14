#!/usr/bin/env python
"""multicoil + time NIK trainer

mirror of train_cart_eval.py but with:
- model input: (kx, ky, t, coil_idx) - t in [-1, 1], coil via learned embedding
- model output: (Re, Im) for that single (coord, t, coil) sample
- single shared envelope normalizer fit on all (coil, t) data
- cartesian eval set synthesized from radial gt_img + coil_maps + SP across all T frames
- sense coil combine at image-metric time
"""
import argparse
import random
import logging
import numpy as np
import torch
import torch.nn.functional as F
# TF32 tensor cores for fp32 matmuls (free speedup on A100, negligible precision cost
# for k-space MSE). We train in fp32, so this accelerates the WIRE linear layers.
torch.set_float32_matmul_precision("high")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb
from coolname import generate_slug

from utils.io_utils import load_config, setup_logging, unique_output_dir, copy_config_to_output
from utils.wandb_utils import WandbLogger

from nik_io import load_event, synthesize_cartesian_from_radial
from nik_model import WIRE_KXY_COIL_T_REIM
from nik_train import prepare_tensors
from kspace_normalization import compute_dcf_radial, compute_radius, KSpaceNormalizer
from nik_focal_loss import composable_kspace_loss, split_residual_norm_by_k, _residual_magsq


@torch.no_grad()
def _predict_chunked(model, xs, ts, cs, chunk=262144):
    """run the (kxy, t, coil_idx) model in chunks to bound activation memory"""
    outs = []
    for i in range(0, xs.shape[0], chunk):
        outs.append(model(xs[i:i + chunk], ts[i:i + chunk], cs[i:i + chunk]))
    return torch.cat(outs, dim=0)


def _highband_energy(img, hi_frac=0.5):
    """energy in the outer (|k| > hi_frac * r_max) radial band of the image's 2D spectrum.
    proxy for fine-detail / sharpness content. returns absolute high-band energy."""
    F = np.fft.fftshift(np.abs(np.fft.fft2(img)))
    Hh, Ww = img.shape
    cy, cx = Hh // 2, Ww // 2
    Y, X = np.indices((Hh, Ww))
    r = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2).astype(int)
    prof = np.bincount(r.ravel(), F.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    rmax = len(prof)
    return float(prof[int(rmax * hi_frac):].sum())


def _radial_grad_profile(img, nbins=14):
    """per-pixel local gradient energy summed into annuli vs IMAGE radius.
    returns (sum_per_bin, count_per_bin) so callers can accumulate across frames.
    distinguishes uniform blur (flat ratio vs radius) from radius-dependent blur."""
    gy, gx = np.gradient(img.astype(np.float64))
    eg = gy ** 2 + gx ** 2
    Hh, Ww = img.shape
    cy, cx = Hh / 2.0, Ww / 2.0
    Y, X = np.indices((Hh, Ww))
    r = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    ri = np.clip((r / r.max() * nbins).astype(int), 0, nbins - 1).ravel()
    sums = np.bincount(ri, eg.ravel(), minlength=nbins)
    cnts = np.bincount(ri, minlength=nbins).astype(np.float64)
    return sums, cnts
from nik_recon import (
    ifft1d_kz_to_z,
    ifft1d_kz_to_z_cartesian,
    make_multicoil_time_radial_dataset,
    make_multicoil_time_cartesian_dataset,
    coil_combine_rss,
)
from nik_metrics import compute_image_metrics, compute_perceptual_metrics


def load_data(config):
    """radial train, synthesized cart eval, full (T, C) coverage"""
    data_cfg = config['data']
    radial_file = data_cfg['radial_file']
    z_slice_raw = data_cfg['z_slice_idx']
    subsample_frac = data_cfg.get('subsample_frac', 1.0)
    seed = config['training']['seed']
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # radial
    print(f"Loading radial data from {radial_file} ...", flush=True)
    event = load_event(radial_file, load_images=True, load_coil_maps=True)
    k_np    = np.transpose(event["k"],    (0, 2, 1, 3))
    traj_np = np.transpose(event["traj"], (0, 2, 1, 3))
    T, S, C, RO = k_np.shape
    print(f"Radial: T={T}, S={S}, C={C}, RO={RO}", flush=True)

    k_t, traj_t, scales, dims, k_scale = prepare_tensors(k_np, traj_np, data_device=device)
    k_img_space, n_z_slices, n_ro_per_slice, _ = ifft1d_kz_to_z(k_t, traj_t, t_frame=0)
    z_slice_idx = n_z_slices // 2 if z_slice_raw == -1 else int(z_slice_raw)
    print(f"Radial z slices: {n_z_slices}, readouts/slice: {n_ro_per_slice}", flush=True)

    # per-spoke continuous time (ms) if the simulator saved it; else None → bin-index fallback
    spoke_timing_dce = event.get("spoke_timing_dce")
    if spoke_timing_dce is not None:
        # transpose to (S, T) to match make_multicoil_time_radial_dataset's indexing.
        # h5 stores with axes reversed vs MATLAB; the matlab array is (min_lines, nt).
        spoke_timing_dce = np.asarray(spoke_timing_dce)
        if spoke_timing_dce.shape[0] == T and spoke_timing_dce.shape[1] != T:
            spoke_timing_dce = spoke_timing_dce.T   # → (S, T)
        t_max_ms = float(spoke_timing_dce.max())
        print(f"per-spoke continuous time loaded: shape={spoke_timing_dce.shape}, "
              f"t_max={t_max_ms:.1f} ms", flush=True)
    else:
        t_max_ms = None
        print("per-spoke timing not in radial file; using bin-index fallback for t", flush=True)

    # multicoil + time radial dataset
    (
        x_all, t_all, coil_all, y_all_raw,
        spoke_id_all, ro_id_all, frame_id_all, meta_rad,
    ) = make_multicoil_time_radial_dataset(
        k_img_space, traj_t, scales, dims,
        z_slice_idx=z_slice_idx, n_slices=n_z_slices, compute_device=device,
        spoke_timing_dce=spoke_timing_dce, t_max_ms=t_max_ms,
    )
    print(f"Radial multicoil+time dataset: N={meta_rad['N']} "
          f"(T={T}, C={C}, n_ro_per_slice={meta_rad['n_ro_per_slice']}, RO={RO})",
          flush=True)

    # spoke subsampling: same train/heldout split across all (t, c)
    n_unique_spokes = int(spoke_id_all.max().item()) + 1
    n_train_spokes  = max(1, int(n_unique_spokes * subsample_frac))
    g = torch.Generator(device=spoke_id_all.device).manual_seed(seed)
    perm = torch.randperm(n_unique_spokes, generator=g, device=spoke_id_all.device)
    train_spokes = perm[:n_train_spokes]
    train_mask = torch.isin(spoke_id_all, train_spokes)
    train_idx = torch.where(train_mask)[0]
    print(f"Spokes: {n_train_spokes}/{n_unique_spokes} ({subsample_frac:.0%}), "
          f"{train_idx.shape[0]} train points (per t, per coil)", flush=True)

    # synthesize cartesian k-space at all T frames, all coils
    print(f"Synthesizing Cartesian k-space from radial ({T} DCE bins, {C} coils) ...", flush=True)
    cart_event = synthesize_cartesian_from_radial(radial_file, T_target=T, event=event)
    k_cart_t  = torch.from_numpy(cart_event["k_cart"].astype(np.complex64)).to(device)
    k_cart_z  = ifft1d_kz_to_z_cartesian(k_cart_t)
    z_slice_cart = k_cart_z.shape[2] // 2 if z_slice_raw == -1 else int(z_slice_raw)

    # cart bin centers in ms, matched to the same normalization as radial spoke times
    rc_tim_s = event.get("rc_tim")
    if rc_tim_s is not None and t_max_ms is not None:
        rc_tim_ms = np.asarray(rc_tim_s).reshape(-1) * 1000.0
        frame_times_ms = rc_tim_ms[:T] if rc_tim_ms.size >= T else rc_tim_ms
    else:
        frame_times_ms = None

    (
        x_cart, t_cart, coil_cart, y_cart_raw, frame_id_cart, meta_cart,
    ) = make_multicoil_time_cartesian_dataset(
        k_cart_z, z_slice_idx=z_slice_cart, scales_radial=scales, compute_device=device,
        frame_times_ms=frame_times_ms, t_max_ms=t_max_ms,
    )
    print(f"Cart multicoil+time eval: N={meta_cart['N']} "
          f"(T={meta_cart['T']}, C={meta_cart['n_coils']}, "
          f"nky={meta_cart['nky']}, nkx={meta_cart['nkx']})", flush=True)

    coil_maps = cart_event.get("coil_maps")

    return {
        # radial
        "x_all": x_all, "t_all": t_all, "coil_all": coil_all,
        "y_all_raw": y_all_raw,
        "spoke_id_all": spoke_id_all, "ro_id_all": ro_id_all, "frame_id_all": frame_id_all,
        "train_idx": train_idx,
        "k_img_space": k_img_space, "traj_t": traj_t,
        "T": T, "S": S, "C": C, "RO": RO,
        "n_ro_per_slice": n_ro_per_slice,
        "z_slice_idx": z_slice_idx, "n_z_slices": n_z_slices,
        "scales": scales, "dims": dims,
        # cart
        "x_cart": x_cart, "t_cart": t_cart, "coil_cart": coil_cart,
        "y_cart_raw": y_cart_raw, "frame_id_cart": frame_id_cart,
        "meta_cart": meta_cart,
        "coil_maps": coil_maps,
        "z_slice_cart": z_slice_cart,
        # ground-truth + CS reference images
        "gt_pad": cart_event.get("gt_img"),     # (T, kz, RL_pad, AP_pad) binned bare GT, padded
        "cs_img": event.get("rc_img"),          # (n_rc, n_slices, RL, AP) CS recon, magnitude
        "cs_tim": event.get("rc_tim"),          # (n_rc,) CS recon frame times (s)
        # bookkeeping
        "subsample_frac": subsample_frac,
        "n_unique_spokes": n_unique_spokes,
        "n_train_spokes": n_train_spokes,
    }


def main(config_path, data):
    """single run, multicoil + time radial train, multicoil + time cart eval"""
    random.seed()
    run_name = generate_slug(3) + "_mct_carteval"
    config = load_config(config_path)
    output_dir = unique_output_dir(config, run_name)
    copy_config_to_output(config_path, output_dir)
    setup_logging(config, output_dir)
    logging.info(f"Run: {run_name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = config['training']['seed']
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    # flat config for wandb
    mc = config['model']
    tc = config['training']
    train_config = {
        "model_family": mc.get('model_family', 'wire_coil_t'),
        "hidden":       mc['hidden'],
        "depth":        mc['depth'],
        "w0":           float(mc.get('w0', 30.0)),
        "s0":           float(mc.get('s0', 10.0)),
        "coil_embed_dim": int(mc.get('coil_embed_dim', 8)),
        "dropout":      float(mc.get('dropout', 0.0)),
        "optimizer":    tc['optimizer'],
        "lr":           tc['lr'],
        "batch_size":   tc['batch_size'],
        "steps":        tc['steps'],
        "eval_every":   tc['eval_every'],
        "grad_clip":    tc['grad_clip'],
        "weight_decay": tc.get('weight_decay', 0.0),
        "seed":         tc['seed'],
        "subsample_frac": data["subsample_frac"],
        "scheduler_patience": tc.get('scheduler_patience', 0),
        "scheduler_factor":   tc.get('scheduler_factor',   0.5),
        "scheduler_min_lr":   tc.get('scheduler_min_lr',   1e-6),
    }
    wandb_logger = WandbLogger(
        config=train_config, output_dir=output_dir,
        run_name=run_name, job_type="multicoil_time_cart_eval",
    )
    wandb_logger.initialize()

    # resolve hyperparams (wandb sweep overrides)
    wc = wandb.config
    hidden          = int(getattr(wc, "hidden", mc['hidden']))
    depth           = int(getattr(wc, "depth",  mc['depth']))
    w0              = float(getattr(wc, "w0",   mc.get('w0', 30.0)))
    s0              = float(getattr(wc, "s0",   mc.get('s0', 10.0)))
    coil_embed_dim  = int(getattr(wc, "coil_embed_dim", mc.get('coil_embed_dim', 8)))
    dropout         = float(getattr(wc, "dropout", mc.get('dropout', 0.0)))
    optimizer_name  = str(getattr(wc, "optimizer", tc['optimizer']))
    lr              = float(getattr(wc, "lr", tc['lr']))
    batch_size      = int(getattr(wc, "batch_size", tc['batch_size']))
    steps           = int(getattr(wc, "steps", tc['steps']))
    eval_every      = int(getattr(wc, "eval_every", tc['eval_every']))
    grad_clip       = float(getattr(wc, "grad_clip", tc['grad_clip']))
    weight_decay    = float(getattr(wc, "weight_decay", tc.get('weight_decay', 0.0)))
    subsample_frac  = float(getattr(wc, "subsample_frac", data["subsample_frac"]))
    scheduler_patience = int(getattr(wc, "scheduler_patience", tc.get('scheduler_patience', 0)))
    scheduler_factor   = float(getattr(wc, "scheduler_factor",   tc.get('scheduler_factor',   0.5)))
    scheduler_min_lr   = float(getattr(wc, "scheduler_min_lr",   tc.get('scheduler_min_lr',   1e-6)))
    plot_every  = tc['plot']['plot_every']
    log_scale   = tc['plot'].get('log_scale', True)
    console_every = config['logging']['console_every']

    # unpack
    x_all        = data["x_all"]
    t_all        = data["t_all"]
    coil_all     = data["coil_all"]
    y_all_raw    = data["y_all_raw"]
    spoke_id_all = data["spoke_id_all"]
    train_idx    = data["train_idx"]
    C            = data["C"]
    T            = data["T"]
    x_cart       = data["x_cart"]
    t_cart       = data["t_cart"]
    coil_cart    = data["coil_cart"]
    y_cart_raw   = data["y_cart_raw"]
    meta_cart    = data["meta_cart"]
    nky, nkx     = meta_cart["nky"], meta_cart["nkx"]

    # resubsample if changed via sweep
    if subsample_frac != data["subsample_frac"]:
        n_unique = data["n_unique_spokes"]
        n_train  = max(1, int(n_unique * subsample_frac))
        g = torch.Generator(device=spoke_id_all.device).manual_seed(seed)
        perm = torch.randperm(n_unique, generator=g, device=spoke_id_all.device)
        train_mask = torch.isin(spoke_id_all, perm[:n_train])
        train_idx  = torch.where(train_mask)[0]
        logging.info(f"Re-subsampled: {n_train}/{n_unique} spokes ({subsample_frac:.0%})")

    # loss + normalization config
    loss_cfg = config.get('loss', {})
    norm_cfg = config.get('normalization', {})

    def _loss_or_norm(name, default):
        return getattr(wc, name, loss_cfg.get(name, norm_cfg.get(name, default)))

    use_envelope      = bool(_loss_or_norm('use_envelope', True))
    use_dcf           = bool(_loss_or_norm('use_dcf', True))
    dcf_power         = float(_loss_or_norm('dcf_power', 0.0))
    use_focal         = bool(_loss_or_norm('use_focal', False))
    focal_alpha       = float(_loss_or_norm('focal_alpha', 1.0))
    focal_normalize   = bool(_loss_or_norm('focal_normalize', True))
    focal_log_matrix  = bool(_loss_or_norm('focal_log_matrix', False))
    focal_warmup_steps = int(_loss_or_norm('focal_warmup_steps', 1000))
    dcf_method = norm_cfg.get('dcf_method', 'simple_ramp')

    # dcf from kxy alone (geometry, shared across (t, coil))
    kcoords_radial = x_all
    dcf = compute_dcf_radial(kcoords_radial, method=dcf_method) if use_dcf else torch.ones(
        kcoords_radial.shape[0], device=kcoords_radial.device
    )

    # one shared envelope, fit on all train-spoke samples (across all t, c)
    kcoords_train = kcoords_radial[train_idx]
    y_train_raw   = y_all_raw[train_idx]
    dcf_train_norm = dcf[train_idx]

    normalizer = KSpaceNormalizer()
    if use_envelope:
        normalizer.fit(
            kcoords_train, y_train_raw, dcf=dcf_train_norm,
            envelope_bins=norm_cfg.get('envelope_bins', 128),
            envelope_statistic=norm_cfg.get('envelope_statistic', 'weighted_rms'),
            envelope_smooth_method=norm_cfg.get('envelope_smooth_method', 'moving_average'),
            envelope_smooth_width=norm_cfg.get('envelope_smooth_width', 5),
            envelope_floor_fraction=norm_cfg.get('envelope_floor_fraction', 1e-3),
            global_scale_method=norm_cfg.get('global_scale_method', 'weighted_rms'),
        )
    else:
        from kspace_normalization import compute_global_scale, _to_complex, RadialEnvelope
        y_c = _to_complex(y_train_raw)
        normalizer.global_scale = compute_global_scale(y_c, dcf=dcf_train_norm)
        r_max = float(compute_radius(kcoords_train).max().item())
        normalizer.envelope = RadialEnvelope(
            bin_centers=torch.linspace(0, r_max, 128),
            raw_shell_values=torch.ones(128),
            smoothed_shell_values=torch.ones(128),
            floor_value=1.0, r_max=r_max,
            statistic="flat", smooth_method="none",
        )
        normalizer._fitted = True

    y_all  = normalizer.normalize(kcoords_radial, y_all_raw)
    y_cart = normalizer.normalize(x_cart,         y_cart_raw)

    logging.info(
        f"Normalization: use_envelope={use_envelope}, use_dcf={use_dcf}, "
        f"dcf_power={dcf_power}, use_focal={use_focal}, focal_alpha={focal_alpha}, "
        f"focal_log_matrix={focal_log_matrix}, focal_warmup_steps={focal_warmup_steps}, "
        f"global_scale={normalizer.global_scale:.4f}"
    )
    wandb.config.update({
        "use_envelope": use_envelope, "use_dcf": use_dcf, "dcf_power": dcf_power,
        "use_focal": use_focal, "focal_alpha": focal_alpha,
        "focal_normalize": focal_normalize, "focal_log_matrix": focal_log_matrix,
        "focal_warmup_steps": focal_warmup_steps,
        "subsample_frac_actual": subsample_frac,
        "n_coils": C, "T": T,
        "n_cart_eval_points": meta_cart["N"], "nky": nky, "nkx": nkx,
    }, allow_val_change=True)

    # build model
    model = WIRE_KXY_COIL_T_REIM(
        n_coils=C, coil_embed_dim=coil_embed_dim,
        hidden=hidden, depth=depth, w0=w0, s0=s0, dropout=dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    wandb.config.update({"n_params": n_params, "model_family": "wire_coil_t"}, allow_val_change=True)
    logging.info(f"Model: WIRE_KXY_COIL_T h={hidden} d={depth} w0={w0} s0={s0} "
                 f"coil_embed_dim={coil_embed_dim} params={n_params}")

    # torch.compile: fuse the Gabor-layer kernels (model is launch-bound at this size).
    # state_dict save/restore stays consistent since it's all on the same (compiled) object.
    use_compile = bool(tc.get("compile", True))
    if use_compile and device == "cuda":
        try:
            model = torch.compile(model)
            logging.info("torch.compile enabled")
        except Exception as e:
            logging.warning(f"torch.compile failed, falling back to eager: {e}")

    # optimizer
    if optimizer_name == "Adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "SGD":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    scheduler = None
    scheduler_type = tc.get('scheduler_type', 'plateau')
    if scheduler_type == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=lr * 10, total_steps=int(steps),
            pct_start=0.1, anneal_strategy='cos',
            div_factor=10.0, final_div_factor=1e3,
        )
    elif scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=int(steps), eta_min=scheduler_min_lr,
        )
    elif scheduler_patience > 0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=scheduler_factor,
            patience=scheduler_patience, min_lr=scheduler_min_lr,
        )

    # device tensors
    x_all_dev  = x_all.to(device)
    t_all_dev  = t_all.to(device)
    coil_all_dev = coil_all.to(device)
    y_all_dev  = y_all.to(device)
    dcf_dev    = dcf.to(device)

    x_train    = x_all_dev[train_idx]
    t_train    = t_all_dev[train_idx]
    coil_train = coil_all_dev[train_idx]
    y_train    = y_all_dev[train_idx]
    dcf_train  = dcf_dev[train_idx]
    N_train    = x_train.shape[0]

    # heldout spokes
    train_spoke_set = torch.unique(spoke_id_all[train_idx])
    heldout_mask    = ~torch.isin(spoke_id_all, train_spoke_set)
    heldout_idx     = torch.where(heldout_mask)[0]
    has_heldout = heldout_idx.numel() > 0
    if has_heldout:
        x_heldout    = x_all_dev[heldout_idx]
        t_heldout    = t_all_dev[heldout_idx]
        coil_heldout = coil_all_dev[heldout_idx]
        y_heldout    = y_all_dev[heldout_idx]
        logging.info(f"Heldout radial: {heldout_idx.shape[0]} points "
                     f"({data['n_unique_spokes'] - len(train_spoke_set)} spokes)")
    else:
        x_heldout = t_heldout = coil_heldout = y_heldout = None
        logging.info("No held-out radial (subsample_frac=1.0)")

    # cart eval tensors + disk mask
    x_cart_dev    = x_cart.to(device)
    t_cart_dev    = t_cart.to(device)
    coil_cart_dev = coil_cart.to(device)
    y_cart_dev    = y_cart.to(device)
    radial_rmax   = float(compute_radius(kcoords_train).max().item())
    cart_r        = compute_radius(x_cart_dev)
    cart_in_disk_mask = cart_r <= (radial_rmax + 1e-6)
    n_cart_in_disk = int(cart_in_disk_mask.sum().item())
    wandb.config.update({
        "radial_rmax": radial_rmax,
        "n_cart_eval_points_in_disk": n_cart_in_disk,
    }, allow_val_change=True)

    warmup_steps = tc.get('warmup_steps', steps // 5)
    image_metric_every = tc.get('image_metric_every', 1000)

    # plot steps
    plot_steps = {1, steps}
    s = plot_every
    while s <= steps:
        plot_steps.add(s); s += plot_every

    # training
    model.train()
    best_heldout_loss = float("inf")
    best_step = -1
    best_cart_loss_in_disk = float("inf")
    best_state = None
    last_cart_loss = None
    last_cart_loss_in_disk = None
    last_heldout_loss = None

    logging.info(
        f"Training {steps} steps on {N_train} radial points, "
        f"eval on {meta_cart['N']} cart points ({n_cart_in_disk} in-disk)"
    )

    for step in range(1, steps + 1):
        idx = torch.randint(0, N_train, (batch_size,), device=device)
        x_b  = x_train[idx]
        t_b  = t_train[idx]
        c_b  = coil_train[idx]
        y_b  = y_train[idx]
        w_b  = dcf_train[idx]

        opt.zero_grad(set_to_none=True)
        y_pred = model(x_b, t_b, c_b)

        focal_progress = (
            min(1.0, step / float(focal_warmup_steps)) if focal_warmup_steps > 0 else 1.0
        )
        want_diag = (step % eval_every == 0 or step == steps or step == 1)
        if want_diag:
            loss, focal_diag = composable_kspace_loss(
                y_pred, y_b,
                dcf=w_b, use_dcf=use_dcf, dcf_power=dcf_power,
                use_focal=use_focal, focal_alpha=focal_alpha,
                focal_normalize=focal_normalize, focal_log_matrix=focal_log_matrix,
                focal_warmup_progress=focal_progress,
                return_diagnostics=True,
            )
        else:
            loss = composable_kspace_loss(
                y_pred, y_b,
                dcf=w_b, use_dcf=use_dcf, dcf_power=dcf_power,
                use_focal=use_focal, focal_alpha=focal_alpha,
                focal_normalize=focal_normalize, focal_log_matrix=focal_log_matrix,
                focal_warmup_progress=focal_progress,
                return_diagnostics=False,
            )
            focal_diag = None

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        if scheduler is not None and scheduler_type in ("onecycle", "cosine"):
            scheduler.step()
        train_loss = float(loss.item())

        # eval
        if step % eval_every == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                # --- in-loop full-Cartesian eval disabled for speed (70M-pt forward
                #     every eval was ~8x the training compute and only fed a logged
                #     scalar; best-state + scheduler run on heldout below).
                #     re-enable if you need cart_eval curves during training. ---
                # cart_pred = _predict_chunked(model, x_cart_dev, t_cart_dev, coil_cart_dev)
                # last_cart_loss = float(F.mse_loss(cart_pred, y_cart_dev).item())
                # last_cart_loss_in_disk = float(
                #     F.mse_loss(cart_pred[cart_in_disk_mask], y_cart_dev[cart_in_disk_mask]).item()
                # )
                # del cart_pred
                last_cart_loss = None
                last_cart_loss_in_disk = None
                if has_heldout:
                    held_pred = _predict_chunked(model, x_heldout, t_heldout, coil_heldout)
                    last_heldout_loss = float(F.mse_loss(held_pred, y_heldout).item())
                    del held_pred
                else:
                    last_heldout_loss = None
            model.train()

            if (step >= warmup_steps and last_heldout_loss is not None
                    and last_heldout_loss < best_heldout_loss):
                best_heldout_loss = last_heldout_loss
                best_step = step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if step >= warmup_steps and last_cart_loss_in_disk is not None:
                best_cart_loss_in_disk = min(best_cart_loss_in_disk, last_cart_loss_in_disk)

            if scheduler is not None and scheduler_type == "plateau":
                sched_metric = last_heldout_loss if last_heldout_loss is not None else last_cart_loss_in_disk
                if sched_metric is not None:
                    scheduler.step(sched_metric)

        # logging
        log_dict = {"train/train_loss": train_loss}
        if focal_diag is not None:
            for k_, v_ in focal_diag.items():
                log_dict[f"train/{k_}"] = v_
            log_dict["train/focal_progress"] = focal_progress
            with torch.no_grad():
                r_magsq = _residual_magsq(y_pred.detach(), y_b)
                kr_split = split_residual_norm_by_k(r_magsq, x_b)
                for k_, v_ in kr_split.items():
                    log_dict[f"train/{k_}"] = v_
        if step % eval_every == 0 or step == steps:
            if last_cart_loss is not None:
                log_dict["train/cart_eval_loss"] = last_cart_loss
            if last_cart_loss_in_disk is not None:
                log_dict["train/cart_eval_loss_in_disk"] = last_cart_loss_in_disk
            if last_heldout_loss is not None:
                log_dict["train/heldout_spoke_loss"] = last_heldout_loss
        if scheduler is not None:
            log_dict["train/lr"] = opt.param_groups[0]["lr"]
        wandb_logger.log(log_dict, step=step)

        if step % console_every == 0:
            msg = f"step {step:6d}  train {train_loss:.3e}"
            if last_cart_loss is not None:        msg += f"  cart_full {last_cart_loss:.3e}"
            if last_cart_loss_in_disk is not None: msg += f"  cart_disk {last_cart_loss_in_disk:.3e}"
            if last_heldout_loss is not None:     msg += f"  heldout {last_heldout_loss:.3e}"
            logging.info(msg)

    # restore best
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        logging.info(f"Loaded best model from step {best_step}, heldout_loss={best_heldout_loss:.4e}")
    else:
        logging.info("No best checkpoint, using final state")
    model.eval()

    # final eval + SENSE combined image metric at every frame
    final_cart_loss = None
    final_cart_loss_in_disk = None
    final_heldout_loss = None
    with torch.no_grad():
        cart_pred_eval = _predict_chunked(model, x_cart_dev, t_cart_dev, coil_cart_dev)
        final_cart_loss = float(F.mse_loss(cart_pred_eval, y_cart_dev).item())
        final_cart_loss_in_disk = float(
            F.mse_loss(cart_pred_eval[cart_in_disk_mask], y_cart_dev[cart_in_disk_mask]).item()
        )
        del cart_pred_eval
        if has_heldout:
            held_pred = _predict_chunked(model, x_heldout, t_heldout, coil_heldout)
            final_heldout_loss = float(F.mse_loss(held_pred, y_heldout).item())
            del held_pred

    # image-domain evaluation: model / NUFFT / CS, all vs ground truth (native FOV)
    try:
        coil_maps = data["coil_maps"]                 # (C, kz, RL_pad, AP_pad)
        z_cart    = data["z_slice_cart"]
        gt_pad    = data["gt_pad"]                     # (T, kz, RL_pad, AP_pad) binned bare GT
        cs_img    = data["cs_img"]                     # (n_rc, n_slices, RL, AP) or None
        if coil_maps is None or gt_pad is None:
            raise RuntimeError("coil_maps or gt_pad missing; cannot evaluate vs GT")
        sens = coil_maps[:, z_cart, :, :].astype(np.complex64)   # (C, RL_pad, AP_pad)

        # model predicted cart k-space (chunked, denorm, disk-masked)
        with torch.no_grad():
            cart_pred_norm   = _predict_chunked(model, x_cart_dev, t_cart_dev, coil_cart_dev)
            cart_pred_denorm = normalizer.denormalize(x_cart_dev, cart_pred_norm)
        pred_r = cart_pred_denorm[:, 0].view(T, C, nky, nkx).cpu().numpy()
        pred_i = cart_pred_denorm[:, 1].view(T, C, nky, nkx).cpu().numpy()
        k_pred_TC = pred_r + 1j * pred_i
        N_per_tc = meta_cart["N_per_tc"]
        cart_disk_mask_2d = cart_in_disk_mask[:N_per_tc].view(nky, nkx).cpu().numpy()
        k_pred_TC = k_pred_TC * cart_disk_mask_2d[None, None, :, :]

        # crop from padded grid to native FOV (matches GT/CS), center crop both axes
        RL_pad, AP_pad = gt_pad.shape[2], gt_pad.shape[3]
        if cs_img is not None:
            nat_RL, nat_AP = cs_img.shape[2], cs_img.shape[3]
        else:
            nat_RL, nat_AP = RL_pad, AP_pad
        r0 = (RL_pad - nat_RL) // 2; r1 = r0 + nat_RL
        a0 = (AP_pad - nat_AP) // 2; a1 = a0 + nat_AP
        crop    = lambda im: im[r0:r1, a0:a1]
        norm_im = lambda a: a / (a.max() + 1e-12)

        if T <= 6:
            frames_to_show = list(range(T))
        else:
            frames_to_show = sorted(set([0, T//5, 2*T//5, T//2, 3*T//5, 4*T//5, T-1]))

        cs_aligned = (cs_img is not None and cs_img.shape[0] == T)
        denom_s = np.sum(np.abs(sens) ** 2, axis=0) + 1e-10

        # per-frame: model vs GT and CS vs GT over all frames (cheap)
        model_metrics, cs_metrics = [], []
        gt_imgs, model_imgs, cs_imgs_show = {}, {}, {}
        hb_gt = hb_model = hb_cs = 0.0   # accumulate high-band energy for retention metric
        NB = 14   # annuli for radial sharpness-vs-image-radius profile
        rp_gt = np.zeros(NB); rp_md = np.zeros(NB); rp_cs = np.zeros(NB); rp_cnt = np.zeros(NB)
        for t_idx in range(T):
            gt_native = norm_im(np.abs(crop(gt_pad[t_idx, z_cart])))

            pred_coil = np.stack(
                [np.fft.fftshift(np.fft.ifft2(k_pred_TC[t_idx, c])).T for c in range(C)], axis=0)
            img_model = np.abs(np.sum(np.conj(sens) * pred_coil, axis=0) / denom_s)
            img_model_n = norm_im(crop(img_model))
            mm = compute_image_metrics(img_model_n, gt_native)
            mp = compute_perceptual_metrics(img_model_n, gt_native)
            model_metrics.append({"frame": t_idx, "psnr": mm["psnr_db"], "ssim": mm["ssim"],
                                  "dists": mp["DISTS"], "haarpsi": mp["HaarPSI"]})

            hb_gt    += _highband_energy(gt_native)
            hb_model += _highband_energy(img_model_n)
            sg, cg = _radial_grad_profile(gt_native, NB);     rp_gt += sg; rp_cnt += cg
            sm, _  = _radial_grad_profile(img_model_n, NB);   rp_md += sm

            if cs_aligned:
                img_cs_n = norm_im(np.abs(cs_img[t_idx, z_cart]))
                cm = compute_image_metrics(img_cs_n, gt_native)
                cp = compute_perceptual_metrics(img_cs_n, gt_native)
                cs_metrics.append({"frame": t_idx, "psnr": cm["psnr_db"], "ssim": cm["ssim"],
                                   "dists": cp["DISTS"], "haarpsi": cp["HaarPSI"]})
                hb_cs += _highband_energy(img_cs_n)
                sc, _ = _radial_grad_profile(img_cs_n, NB);    rp_cs += sc

            if t_idx in frames_to_show:
                gt_imgs[t_idx]    = gt_native
                model_imgs[t_idx] = img_model_n
                if cs_aligned:
                    cs_imgs_show[t_idx] = img_cs_n

        # NUFFT vs GT at shown frames only (expensive)
        from nik_recon import nufft2d_recon
        train_spoke_ids = torch.unique(spoke_id_all[train_idx]).cpu().numpy()
        k_img_sub = torch.zeros_like(data["k_img_space"])
        for sp_id in train_spoke_ids:
            k_img_sub[:, sp_id, :, :, :] = data["k_img_space"][:, sp_id, :, :, :]
        z_rad, n_z_rad = data["z_slice_idx"], data["n_z_slices"]
        nufft_imgs, nufft_metrics = {}, []
        for t_idx in frames_to_show:
            coil_imgs = []
            for c in range(C):
                img_c = nufft2d_recon(
                    k_img_sub, data["traj_t"], t_frame=t_idx, coil_idx=c,
                    z_slice_idx=z_rad, scales=data["scales"],
                    img_size=(nky, nkx), n_slices=n_z_rad, return_complex=True)
                coil_imgs.append(img_c)
            coil_imgs = np.asarray(coil_imgs)
            nh, nw = coil_imgs.shape[1], coil_imgs.shape[2]
            if nh != nky or nw != nkx:
                y0 = (nh - nky)//2; x0 = (nw - nkx)//2
                coil_imgs = coil_imgs[:, y0:y0+nky, x0:x0+nkx]
            img_nufft = np.abs(np.sum(np.conj(sens) * coil_imgs, axis=0) / denom_s)
            img_nufft_n = norm_im(crop(img_nufft))
            nufft_imgs[t_idx] = img_nufft_n
            nm  = compute_image_metrics(img_nufft_n, gt_imgs[t_idx])
            npc = compute_perceptual_metrics(img_nufft_n, gt_imgs[t_idx])
            nufft_metrics.append({"frame": t_idx, "psnr": nm["psnr_db"],
                                  "dists": npc["DISTS"], "haarpsi": npc["HaarPSI"]})

        def _agg(lst, k): return float(np.mean([d[k] for d in lst])) if lst else float("nan")
        model_psnr, model_dists, model_hp = _agg(model_metrics,"psnr"), _agg(model_metrics,"dists"), _agg(model_metrics,"haarpsi")
        cs_psnr,    cs_dists,    cs_hp     = _agg(cs_metrics,"psnr"),    _agg(cs_metrics,"dists"),    _agg(cs_metrics,"haarpsi")
        nufft_psnr, nufft_dists, nufft_hp = _agg(nufft_metrics,"psnr"), _agg(nufft_metrics,"dists"), _agg(nufft_metrics,"haarpsi")

        # high-band (fine-detail) retention vs GT: how much periphery-of-kspace energy each recovers
        model_hb_ret = float(hb_model / (hb_gt + 1e-12))
        cs_hb_ret    = float(hb_cs / (hb_gt + 1e-12)) if cs_aligned else float("nan")
        logging.info(f"high-band retention vs GT:  MODEL={model_hb_ret:.3f}  CS={cs_hb_ret:.3f}")

        # radial sharpness profile: edge-energy ratio vs IMAGE radius. flat -> uniform
        # blur; falling -> radius-dependent blur (INR k-space bandwidth limit).
        mean_gt = rp_gt / np.maximum(rp_cnt, 1)
        mean_md = rp_md / np.maximum(rp_cnt, 1)
        mean_cs = rp_cs / np.maximum(rp_cnt, 1)
        valid = mean_gt > (1e-3 * mean_gt.max())   # drop near-empty annuli (center dot / padded corners)
        rad_frac = (np.arange(NB) + 0.5) / NB
        ratio_md = np.where(valid, mean_md / np.maximum(mean_gt, 1e-12), np.nan)
        ratio_cs = np.where(valid, mean_cs / np.maximum(mean_gt, 1e-12), np.nan)
        def _band(arr, lo, hi):
            m = valid & (rad_frac >= lo) & (rad_frac < hi)
            return float(np.nanmean(arr[m])) if m.any() else float("nan")
        md_in, md_mid, md_out = _band(ratio_md,0,.33), _band(ratio_md,.33,.66), _band(ratio_md,.66,1.01)
        cs_in, cs_mid, cs_out = _band(ratio_cs,0,.33), _band(ratio_cs,.33,.66), _band(ratio_cs,.66,1.01)
        model_radial_falloff = float(md_out / (md_in + 1e-12))   # <1 => sharper center than edge
        logging.info(f"radial sharpness MODEL/GT  inner={md_in:.3f} mid={md_mid:.3f} outer={md_out:.3f}  "
                     f"(falloff out/in={model_radial_falloff:.3f})")
        logging.info(f"radial sharpness CS/GT     inner={cs_in:.3f} mid={cs_mid:.3f} outer={cs_out:.3f}")

        # log scalars (means) + model per-frame
        log_dict = {
            "model_vs_gt/psnr_mean": model_psnr, "model_vs_gt/dists_mean": model_dists, "model_vs_gt/haarpsi_mean": model_hp,
            "nufft_vs_gt/psnr_mean": nufft_psnr, "nufft_vs_gt/dists_mean": nufft_dists, "nufft_vs_gt/haarpsi_mean": nufft_hp,
        }
        if cs_aligned:
            log_dict.update({"cs_vs_gt/psnr_mean": cs_psnr, "cs_vs_gt/dists_mean": cs_dists, "cs_vs_gt/haarpsi_mean": cs_hp})
        for d in model_metrics:
            log_dict[f"model_vs_gt/psnr_t{d['frame']}"]    = d["psnr"]
            log_dict[f"model_vs_gt/dists_t{d['frame']}"]   = d["dists"]
            log_dict[f"model_vs_gt/haarpsi_t{d['frame']}"] = d["haarpsi"]
        log_dict["model_vs_gt/high_band_retention"] = model_hb_ret
        log_dict["model_vs_gt/sharp_inner"] = md_in
        log_dict["model_vs_gt/sharp_mid"]   = md_mid
        log_dict["model_vs_gt/sharp_outer"] = md_out
        log_dict["model_vs_gt/radial_falloff"] = model_radial_falloff
        if cs_aligned:
            log_dict["cs_vs_gt/high_band_retention"] = cs_hb_ret
            log_dict["cs_vs_gt/sharp_inner"] = cs_in
            log_dict["cs_vs_gt/sharp_outer"] = cs_out
        wandb_logger.log(log_dict, step=steps)
        wandb.run.summary.update({
            "model_vs_gt_psnr": model_psnr, "model_vs_gt_dists": model_dists, "model_vs_gt_haarpsi": model_hp,
            "nufft_vs_gt_psnr": nufft_psnr, "nufft_vs_gt_dists": nufft_dists, "nufft_vs_gt_haarpsi": nufft_hp,
            "cs_vs_gt_psnr": cs_psnr, "cs_vs_gt_dists": cs_dists, "cs_vs_gt_haarpsi": cs_hp,
            "model_minus_cs_psnr": model_psnr - cs_psnr if cs_aligned else float("nan"),
            "model_high_band_retention": model_hb_ret, "cs_high_band_retention": cs_hb_ret,
            "model_sharp_inner": md_in, "model_sharp_mid": md_mid, "model_sharp_outer": md_out,
            "model_radial_falloff": model_radial_falloff,
            "cs_sharp_inner": cs_in, "cs_sharp_outer": cs_out,
        })

        # radial sharpness TABLE: per-annulus edge-energy ratio so the falloff curve is visible
        rad_table = wandb.Table(columns=["radius_frac", "gt_grad", "cs_grad", "model_grad",
                                         "cs_over_gt", "model_over_gt"])
        for i in range(NB):
            if not valid[i]:
                continue
            rad_table.add_data(round(float(rad_frac[i]), 3), float(mean_gt[i]), float(mean_cs[i]),
                               float(mean_md[i]), round(float(ratio_cs[i]), 4), round(float(ratio_md[i]), 4))
        wandb_logger.log({"results/radial_sharpness_table": rad_table}, step=steps)

        # summary TABLE (method x metric) so wandb shows a sortable table, not just charts
        def _mean_ssim(lst): return float(np.mean([d.get("ssim", float("nan")) for d in lst])) if lst else float("nan")
        summary_table = wandb.Table(columns=["method", "PSNR", "DISTS", "HaarPSI", "SSIM", "high_band_ret", "n_frames"])
        summary_table.add_data("model", round(model_psnr, 3), round(model_dists, 4), round(model_hp, 4),
                               round(_mean_ssim(model_metrics), 4), round(model_hb_ret, 4), T)
        summary_table.add_data("NUFFT_subsampled", round(nufft_psnr, 3), round(nufft_dists, 4), round(nufft_hp, 4),
                               float("nan"), float("nan"), len(frames_to_show))
        if cs_aligned:
            summary_table.add_data("CS_recon", round(cs_psnr, 3), round(cs_dists, 4), round(cs_hp, 4),
                                   round(_mean_ssim(cs_metrics), 4), round(cs_hb_ret, 4), T)
        wandb_logger.log({"results/summary_table": summary_table}, step=steps)

        # per-frame TABLE: model & CS metrics for every frame
        pf_cols = ["frame", "model_PSNR", "model_DISTS", "model_HaarPSI"]
        if cs_aligned:
            pf_cols += ["cs_PSNR", "cs_DISTS", "cs_HaarPSI"]
        pf_table = wandb.Table(columns=pf_cols)
        for i, d in enumerate(model_metrics):
            row = [d["frame"], round(d["psnr"], 3), round(d["dists"], 4), round(d["haarpsi"], 4)]
            if cs_aligned:
                cd = cs_metrics[i]
                row += [round(cd["psnr"], 3), round(cd["dists"], 4), round(cd["haarpsi"], 4)]
            pf_table.add_data(*row)
        wandb_logger.log({"results/per_frame_table": pf_table}, step=steps)

        # comparison figure: rows = GT / Model / NUFFT / CS, cols = frames
        rows = [("Ground truth", gt_imgs), ("Model", model_imgs), ("NUFFT (subsampled)", nufft_imgs)]
        if cs_aligned:
            rows.append(("CS recon", cs_imgs_show))
        n_cols = len(frames_to_show); n_rows = len(rows)
        fig, ax = plt.subplots(n_rows, n_cols, figsize=(2.4*n_cols, 2.2*n_rows))
        ax = np.atleast_2d(ax)
        for ri, (lbl, imgs) in enumerate(rows):
            for j, t_idx in enumerate(frames_to_show):
                ax[ri, j].imshow(imgs[t_idx], cmap="gray"); ax[ri, j].axis("off")
                if ri == 0: ax[ri, j].set_title(f"t={t_idx}", fontsize=9)
            ax[ri, 0].text(-0.12, 0.5, lbl, transform=ax[ri, 0].transAxes,
                           rotation=90, va="center", ha="right", fontsize=10)
        plt.tight_layout()
        wandb_logger.log_figures({"plots/recon_comparison": fig}, step=steps)

        # individual images for flipping through in wandb
        recon_log = {}
        for t_idx in frames_to_show:
            recon_log[f"recon/gt_t{t_idx}"]    = wandb.Image(gt_imgs[t_idx])
            recon_log[f"recon/model_t{t_idx}"] = wandb.Image(model_imgs[t_idx])
            recon_log[f"recon/nufft_t{t_idx}"] = wandb.Image(nufft_imgs[t_idx])
            if cs_aligned:
                recon_log[f"recon/cs_t{t_idx}"] = wandb.Image(cs_imgs_show[t_idx])
        wandb_logger.log(recon_log, step=steps)

        # per-frame metric curves (model + CS over all frames)
        try:
            figc, axc = plt.subplots(1, 3, figsize=(15, 3))
            xs = list(range(T))
            for key, axi, title in [("psnr", axc[0], "PSNR"), ("dists", axc[1], "DISTS"), ("haarpsi", axc[2], "HaarPSI")]:
                axi.plot(xs, [d[key] for d in model_metrics], "o-", ms=3, label="model")
                if cs_aligned:
                    axi.plot(xs, [d[key] for d in cs_metrics], "s-", ms=3, label="CS")
                axi.set_title(f"{title} per frame  (vs GT)"); axi.set_xlabel("t"); axi.legend(fontsize=7)
            plt.tight_layout()
            wandb_logger.log_figures({"plots/per_frame_metrics": figc}, step=steps)
        except Exception as e:
            logging.warning(f"per-frame metric plot failed: {e}")

        logging.info(f"vs GT (mean over T={T}):  model PSNR={model_psnr:.2f}  DISTS={model_dists:.4f}  HaarPSI={model_hp:.4f}")
        logging.info(f"vs GT (mean over {len(frames_to_show)} frames):  NUFFT PSNR={nufft_psnr:.2f}  DISTS={nufft_dists:.4f}  HaarPSI={nufft_hp:.4f}")
        if cs_aligned:
            logging.info(f"vs GT (mean over T={T}):  CS    PSNR={cs_psnr:.2f}  DISTS={cs_dists:.4f}  HaarPSI={cs_hp:.4f}")

    except Exception as e:
        logging.warning(f"Final image-domain evaluation failed: {e}")
        import traceback; traceback.print_exc()

    # save the underlying module (strip torch.compile's _orig_mod wrapper) so the
    # checkpoint loads cleanly into a plain (uncompiled) model later
    model_to_save = getattr(model, "_orig_mod", model)
    wandb_logger.save_model(model_to_save, "model_best.pth", opt, steps, output_dir)
    wandb.run.summary.update({
        "best_heldout_spoke_loss":     best_heldout_loss,
        "best_step":                   best_step,
        "best_cart_eval_loss_in_disk": best_cart_loss_in_disk,
        "final_cart_eval_loss":        final_cart_loss,
        "final_cart_eval_loss_in_disk": final_cart_loss_in_disk,
        "final_heldout_spoke_loss":    final_heldout_loss if final_heldout_loss is not None else float("nan"),
        "total_steps":                 steps,
    })
    logging.info(
        f"Done. best_heldout={best_heldout_loss:.3e}  "
        f"best_cart_disk={best_cart_loss_in_disk:.3e}"
    )
    wandb_logger.finish()

    # release gpu memory before next sweep agent picks up
    try:
        del model, opt, best_state
        if scheduler is not None:
            del scheduler
    except Exception:
        pass
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="multicoil + time NIK: radial train, cartesian eval"
    )
    parser.add_argument('config_path', type=str)
    parser.add_argument('--sweep-id', type=str, default=None)
    parser.add_argument('--count',    type=int, default=50)
    parser.add_argument('--single',   action='store_true')
    args = parser.parse_args()

    config = load_config(args.config_path)
    data = load_data(config)

    if args.single:
        main(args.config_path, data)
    elif args.sweep_id:
        wandb.agent(args.sweep_id, function=lambda: main(args.config_path, data),
                    count=args.count)
    else:
        sweep_cfg = config.get('sweep')
        if not sweep_cfg:
            raise SystemExit("config has no [sweep] block; use --single for a single run")
        sweep_id = wandb.sweep(sweep=sweep_cfg)
        wandb.agent(sweep_id, function=lambda: main(args.config_path, data),
                    count=args.count)
