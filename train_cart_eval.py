#!/usr/bin/env python
"""radial training, cart eval"""
import argparse
import random
import logging
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb
from coolname import generate_slug

from utils.io_utils import (
    load_config, setup_logging, unique_output_dir, copy_config_to_output,
)
from utils.wandb_utils import WandbLogger

from nik_io import load_event, synthesize_cartesian_from_radial
from nik_model import (
    NIK_SIREN_KXY_REIM,
    NIK_SIREN_KXY_FF_REIM,
    ReLU_MLP_KXY_REIM,
    ELU_MLP_KXY_REIM,
    FF_ReLU_MLP_KXY_REIM,
    FF_ELU_MLP_KXY_REIM,
    WIRE_KXY_REIM,
    PolarKSpaceNet,
)
# weighted complex mse primary
from nik_loss import get_loss_fn
from nik_train import prepare_tensors
from kspace_normalization import compute_dcf_radial, KSpaceNormalizer
from losses import weighted_complex_mse
from nik_focal_loss import composable_kspace_loss, split_residual_norm_by_k, _residual_magsq
from nik_recon import (
    ifft1d_kz_to_z,
    make_fixed_frame_zslice_coil_dataset,
    ifft1d_kz_to_z_cartesian,
    make_cartesian_eval_dataset,
)
from nik_metrics import compute_image_metrics, compute_perceptual_metrics
from wandb_logger import (
    make_spoke_figure,
    make_error_map_figure,
    make_cartesian_error_map,
    make_cartesian_image_comparison,
)


def load_data(config):
    """radial train, cart eval data"""
    data_cfg = config['data']
    radial_file = data_cfg['radial_file']
    t_frame = data_cfg['t_frame']
    coil_idx = data_cfg['coil_idx']
    z_slice_raw = data_cfg['z_slice_idx']
    subsample_frac = data_cfg.get('subsample_frac', 1.0)
    seed = config['training']['seed']

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # radial data
    print(f"Loading radial data from {radial_file} ...", flush=True)
    event = load_event(radial_file, load_images=True, load_coil_maps=True)
    k_np, traj_np = event["k"], event["traj"]
    gt_img = event.get("gt_img")
    coil_maps_radial = event.get("coil_maps")

    # tscro layout
    k_np = np.transpose(k_np, (0, 2, 1, 3))
    traj_np = np.transpose(traj_np, (0, 2, 1, 3))
    T, S, C, RO = k_np.shape
    print(f"Radial k-space shape: T={T}, S={S}, C={C}, RO={RO}", flush=True)

    k_t, traj_t, scales, dims, k_scale = prepare_tensors(
        k_np, traj_np, data_device="cuda" if device == "cuda" else "cpu"
    )

    k_img_space, n_z_slices, n_ro_per_slice, kz_sort_order = ifft1d_kz_to_z(
        k_t, traj_t, t_frame=t_frame
    )
    print(f"Radial z slices: {n_z_slices}, readouts/slice: {n_ro_per_slice}", flush=True)

    z_slice_idx = n_z_slices // 2 if z_slice_raw == -1 else int(z_slice_raw)

    # raw kspace values
    x_all, y_all, kx_all, ky_all, spoke_id_all, ro_id_all, meta = \
        make_fixed_frame_zslice_coil_dataset(
            k_img_space, traj_t, scales, dims,
            y_scale=torch.tensor(1.0),
            t_fixed=t_frame,
            coil_fixed=coil_idx,
            z_slice_idx=z_slice_idx,
            n_slices=n_z_slices,
            compute_device=device,
        )
    print(f"Radial dataset: {meta['N']} points", flush=True)

    # subsample spokes
    n_unique_spokes = int(spoke_id_all.max().item()) + 1
    n_train_spokes = max(1, int(n_unique_spokes * subsample_frac))

    g = torch.Generator(device=spoke_id_all.device)
    g.manual_seed(seed)
    perm = torch.randperm(n_unique_spokes, generator=g, device=spoke_id_all.device)
    train_spokes = perm[:n_train_spokes]

    train_mask = torch.isin(spoke_id_all, train_spokes)
    train_idx = torch.where(train_mask)[0]
    print(f"Spoke subsampling: {n_train_spokes}/{n_unique_spokes} spokes "
          f"({subsample_frac:.0%}), {train_idx.shape[0]} points", flush=True)

    # cartesian data: synthesize from radial gt_img + coil_maps + SP
    print(f"Synthesizing Cartesian k-space from radial file ({T} DCE bins) ...", flush=True)
    cart_event = synthesize_cartesian_from_radial(radial_file, T_target=T, event=event)
    k_cart_np = cart_event["k_cart"]
    coil_maps_cart = cart_event.get("coil_maps")

    k_cart_t = torch.from_numpy(k_cart_np.astype(np.complex64))
    if device == "cuda":
        k_cart_t = k_cart_t.cuda()
    print(f"Cartesian k-space shape: {k_cart_t.shape}", flush=True)

    # cart kz to z
    k_cart_z = ifft1d_kz_to_z_cartesian(k_cart_t)
    nz_cart = k_cart_z.shape[2]

    # same zslice
    z_slice_cart = nz_cart // 2 if z_slice_raw == -1 else int(z_slice_raw)

    # raw cart kspace
    x_cart, y_cart, meta_cart = make_cartesian_eval_dataset(
        k_cart_z,
        t_fixed=min(t_frame, k_cart_z.shape[0] - 1),
        coil_fixed=coil_idx,
        z_slice_idx=z_slice_cart,
        scales_radial=scales,
        y_scale=torch.tensor(1.0),
        compute_device=device,
    )
    print(f"Cartesian eval dataset: {meta_cart['N']} points "
          f"(nky={meta_cart['nky']}, nkx={meta_cart['nkx']})", flush=True)

    ref_img_slice = None

    return {
        # raw radial
        "x_all": x_all, "y_all_raw": y_all,
        "spoke_id_all": spoke_id_all, "ro_id_all": ro_id_all,
        "train_idx": train_idx,
        "meta": meta, "k_scale": k_scale,
        "k_img_space": k_img_space, "traj_t": traj_t,
        "n_ro_per_slice": n_ro_per_slice,
        "T": T, "S": S, "C": C, "RO": RO,
        "z_slice_idx": z_slice_idx,
        "n_z_slices": n_z_slices,
        "scales": scales, "dims": dims,
        "coil_maps_radial": coil_maps_radial,
        # raw cart eval
        "x_cart": x_cart, "y_cart_raw": y_cart,
        "meta_cart": meta_cart,
        "ref_img_slice": ref_img_slice,
        "coil_maps_cart": coil_maps_cart,
        # shared
        "gt_img": gt_img,
        "subsample_frac": subsample_frac,
        "n_unique_spokes": n_unique_spokes,
        "n_train_spokes": n_train_spokes,
    }


def main(config_path, data):
    """single run, radial train, cart eval"""
    random.seed()
    run_name = generate_slug(3) + "_carteval"
    config = load_config(config_path)
    output_dir = unique_output_dir(config, run_name)
    copy_config_to_output(config_path, output_dir)
    setup_logging(config, output_dir)

    logging.info(f"Run: {run_name}")

    # flat config
    train_config = {
        "model_family": config['model'].get('model_family', 'siren'),
        "hidden": config['model']['hidden'],
        "depth": config['model']['depth'],
        "w0": config['model'].get('w0', 15),
        "k_freq": config['model'].get('k_freq', 64),
        "k_sigma": config['model'].get('k_sigma', 6.0),
        "s0": config['model'].get('s0', 10.0),
        "n_angular_modes": config['model'].get('n_angular_modes', 16),
        "radial_type": config['model'].get('radial_type', 'wire'),
        "dropout": config['model'].get('dropout', 0.0),
        "envelope_smooth_method": config.get('normalization', {}).get('envelope_smooth_method', 'moving_average'),
        "optimizer": config['training']['optimizer'],
        "lr": config['training']['lr'],
        "batch_size": config['training']['batch_size'],
        "steps": config['training']['steps'],
        "eval_every": config['training']['eval_every'],
        "grad_clip": config['training']['grad_clip'],
        "weight_decay": config['training'].get('weight_decay', 0.0),
        "loss_type": config['training'].get('loss_type', "mse"),
        "seed": config['training']['seed'],
        "subsample_frac": data["subsample_frac"],
        "scheduler_patience": config['training'].get('scheduler_patience', 0),
        "scheduler_factor": config['training'].get('scheduler_factor', 0.5),
        "scheduler_min_lr": config['training'].get('scheduler_min_lr', 1e-6),
    }

    wandb_logger = WandbLogger(
        config=train_config,
        output_dir=output_dir,
        run_name=run_name,
        job_type="training_cart_eval",
    )
    wandb_logger.initialize()

    # hyperparam resolve
    wc = wandb.config
    model_family = getattr(wc, "model_family", "siren")
    hidden = int(wc.hidden)
    depth = int(wc.depth)
    w0 = float(getattr(wc, "w0", 15))
    k_freq = int(getattr(wc, "k_freq", 64))
    k_sigma = float(getattr(wc, "k_sigma", 6.0))
    s0 = float(getattr(wc, "s0", 10.0))
    n_angular_modes = int(getattr(wc, "n_angular_modes", 16))
    radial_type = str(getattr(wc, "radial_type", "wire"))
    dropout = float(getattr(wc, "dropout", 0.0))
    optimizer_name = str(wc.optimizer)
    lr = float(wc.lr)
    batch_size = int(wc.batch_size)
    steps = int(wc.steps)
    eval_every = int(wc.eval_every)
    grad_clip = float(wc.grad_clip)
    weight_decay = float(getattr(wc, "weight_decay", 0.0))
    loss_type = str(getattr(wc, "loss_type", "mse"))
    loss_fn = get_loss_fn(loss_type)
    seed = int(wc.seed)
    subsample_frac = float(getattr(wc, "subsample_frac", data["subsample_frac"]))
    scheduler_patience = int(getattr(wc, "scheduler_patience", 0))
    scheduler_factor = float(getattr(wc, "scheduler_factor", 0.5))
    scheduler_min_lr = float(getattr(wc, "scheduler_min_lr", 1e-6))

    plot_every = config['training']['plot']['plot_every']
    log_scale = config['training']['plot']['log_scale']
    console_every = config['logging']['console_every']

    # reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # unpack raw
    x_all = data["x_all"]
    y_all_raw = data["y_all_raw"]
    spoke_id_all = data["spoke_id_all"]
    ro_id_all = data["ro_id_all"]
    train_idx = data["train_idx"]
    k_scale = data["k_scale"]
    meta_cart = data["meta_cart"]
    x_cart = data["x_cart"]
    y_cart_raw = data["y_cart_raw"]
    nky, nkx = meta_cart["nky"], meta_cart["nkx"]
    ref_img_slice = data.get("ref_img_slice")

    # loss + normalization config
    # priority: [loss] block (new), then [normalization] block (legacy), then defaults
    loss_cfg = config.get('loss', {})
    norm_cfg = config.get('normalization', {})

    def _loss_or_norm(name, default):
        # prefer wandb.config sweep override, then [loss], then [normalization]
        return getattr(wc, name, loss_cfg.get(name, norm_cfg.get(name, default)))

    use_envelope     = bool(_loss_or_norm('use_envelope', True))
    use_dcf          = bool(_loss_or_norm('use_dcf', True))
    dcf_power        = float(_loss_or_norm('dcf_power', 0.0))
    use_focal        = bool(_loss_or_norm('use_focal', False))
    focal_alpha      = float(_loss_or_norm('focal_alpha', 1.0))
    focal_normalize  = bool(_loss_or_norm('focal_normalize', True))
    focal_log_matrix = bool(_loss_or_norm('focal_log_matrix', False))
    focal_warmup_steps = int(_loss_or_norm('focal_warmup_steps', 1000))
    dcf_method = norm_cfg.get('dcf_method', 'simple_ramp')
    envelope_smooth_method = str(getattr(wc, "envelope_smooth_method",
        norm_cfg.get('envelope_smooth_method', 'moving_average')))

    # resubsample
    if subsample_frac != data["subsample_frac"]:
        n_unique_spokes = data["n_unique_spokes"]
        n_train_spokes = max(1, int(n_unique_spokes * subsample_frac))
        g = torch.Generator(device=spoke_id_all.device)
        g.manual_seed(seed)
        perm = torch.randperm(n_unique_spokes, generator=g, device=spoke_id_all.device)
        train_spokes = perm[:n_train_spokes]
        train_mask = torch.isin(spoke_id_all, train_spokes)
        train_idx = torch.where(train_mask)[0]
        logging.info(f"Re-subsampled: {n_train_spokes}/{n_unique_spokes} spokes ({subsample_frac:.0%})")

    kcoords_radial = x_all[:, :2]

    if use_dcf:
        dcf = compute_dcf_radial(kcoords_radial, method=dcf_method)
    else:
        dcf = torch.ones(kcoords_radial.shape[0], device=kcoords_radial.device)

    # fit on train only
    kcoords_train = kcoords_radial[train_idx]
    y_train_raw_for_norm = y_all_raw[train_idx]
    dcf_train_for_norm = dcf[train_idx]

    normalizer = KSpaceNormalizer()
    if use_envelope:
        normalizer.fit(
            kcoords_train, y_train_raw_for_norm, dcf=dcf_train_for_norm,
            envelope_bins=norm_cfg.get('envelope_bins', 128),
            envelope_statistic=norm_cfg.get('envelope_statistic', 'weighted_rms'),
            envelope_smooth_method=envelope_smooth_method,
            envelope_smooth_width=norm_cfg.get('envelope_smooth_width', 5),
            envelope_floor_fraction=norm_cfg.get('envelope_floor_fraction', 1e-3),
            global_scale_method=norm_cfg.get('global_scale_method', 'weighted_rms'),
        )
    else:
        from kspace_normalization import compute_global_scale, compute_radius, _to_complex, RadialEnvelope
        y_c = _to_complex(y_train_raw_for_norm)
        normalizer.global_scale = compute_global_scale(y_c, dcf=dcf_train_for_norm)
        r_max = float(compute_radius(kcoords_train).max().item())
        normalizer.envelope = RadialEnvelope(
            bin_centers=torch.linspace(0, r_max, 128),
            raw_shell_values=torch.ones(128),
            smoothed_shell_values=torch.ones(128),
            floor_value=1.0, r_max=r_max,
            statistic="flat", smooth_method="none",
        )
        normalizer._fitted = True

    y_all = normalizer.normalize(kcoords_radial, y_all_raw)
    y_cart = normalizer.normalize(x_cart[:, :2], y_cart_raw)

    logging.info(
        f"Normalization: fit_on_train_only=True, use_envelope={use_envelope}, "
        f"use_dcf={use_dcf}, dcf_power={dcf_power}, "
        f"use_focal={use_focal}, focal_alpha={focal_alpha}, "
        f"focal_log_matrix={focal_log_matrix}, focal_normalize={focal_normalize}, "
        f"focal_warmup_steps={focal_warmup_steps}, "
        f"global_scale={normalizer.global_scale:.4f}"
    )
    wandb.config.update({
        "use_envelope": use_envelope, "use_dcf": use_dcf, "dcf_power": dcf_power,
        "use_focal": use_focal, "focal_alpha": focal_alpha,
        "focal_normalize": focal_normalize, "focal_log_matrix": focal_log_matrix,
        "focal_warmup_steps": focal_warmup_steps,
        "normalizer_fit_on_train_only": True,
    }, allow_val_change=True)

    wandb.config.update({
        "n_train_points": int(train_idx.shape[0]),
        "n_cart_eval_points": meta_cart["N"],
        "nky": nky, "nkx": nkx,
        "subsample_frac_actual": subsample_frac,
    }, allow_val_change=True)

    # polar s_max
    _x_tmp = x_all[:, :2].to(device)
    _kx, _ky = _x_tmp[:, 0], _x_tmp[:, 1]
    _theta = torch.atan2(_ky, _kx)
    _theta0 = torch.remainder(_theta + 0.5 * np.pi, np.pi) - 0.5 * np.pi
    _s_coord = _kx * torch.cos(_theta0) + _ky * torch.sin(_theta0)
    s_max = float(_s_coord.abs().max().item())
    del _x_tmp, _kx, _ky, _theta, _theta0, _s_coord

    # build model
    if model_family == "relu":
        model = ReLU_MLP_KXY_REIM(in_dim=2, hidden=hidden, depth=depth).to(device)
    elif model_family == "elu":
        model = ELU_MLP_KXY_REIM(in_dim=2, hidden=hidden, depth=depth).to(device)
    elif model_family == "ff_relu":
        model = FF_ReLU_MLP_KXY_REIM(in_dim=2, k_freq=k_freq, k_sigma=k_sigma, hidden=hidden, depth=depth).to(device)
    elif model_family == "ff_elu":
        model = FF_ELU_MLP_KXY_REIM(in_dim=2, k_freq=k_freq, k_sigma=k_sigma, hidden=hidden, depth=depth).to(device)
    elif model_family == "ff_siren":
        model = NIK_SIREN_KXY_FF_REIM(x_dim=2, k_freq=k_freq, k_sigma=k_sigma, hidden=hidden, depth=depth, w0=w0).to(device)
    elif model_family == "polar":
        model = PolarKSpaceNet(n_angular_modes=n_angular_modes, radial_depth=depth, radial_width=hidden,
                               radial_type=radial_type, omega_0=w0, s_0=s0, s_max=s_max).to(device)
    elif model_family == "wire":
        model = WIRE_KXY_REIM(in_dim=2, hidden=hidden, depth=depth, w0=w0, s0=s0, dropout=dropout).to(device)
    else:
        model = NIK_SIREN_KXY_REIM(in_dim=2, hidden=hidden, depth=depth, w0=w0).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    wandb.config.update({"n_params": n_params, "model_family": model_family}, allow_val_change=True)
    logging.info(f"Model: {model_family}, hidden={hidden}, depth={depth}, params={n_params}")

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
    scheduler_type = config['training'].get('scheduler_type', 'plateau')
    if scheduler_type == "onecycle":
        # cosine warmup decay
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=lr * 10,  # 10x peak above base lr
            total_steps=int(config['training']['steps']),
            pct_start=0.1, anneal_strategy='cos',
            div_factor=10.0, final_div_factor=1e3,
        )
        logging.info(f"Scheduler: OneCycleLR (max_lr={lr*10:.2e}, total={config['training']['steps']})")
    elif scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=int(config['training']['steps']), eta_min=scheduler_min_lr,
        )
        logging.info(f"Scheduler: CosineAnnealingLR (eta_min={scheduler_min_lr})")
    elif scheduler_patience > 0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=scheduler_factor,
            patience=scheduler_patience, min_lr=scheduler_min_lr,
        )
        logging.info(f"Scheduler: ReduceLROnPlateau (patience={scheduler_patience})")

    # training tensors
    x_all_2d = x_all[:, :2].to(device)
    y_all_dev = y_all.to(device)
    x_train = x_all_2d[train_idx]
    y_train = y_all_dev[train_idx]
    N_train = x_train.shape[0]

    # training dcf
    dcf = dcf.to(device)
    dcf_train = dcf[train_idx]

    # heldout spokes
    all_spokes = torch.arange(data["n_unique_spokes"], device=spoke_id_all.device)
    train_spoke_set = torch.unique(spoke_id_all[train_idx])
    heldout_mask = ~torch.isin(spoke_id_all, train_spoke_set)
    heldout_idx = torch.where(heldout_mask)[0]
    has_heldout = heldout_idx.numel() > 0
    if has_heldout:
        x_heldout = x_all_2d[heldout_idx]
        y_heldout = y_all_dev[heldout_idx]
        logging.info(f"Held-out radial spokes: {data['n_unique_spokes'] - len(train_spoke_set)} spokes, "
                     f"{heldout_idx.shape[0]} points")
    else:
        x_heldout = y_heldout = None
        logging.info("No held-out radial spokes (subsample_frac=1.0)")

    # cart eval tensors
    x_cart_dev = x_cart.to(device)
    y_cart_dev = y_cart.to(device)
    radial_rmax = float(torch.sqrt((kcoords_train[:, 0] ** 2 + kcoords_train[:, 1] ** 2)).max().item())
    cart_r_dev = torch.sqrt(x_cart_dev[:, 0] ** 2 + x_cart_dev[:, 1] ** 2)
    cart_in_disk_mask = cart_r_dev <= (radial_rmax + 1e-6)
    cart_kspace_mask = cart_in_disk_mask.reshape(nky, nkx)
    n_cart_in_disk = int(cart_in_disk_mask.sum().item())
    wandb.config.update({
        "radial_rmax": radial_rmax,
        "n_cart_eval_points_in_disk": n_cart_in_disk,
        "cart_eval_in_disk_fraction": n_cart_in_disk / max(1, meta_cart["N"]),
    }, allow_val_change=True)
    with torch.no_grad():
        y_meas_denorm_hist = normalizer.denormalize(x_cart_dev, y_cart_dev)
        k_meas_hist = torch.complex(
            y_meas_denorm_hist[:, 0], y_meas_denorm_hist[:, 1]
        ).reshape(nky, nkx)
        k_meas_hist = k_meas_hist * cart_kspace_mask
        cart_ref_img = torch.fft.ifft2(k_meas_hist).abs().cpu().numpy().T
    cart_image_metrics_enabled = True

    def _compute_cartesian_image_history_metrics(cart_pred_norm: torch.Tensor) -> dict:
        cart_pred_denorm = normalizer.denormalize(x_cart_dev, cart_pred_norm)
        k_pred = torch.complex(
            cart_pred_denorm[:, 0], cart_pred_denorm[:, 1]
        ).reshape(nky, nkx)
        k_pred = k_pred * cart_kspace_mask
        img_pred = torch.fft.fftshift(torch.fft.ifft2(k_pred)).abs().cpu().numpy().T

        img_metrics = compute_image_metrics(img_pred, cart_ref_img)
        perc_metrics = compute_perceptual_metrics(img_pred, cart_ref_img)
        return {
            "train/cart_ref_ssim": img_metrics["ssim"],
            "train/cart_ref_dists": perc_metrics["DISTS"],
            "train/cart_ref_haarpsi": perc_metrics["HaarPSI"],
        }

    # plot metadata
    train_spoke_show = int(torch.unique(spoke_id_all[train_idx])[0].item())
    RO_total = int(ro_id_all.max().item()) + 1

    plot_steps = {1, steps}
    s = plot_every
    while s <= steps:
        plot_steps.add(s)
        s += plot_every

    # training loop
    model.train()
    best_heldout_loss = float("inf")
    best_step = -1
    best_cart_loss_in_disk = float("inf")
    best_state = None
    last_cart_loss = None
    last_cart_loss_in_disk = None
    last_heldout_loss = None

    logging.info(f"Training for {steps} steps on {N_train} radial points, "
                 f"eval on {meta_cart['N']} Cartesian points "
                 f"({n_cart_in_disk} in-disk)")

    diag_every = max(1, eval_every)
    for step in range(1, steps + 1):
        # train step
        idx = torch.randint(0, N_train, (batch_size,), device=device)
        x = x_train[idx]
        y = y_train[idx]
        w = dcf_train[idx]

        opt.zero_grad(set_to_none=True)
        y_pred = model(x)

        focal_progress = (
            min(1.0, step / float(focal_warmup_steps)) if focal_warmup_steps > 0 else 1.0
        )
        want_diag = (step % diag_every == 0 or step == steps or step == 1)
        if want_diag:
            loss, focal_diag = composable_kspace_loss(
                y_pred, y,
                dcf=w, use_dcf=use_dcf, dcf_power=dcf_power,
                use_focal=use_focal, focal_alpha=focal_alpha,
                focal_normalize=focal_normalize, focal_log_matrix=focal_log_matrix,
                focal_warmup_progress=focal_progress,
                return_diagnostics=True,
            )
        else:
            loss = composable_kspace_loss(
                y_pred, y,
                dcf=w, use_dcf=use_dcf, dcf_power=dcf_power,
                use_focal=use_focal, focal_alpha=focal_alpha,
                focal_normalize=focal_normalize, focal_log_matrix=focal_log_matrix,
                focal_warmup_progress=focal_progress,
                return_diagnostics=False,
            )
            focal_diag = None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        # scheduler step
        if scheduler is not None and scheduler_type in ("onecycle", "cosine"):
            scheduler.step()

        train_loss = float(loss.item())

        # eval
        if step % eval_every == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                cart_pred = model(x_cart_dev)
                last_cart_loss = float(F.mse_loss(cart_pred, y_cart_dev).item())
                last_cart_loss_in_disk = float(
                    F.mse_loss(cart_pred[cart_in_disk_mask], y_cart_dev[cart_in_disk_mask]).item()
                )
                if has_heldout:
                    heldout_pred = model(x_heldout)
                    last_heldout_loss = float(F.mse_loss(heldout_pred, y_heldout).item())
                else:
                    last_heldout_loss = None
            model.train()

            # post warmup checkpoint, heldout selection
            warmup_steps = config['training'].get('warmup_steps', steps // 5)
            if (step >= warmup_steps and last_heldout_loss is not None
                    and last_heldout_loss < best_heldout_loss):
                best_heldout_loss = last_heldout_loss
                best_step = step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if step >= warmup_steps:
                best_cart_loss_in_disk = min(best_cart_loss_in_disk, last_cart_loss_in_disk)

            if scheduler is not None and scheduler_type == "plateau":
                # prefer heldout spoke loss (same signal as best-state restore);
                # fall back to cart-eval-in-disk when subsample_frac=1.0 leaves no heldout
                sched_metric = last_heldout_loss if last_heldout_loss is not None else last_cart_loss_in_disk
                scheduler.step(sched_metric)

        # logging
        log_dict = {"train/train_loss": train_loss}
        if focal_diag is not None:
            for k, v in focal_diag.items():
                log_dict[f"train/{k}"] = v
            log_dict["train/focal_progress"] = focal_progress
            # low/high |k| residual split (using batch kcoords)
            with torch.no_grad():
                r_magsq = _residual_magsq(y_pred.detach(), y)
                kr_split = split_residual_norm_by_k(r_magsq, x[:, :2])
                for k_, v_ in kr_split.items():
                    log_dict[f"train/{k_}"] = v_
        if step % eval_every == 0 or step == steps:
            log_dict["train/cart_eval_loss"] = last_cart_loss
            log_dict["train/cart_eval_loss_full"] = last_cart_loss
            log_dict["train/cart_eval_loss_in_disk"] = last_cart_loss_in_disk
            if last_heldout_loss is not None:
                log_dict["train/heldout_spoke_loss"] = last_heldout_loss
            image_metric_every = config['training'].get('image_metric_every', 1000)
            if cart_image_metrics_enabled and (step % image_metric_every == 0 or step == steps):
                try:
                    log_dict.update(_compute_cartesian_image_history_metrics(cart_pred))
                except Exception as e:
                    cart_image_metrics_enabled = False
                    logging.warning(f"Disabling per-eval Cartesian image metrics: {e}")
        if scheduler is not None:
            log_dict["train/lr"] = opt.param_groups[0]["lr"]
        wandb_logger.log(log_dict, step=step)

        # figure logging
        if step in plot_steps:
            model.eval()
            with torch.no_grad():
                figures = {}

                # spoke plot
                fig = make_spoke_figure(
                    model,
                    x_all=x_all, y_all=y_all,
                    spoke_id_all=spoke_id_all,
                    ro_id_all=ro_id_all,
                    spoke_id=train_spoke_show,
                    y_scale=1.0,
                    n_s=4096,
                    title_prefix=f"[train] step {step}",
                    log_scale=log_scale,
                )
                figures["plots/spoke_train"] = fig

                # radial error map
                fig_err = make_error_map_figure(
                    model,
                    x_sub=x_all[train_idx],
                    y_sub=y_all[train_idx],
                    y_scale=1.0,
                    title_prefix=f"[radial train] step {step}",
                )
                figures["plots/error_map_radial_train"] = fig_err

                # cart error map
                fig_cart_err = make_cartesian_error_map(
                    model,
                    x_cart=x_cart, y_cart=y_cart,
                    y_scale=1.0,
                    nky=nky, nkx=nkx,
                    title_prefix=f"[cart eval] step {step}",
                )
                figures["plots/cart_error_map"] = fig_cart_err

                # cart image compare
                fig_cart_img = make_cartesian_image_comparison(
                    model,
                    x_cart=x_cart, y_cart=y_cart,
                    normalizer=normalizer,
                    nky=nky, nkx=nkx,
                    kspace_mask=cart_in_disk_mask,
                    ref_img_slice=ref_img_slice,
                    title_prefix=f"step {step}",
                )
                figures["plots/cart_image_comparison"] = fig_cart_img

                # radial nufft compare
                try:
                    from nik_recon import nufft2d_recon
                    _normalizer = normalizer
                    _x_rad = x_all[:, :2].to(device)
                    _y_pred_norm = model(_x_rad)
                    _y_pred_denorm = _normalizer.denormalize(_x_rad, _y_pred_norm)
                    _k_pred = torch.complex(_y_pred_denorm[:, 0], _y_pred_denorm[:, 1])
                    _k_pred_slice = _k_pred.reshape(data["n_ro_per_slice"], data["RO"])

                    _k_img_pred = torch.zeros_like(data["k_img_space"])
                    _t = config['data']['t_frame']
                    _c = config['data']['coil_idx']
                    _z = data["z_slice_idx"]
                    _k_img_pred[_t, :, _c, _z, :] = _k_pred_slice

                    img_rad_pred = nufft2d_recon(
                        _k_img_pred, data["traj_t"], t_frame=_t, coil_idx=_c,
                        z_slice_idx=_z, scales=data["scales"],
                        img_size=(312, 312), n_slices=data["n_z_slices"],
                    )
                    img_rad_meas = nufft2d_recon(
                        data["k_img_space"], data["traj_t"], t_frame=_t, coil_idx=_c,
                        z_slice_idx=_z, scales=data["scales"],
                        img_size=(312, 312), n_slices=data["n_z_slices"],
                    )

                    def _norm01(x):
                        mx = x.max()
                        return x / mx if mx > 0 else x

                    fig_rad, ax_rad = plt.subplots(1, 3, figsize=(18, 5))
                    ax_rad[0].imshow(_norm01(img_rad_meas), cmap="gray")
                    ax_rad[0].set_title(f"step {step} Radial NUFFT (measured)")
                    ax_rad[1].imshow(_norm01(img_rad_pred), cmap="gray")
                    ax_rad[1].set_title(f"step {step} Radial NUFFT (predicted)")
                    diff_rad = np.abs(_norm01(img_rad_pred) - _norm01(img_rad_meas))
                    ax_rad[2].imshow(diff_rad, cmap="hot")
                    ax_rad[2].set_title(f"step {step} |pred - meas|")
                    for ax in ax_rad:
                        ax.axis("off")
                    plt.tight_layout()
                    figures["plots/radial_image_comparison"] = fig_rad
                except Exception as e:
                    logging.warning(f"Radial NUFFT recon failed: {e}")

            wandb_logger.log_figures(figures, step=step)
            model.train()

        # console logging
        if step % console_every == 0:
            msg = f"step {step:6d}  train {train_loss:.3e}"
            if last_cart_loss is not None:
                msg += f"  cart_full {last_cart_loss:.3e}"
            if last_cart_loss_in_disk is not None:
                msg += f"  cart_disk {last_cart_loss_in_disk:.3e}"
            if last_heldout_loss is not None:
                msg += f"  heldout {last_heldout_loss:.3e}"
            logging.info(msg)

    # restore best, eval
    if best_state is not None:
        model.load_state_dict(best_state)
        logging.info(f"Loaded best model from step {best_step}, heldout_loss={best_heldout_loss:.4e}")
    else:
        logging.info("No best checkpoint, using final state")
    model.eval()

    # normalizer reuse
    final_cart_loss = None
    final_cart_loss_in_disk = None
    final_heldout_loss = None
    with torch.no_grad():
        cart_pred_eval = model(x_cart_dev)
        final_cart_loss = float(F.mse_loss(cart_pred_eval, y_cart_dev).item())
        final_cart_loss_in_disk = float(
            F.mse_loss(cart_pred_eval[cart_in_disk_mask], y_cart_dev[cart_in_disk_mask]).item()
        )
        if has_heldout:
            final_heldout_loss = float(F.mse_loss(model(x_heldout), y_heldout).item())

    try:
        with torch.no_grad():
            # final cart pred, denorm
            cart_pred_norm = model(x_cart_dev)
            cart_pred_denorm = normalizer.denormalize(x_cart[:, :2].to(device), cart_pred_norm)
            # reim shape
            k_pred = torch.complex(cart_pred_denorm[:, 0], cart_pred_denorm[:, 1]).reshape(nky, nkx)

            y_meas_denorm = normalizer.denormalize(x_cart[:, :2].to(device), y_cart_dev)
            k_meas = torch.complex(y_meas_denorm[:, 0], y_meas_denorm[:, 1]).reshape(nky, nkx)

            # disk mask
            k_pred = k_pred * cart_kspace_mask
            k_meas = k_meas * cart_kspace_mask

            # ifft to image
            img_pred = torch.fft.fftshift(torch.fft.ifft2(k_pred)).abs().cpu().numpy().T
            img_meas = torch.fft.ifft2(k_meas).abs().cpu().numpy().T

        # reference, fully sampled
        cart_ref_denorm = img_meas
        display_max = max(img_meas.max(), img_pred.max()) or 1.0

        # raw scale for nufft
        with torch.no_grad():
            k_meas_raw = torch.complex(y_cart_raw[:, 0], y_cart_raw[:, 1]).reshape(nky, nkx).to(device)
            k_meas_raw = k_meas_raw * cart_kspace_mask
            cart_ref_raw = torch.fft.ifft2(k_meas_raw).abs().cpu().numpy().T

        wandb_logger.log({
            "recon/cart_ref": wandb.Image(cart_ref_denorm / display_max),
            "recon/model_cart_pred": wandb.Image(img_pred / display_max),
        }, step=steps)

        # method 1, model vs ref
        img_pred_n = img_pred / (img_pred.max() or 1.0)
        cart_ref_denorm_n = cart_ref_denorm / (cart_ref_denorm.max() or 1.0)
        m1_metrics = compute_image_metrics(img_pred_n, cart_ref_denorm_n)
        m1_perceptual = compute_perceptual_metrics(img_pred_n, cart_ref_denorm_n)
        wandb_logger.log({
            "model_vs_ref/psnr": m1_metrics["psnr_db"],
            "model_vs_ref/ssim": m1_metrics["ssim"],
            "model_vs_ref/nrmse": m1_metrics["nrmse"],
            "model_vs_ref/dists": m1_perceptual["DISTS"],
            "model_vs_ref/haarpsi": m1_perceptual["HaarPSI"],
            "model_vs_ref/vsi": m1_perceptual["VSI"],
        }, step=steps)
        wandb.run.summary.update({
            "model_psnr": m1_metrics["psnr_db"],
            "model_ssim": m1_metrics["ssim"],
            "model_nrmse": m1_metrics["nrmse"],
            "model_dists": m1_perceptual["DISTS"],
            "model_haarpsi": m1_perceptual["HaarPSI"],
            "model_vsi": m1_perceptual["VSI"],
        })
        logging.info(
            f"Model vs Ref:  PSNR={m1_metrics['psnr_db']:.2f}  SSIM={m1_metrics['ssim']:.4f}  "
            f"DISTS={m1_perceptual['DISTS']:.4f}  HaarPSI={m1_perceptual['HaarPSI']:.4f}  "
            f"VSI={m1_perceptual['VSI']:.4f}"
        )

        # method 2, nufft baseline
        from nik_recon import nufft2d_recon
        _t = config['data']['t_frame']
        _c = config['data']['coil_idx']
        _z = data["z_slice_idx"]

        # train only kspace
        k_img_sub = torch.zeros_like(data["k_img_space"])
        # spoke ids, not points
        train_spoke_ids = torch.unique(spoke_id_all[train_idx])
        # copy train spokes
        for sp_id in train_spoke_ids:
            k_img_sub[:, sp_id, :, :, :] = data["k_img_space"][:, sp_id, :, :, :]

        img_nufft_sub = nufft2d_recon(
            k_img_sub, data["traj_t"], t_frame=_t, coil_idx=_c,
            z_slice_idx=_z, scales=data["scales"],
            img_size=(312, 312), n_slices=data["n_z_slices"],
        )

        # crop to cart fov
        gt_h, gt_w = cart_ref_raw.shape
        nufft_h, nufft_w = img_nufft_sub.shape
        if nufft_h != gt_h or nufft_w != gt_w:
            # center crop
            y0 = (nufft_h - gt_h) // 2
            x0 = (nufft_w - gt_w) // 2
            img_nufft_crop = img_nufft_sub[y0:y0+gt_h, x0:x0+gt_w]
        else:
            img_nufft_crop = img_nufft_sub

        # peak normalize
        img_nufft_n = img_nufft_crop / (img_nufft_crop.max() or 1.0)
        cart_ref_raw_n = cart_ref_raw / (cart_ref_raw.max() or 1.0)

        wandb_logger.log({
            "recon/nufft_subsampled": wandb.Image(img_nufft_n),
        }, step=steps)

        m2_metrics = compute_image_metrics(img_nufft_n, cart_ref_raw_n)
        m2_perceptual = compute_perceptual_metrics(img_nufft_n, cart_ref_raw_n)
        wandb_logger.log({
            "nufft_vs_ref/psnr": m2_metrics["psnr_db"],
            "nufft_vs_ref/ssim": m2_metrics["ssim"],
            "nufft_vs_ref/nrmse": m2_metrics["nrmse"],
            "nufft_vs_ref/dists": m2_perceptual["DISTS"],
            "nufft_vs_ref/haarpsi": m2_perceptual["HaarPSI"],
            "nufft_vs_ref/vsi": m2_perceptual["VSI"],
        }, step=steps)
        wandb.run.summary.update({
            "nufft_psnr": m2_metrics["psnr_db"],
            "nufft_ssim": m2_metrics["ssim"],
            "nufft_nrmse": m2_metrics["nrmse"],
            "nufft_dists": m2_perceptual["DISTS"],
            "nufft_haarpsi": m2_perceptual["HaarPSI"],
            "nufft_vsi": m2_perceptual["VSI"],
        })
        logging.info(
            f"NUFFT vs Ref:  PSNR={m2_metrics['psnr_db']:.2f}  SSIM={m2_metrics['ssim']:.4f}  "
            f"DISTS={m2_perceptual['DISTS']:.4f}  HaarPSI={m2_perceptual['HaarPSI']:.4f}  "
            f"VSI={m2_perceptual['VSI']:.4f}"
        )

        # delta vs nufft
        delta_psnr = m1_metrics["psnr_db"] - m2_metrics["psnr_db"]
        delta_dists = m2_perceptual["DISTS"] - m1_perceptual["DISTS"]  # positive = model better
        delta_haarpsi = m1_perceptual["HaarPSI"] - m2_perceptual["HaarPSI"]  # positive = model better
        wandb.run.summary.update({
            "delta_psnr": delta_psnr,
            "delta_dists": delta_dists,
            "delta_haarpsi": delta_haarpsi,
        })
        logging.info(
            f"Model vs NUFFT delta:  dPSNR={delta_psnr:+.2f} dB  "
            f"dDISTS={delta_dists:+.4f}  dHaarPSI={delta_haarpsi:+.4f}"
        )
    except Exception as e:
        logging.warning(f"Final Cartesian evaluation failed: {e}")
        import traceback
        traceback.print_exc()

    # final radial nufft
    try:
        from nik_recon import nufft2d_recon
        _x_rad = x_all[:, :2].to(device)
        with torch.no_grad():
            _y_pred_norm = model(_x_rad)
            _y_pred_denorm = normalizer.denormalize(_x_rad, _y_pred_norm)
        _k_pred = torch.complex(_y_pred_denorm[:, 0], _y_pred_denorm[:, 1])
        _k_pred_slice = _k_pred.reshape(data["n_ro_per_slice"], data["RO"])

        _k_img_pred = torch.zeros_like(data["k_img_space"])
        _t = config['data']['t_frame']
        _c = config['data']['coil_idx']
        _z = data["z_slice_idx"]
        _k_img_pred[_t, :, _c, _z, :] = _k_pred_slice

        img_rad_pred = nufft2d_recon(
            _k_img_pred, data["traj_t"], t_frame=_t, coil_idx=_c,
            z_slice_idx=_z, scales=data["scales"],
            img_size=(312, 312), n_slices=data["n_z_slices"],
        )
        img_rad_meas = nufft2d_recon(
            data["k_img_space"], data["traj_t"], t_frame=_t, coil_idx=_c,
            z_slice_idx=_z, scales=data["scales"],
            img_size=(312, 312), n_slices=data["n_z_slices"],
        )
        wandb_logger.log({
            "recon/radial_predicted": wandb.Image(img_rad_pred / (img_rad_pred.max() or 1.0)),
            "recon/radial_measured": wandb.Image(img_rad_meas / (img_rad_meas.max() or 1.0)),
        }, step=steps)
        rad_metrics = compute_image_metrics(img_rad_pred, img_rad_meas)
        rad_perceptual = compute_perceptual_metrics(img_rad_pred, img_rad_meas)
        wandb_logger.log({
            "metrics/psnr_radial": rad_metrics["psnr_db"],
            "metrics/ssim_radial": rad_metrics["ssim"],
            "metrics/nrmse_radial": rad_metrics["nrmse"],
            "metrics/dists_radial": rad_perceptual["DISTS"],
            "metrics/haarpsi_radial": rad_perceptual["HaarPSI"],
            "metrics/vsi_radial": rad_perceptual["VSI"],
        }, step=steps)
        wandb.run.summary.update({
            "psnr_radial": rad_metrics["psnr_db"],
            "ssim_radial": rad_metrics["ssim"],
            "nrmse_radial": rad_metrics["nrmse"],
            "dists_radial": rad_perceptual["DISTS"],
            "haarpsi_radial": rad_perceptual["HaarPSI"],
            "vsi_radial": rad_perceptual["VSI"],
        })
        logging.info(
            f"Radial metrics: PSNR={rad_metrics['psnr_db']:.2f} dB  "
            f"SSIM={rad_metrics['ssim']:.4f}  NRMSE={rad_metrics['nrmse']:.4f}  "
            f"DISTS={rad_perceptual['DISTS']:.4f}  HaarPSI={rad_perceptual['HaarPSI']:.4f}  "
            f"VSI={rad_perceptual['VSI']:.4f}"
        )
    except Exception as e:
        logging.warning(f"Radial NUFFT evaluation failed: {e}")
        import traceback
        traceback.print_exc()

    # save model
    wandb_logger.save_model(model, "model_best.pth", opt, steps, output_dir)

    wandb.run.summary["best_heldout_spoke_loss"] = best_heldout_loss
    wandb.run.summary["best_step"] = best_step
    wandb.run.summary["best_cart_eval_loss_in_disk"] = best_cart_loss_in_disk
    wandb.run.summary["final_cart_eval_loss"] = final_cart_loss
    wandb.run.summary["final_cart_eval_loss_in_disk"] = final_cart_loss_in_disk
    if final_heldout_loss is not None:
        wandb.run.summary["final_heldout_spoke_loss"] = final_heldout_loss
    wandb.run.summary["final_train_loss"] = train_loss
    wandb.run.summary["total_steps"] = steps

    logging.info(
        f"Done. best_heldout_spoke_loss={best_heldout_loss:.3e}  "
        f"best_cart_eval_loss_in_disk={best_cart_loss_in_disk:.3e}"
    )
    wandb_logger.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NIK DCE-MRI: radial train, Cartesian eval."
    )
    parser.add_argument('config_path', type=str,
                        help='Path to the configuration TOML file.')
    parser.add_argument('--sweep-id', type=str, default=None,
                        help='Existing wandb sweep ID to join.')
    parser.add_argument('--count', type=int, default=50,
                        help='Number of sweep runs.')
    parser.add_argument('--single', action='store_true',
                        help='Run a single training run (no sweep).')
    args = parser.parse_args()

    config = load_config(args.config_path)
    data = load_data(config)

    if args.single:
        main(args.config_path, data)
    elif args.sweep_id:
        wandb.agent(
            args.sweep_id,
            function=lambda: main(args.config_path, data),
            count=args.count,
        )
    else:
        sweep_config = config['sweep']
        sweep_id = wandb.sweep(sweep=sweep_config)
        wandb.agent(
            sweep_id,
            function=lambda: main(args.config_path, data),
            count=args.count,
        )
