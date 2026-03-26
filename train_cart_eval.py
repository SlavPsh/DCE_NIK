#!/usr/bin/env python
"""
train_cart_eval.py -- Train on radial k-space, evaluate on Cartesian grid.

Usage:
    # Single run:
    python train_cart_eval.py config/training_cart_eval.toml --single

    # Create new sweep and run agent:
    python train_cart_eval.py config/training_cart_eval.toml

    # Join existing sweep:
    python train_cart_eval.py config/training_cart_eval.toml --sweep-id ENTITY/PROJECT/ID --count 50
"""
import argparse
import random
import logging
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from coolname import generate_slug

from utils.io_utils import (
    load_config, setup_logging, unique_output_dir, copy_config_to_output,
)
from utils.wandb_utils import WandbLogger

from nik_io import load_event, load_cartesian_kspace
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
from nik_loss import get_loss_fn
from nik_train import prepare_tensors
from nik_recon import (
    ifft1d_kz_to_z,
    make_fixed_frame_zslice_coil_dataset,
    split_points_by_spokes,
    ifft1d_kz_to_z_cartesian,
    make_cartesian_eval_dataset,
)
from nik_metrics import compute_image_metrics
from wandb_logger import (
    make_spoke_figure,
    make_error_map_figure,
    make_cartesian_error_map,
    make_cartesian_image_comparison,
)


def load_data(config):
    """Load radial training data and Cartesian evaluation data."""
    data_cfg = config['data']
    radial_file = data_cfg['radial_file']
    cart_file = data_cfg['cartesian_file']
    t_frame = data_cfg['t_frame']
    coil_idx = data_cfg['coil_idx']
    z_slice_raw = data_cfg['z_slice_idx']
    subsample_frac = data_cfg.get('subsample_frac', 1.0)
    seed = config['training']['seed']

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Load radial data ----
    print(f"Loading radial data from {radial_file} ...", flush=True)
    event = load_event(radial_file, load_images=True, load_coil_maps=True)
    k_np, traj_np = event["k"], event["traj"]
    gt_img = event.get("gt_img")
    coil_maps_radial = event.get("coil_maps")

    # Transpose to (T, S, C, RO)
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

    x_all, y_all, kx_all, ky_all, spoke_id_all, ro_id_all, meta = \
        make_fixed_frame_zslice_coil_dataset(
            k_img_space, traj_t, scales, dims,
            y_scale=k_scale,
            t_fixed=t_frame,
            coil_fixed=coil_idx,
            z_slice_idx=z_slice_idx,
            n_slices=n_z_slices,
            compute_device=device,
        )
    print(f"Radial dataset: {meta['N']} points", flush=True)

    # ---- Subsample spokes ----
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

    # ---- Load Cartesian data ----
    print(f"Loading Cartesian data from {cart_file} ...", flush=True)
    cart_event = load_cartesian_kspace(cart_file, load_images=True, load_coil_maps=True)
    k_cart_np = cart_event["k_cart"]
    coil_maps_cart = cart_event.get("coil_maps")
    cart_meta = cart_event["meta"]
    cart_gt_img = cart_event.get("gt_img")

    k_cart_t = torch.from_numpy(k_cart_np.astype(np.complex64))
    if device == "cuda":
        k_cart_t = k_cart_t.cuda()
    print(f"Cartesian k-space shape: {k_cart_t.shape}", flush=True)

    # kz->z IFFT for Cartesian
    k_cart_z = ifft1d_kz_to_z_cartesian(k_cart_t)
    nz_cart = k_cart_z.shape[2]

    # Use same z_slice_idx (middle by default)
    z_slice_cart = nz_cart // 2 if z_slice_raw == -1 else int(z_slice_raw)

    x_cart, y_cart, meta_cart = make_cartesian_eval_dataset(
        k_cart_z,
        t_fixed=min(t_frame, k_cart_z.shape[0] - 1),
        coil_fixed=coil_idx,
        z_slice_idx=z_slice_cart,
        scales_radial=scales,
        y_scale=k_scale,
        compute_device=device,
    )
    print(f"Cartesian eval dataset: {meta_cart['N']} points "
          f"(nky={meta_cart['nky']}, nkx={meta_cart['nkx']})", flush=True)

    # Sanity check: compare k-space magnitude ranges
    rad_mag = float(torch.abs(y_all).max().item()) * float(k_scale.item() if torch.is_tensor(k_scale) else k_scale)
    cart_mag = float(torch.abs(y_cart).max().item()) * float(k_scale.item() if torch.is_tensor(k_scale) else k_scale)
    print(f"K-space magnitude sanity check: radial_max={rad_mag:.2e}, cart_max={cart_mag:.2e}, "
          f"ratio={cart_mag/(rad_mag+1e-12):.2f}", flush=True)

    # Get GT image slice for comparison
    gt_img_slice = None
    if cart_gt_img is not None:
        t_cart = min(t_frame, cart_gt_img.shape[0] - 1)
        gt_img_slice = cart_gt_img[t_cart, z_slice_cart, :, :]

    return {
        # Radial training data
        "x_all": x_all, "y_all": y_all,
        "spoke_id_all": spoke_id_all, "ro_id_all": ro_id_all,
        "train_idx": train_idx,
        "meta": meta, "k_scale": k_scale,
        "n_ro_per_slice": n_ro_per_slice,
        "T": T, "S": S, "C": C, "RO": RO,
        "z_slice_idx": z_slice_idx,
        "n_z_slices": n_z_slices,
        "scales": scales, "dims": dims,
        "coil_maps_radial": coil_maps_radial,
        # Cartesian eval data
        "x_cart": x_cart, "y_cart": y_cart,
        "meta_cart": meta_cart,
        "gt_img_slice": gt_img_slice,
        "coil_maps_cart": coil_maps_cart,
        # Shared
        "gt_img": gt_img,
        "subsample_frac": subsample_frac,
        "n_unique_spokes": n_unique_spokes,
        "n_train_spokes": n_train_spokes,
    }


def main(config_path, data):
    """Single training run: radial train, Cartesian eval."""
    random.seed()
    run_name = generate_slug(3) + "_carteval"
    config = load_config(config_path)
    output_dir = unique_output_dir(config, run_name)
    copy_config_to_output(config_path, output_dir)
    setup_logging(config, output_dir)

    logging.info(f"Run: {run_name}")

    # Build flat training config for wandb
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

    # Resolve hyperparams (sweep overrides)
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

    # Reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Unpack data ----
    x_all = data["x_all"]
    y_all = data["y_all"]
    spoke_id_all = data["spoke_id_all"]
    ro_id_all = data["ro_id_all"]
    train_idx = data["train_idx"]
    k_scale = data["k_scale"]
    meta_cart = data["meta_cart"]
    x_cart = data["x_cart"]
    y_cart = data["y_cart"]
    nky, nkx = meta_cart["nky"], meta_cart["nkx"]
    gt_img_slice = data.get("gt_img_slice")

    # If sweep overrides subsample_frac, re-subsample
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

    wandb.config.update({
        "n_train_points": int(train_idx.shape[0]),
        "n_cart_eval_points": meta_cart["N"],
        "nky": nky, "nkx": nkx,
        "subsample_frac_actual": subsample_frac,
    }, allow_val_change=True)

    # ---- Compute s_max for polar models ----
    _x_tmp = x_all[:, :2].to(device)
    _kx, _ky = _x_tmp[:, 0], _x_tmp[:, 1]
    _theta = torch.atan2(_ky, _kx)
    _theta0 = torch.remainder(_theta + 0.5 * np.pi, np.pi) - 0.5 * np.pi
    _s_coord = _kx * torch.cos(_theta0) + _ky * torch.sin(_theta0)
    s_max = float(_s_coord.abs().max().item())
    del _x_tmp, _kx, _ky, _theta, _theta0, _s_coord

    # ---- Build model ----
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
        model = WIRE_KXY_REIM(in_dim=2, hidden=hidden, depth=depth, w0=w0, s0=s0).to(device)
    else:
        model = NIK_SIREN_KXY_REIM(in_dim=2, hidden=hidden, depth=depth, w0=w0).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    wandb.config.update({"n_params": n_params, "model_family": model_family}, allow_val_change=True)
    logging.info(f"Model: {model_family}, hidden={hidden}, depth={depth}, params={n_params}")

    # ---- Optimizer ----
    if optimizer_name == "Adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "SGD":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    scheduler = None
    if scheduler_patience > 0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=scheduler_factor,
            patience=scheduler_patience, min_lr=scheduler_min_lr,
        )

    # ---- Prepare training tensors ----
    x_all_2d = x_all[:, :2].to(device)
    y_all_dev = y_all.to(device)
    x_train = x_all_2d[train_idx]
    y_train = y_all_dev[train_idx]
    N_train = x_train.shape[0]

    # Cartesian eval tensors (already on device from load_data)
    x_cart_dev = x_cart.to(device)
    y_cart_dev = y_cart.to(device)

    # ---- Plot metadata ----
    train_spoke_show = int(torch.unique(spoke_id_all[train_idx])[0].item())
    RO_total = int(ro_id_all.max().item()) + 1

    plot_steps = {1, steps}
    s = plot_every
    while s <= steps:
        plot_steps.add(s)
        s += plot_every

    # ---- Training loop ----
    model.train()
    best_cart_loss = float("inf")
    best_state = None
    last_cart_loss = None

    logging.info(f"Training for {steps} steps on {N_train} radial points, "
                 f"eval on {meta_cart['N']} Cartesian points")

    for step in range(1, steps + 1):
        # Training step
        idx = torch.randint(0, N_train, (batch_size,), device=device)
        x = x_train[idx]
        y = y_train[idx]

        opt.zero_grad(set_to_none=True)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        train_loss = float(loss.item())

        # Cartesian evaluation
        if step % eval_every == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                cart_pred = model(x_cart_dev)
                last_cart_loss = float(F.mse_loss(cart_pred, y_cart_dev).item())
            model.train()

            if last_cart_loss < best_cart_loss:
                best_cart_loss = last_cart_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            if scheduler is not None:
                scheduler.step(last_cart_loss)

        # Logging
        log_dict = {"train/train_loss": train_loss}
        if last_cart_loss is not None and (step % eval_every == 0 or step == steps):
            log_dict["train/cart_eval_loss"] = last_cart_loss
        if scheduler is not None:
            log_dict["train/lr"] = opt.param_groups[0]["lr"]
        wandb_logger.log(log_dict, step=step)

        # Figure logging
        if step in plot_steps:
            model.eval()
            with torch.no_grad():
                figures = {}

                # Spoke plot (training data)
                fig = make_spoke_figure(
                    model,
                    x_all=x_all, y_all=y_all,
                    spoke_id_all=spoke_id_all,
                    ro_id_all=ro_id_all,
                    spoke_id=train_spoke_show,
                    y_scale=k_scale,
                    n_s=4096,
                    title_prefix=f"[train] step {step}",
                    log_scale=log_scale,
                )
                figures["plots/spoke_train"] = fig

                # Radial error map (training points)
                fig_err = make_error_map_figure(
                    model,
                    x_sub=x_all[train_idx],
                    y_sub=y_all[train_idx],
                    y_scale=k_scale,
                    title_prefix=f"[radial train] step {step}",
                )
                figures["plots/error_map_radial_train"] = fig_err

                # Cartesian error map
                fig_cart_err = make_cartesian_error_map(
                    model,
                    x_cart=x_cart, y_cart=y_cart,
                    y_scale=k_scale,
                    nky=nky, nkx=nkx,
                    title_prefix=f"[cart eval] step {step}",
                )
                figures["plots/cart_error_map"] = fig_cart_err

                # Cartesian image comparison
                fig_cart_img = make_cartesian_image_comparison(
                    model,
                    x_cart=x_cart, y_cart=y_cart,
                    y_scale=k_scale,
                    nky=nky, nkx=nkx,
                    gt_img_slice=gt_img_slice,
                    title_prefix=f"step {step}",
                )
                figures["plots/cart_image_comparison"] = fig_cart_img

            wandb_logger.log_figures(figures, step=step)
            model.train()

        # Console logging
        if step % console_every == 0:
            msg = f"step {step:6d}  train {train_loss:.3e}"
            if last_cart_loss is not None:
                msg += f"  cart_eval {last_cart_loss:.3e}"
            logging.info(msg)

    # ---- Restore best model and final evaluation ----
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    try:
        with torch.no_grad():
            # Final Cartesian k-space prediction
            ys = float(k_scale.item()) if torch.is_tensor(k_scale) else float(k_scale)
            cart_pred_final = model(x_cart_dev) * ys
            k_pred = torch.complex(cart_pred_final[:, 0], cart_pred_final[:, 1]).reshape(nky, nkx)

            y_meas_scaled = y_cart_dev * ys
            k_meas = torch.complex(y_meas_scaled[:, 0], y_meas_scaled[:, 1]).reshape(nky, nkx)

            # IFFT to image
            img_pred = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(k_pred))).abs().cpu().numpy()
            img_meas = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(k_meas))).abs().cpu().numpy()

        wandb_logger.log({
            "recon/cart_predicted": wandb.Image(img_pred),
            "recon/cart_measured": wandb.Image(img_meas),
        }, step=steps)

        metrics = compute_image_metrics(img_pred, img_meas)
        wandb_logger.log({
            "metrics/psnr_cart": metrics["psnr_db"],
            "metrics/ssim_cart": metrics["ssim"],
            "metrics/nrmse_cart": metrics["nrmse"],
        }, step=steps)
        wandb.run.summary.update({
            "psnr_cart": metrics["psnr_db"],
            "ssim_cart": metrics["ssim"],
            "nrmse_cart": metrics["nrmse"],
        })
        logging.info(
            f"Cart metrics:  PSNR={metrics['psnr_db']:.2f} dB  "
            f"SSIM={metrics['ssim']:.4f}  NRMSE={metrics['nrmse']:.4f}"
        )

        if gt_img_slice is not None:
            gt_slice = np.asarray(gt_img_slice, dtype=np.float64)
            if img_pred.shape != gt_slice.shape:
                from scipy.ndimage import zoom
                zf = (gt_slice.shape[0] / img_pred.shape[0], gt_slice.shape[1] / img_pred.shape[1])
                img_pred_r = zoom(img_pred, zf, order=3)
            else:
                img_pred_r = img_pred
            metrics_gt = compute_image_metrics(img_pred_r, gt_slice)
            wandb_logger.log({
                "metrics/psnr_vs_gt": metrics_gt["psnr_db"],
                "metrics/ssim_vs_gt": metrics_gt["ssim"],
                "metrics/nrmse_vs_gt": metrics_gt["nrmse"],
            }, step=steps)
            wandb.run.summary.update({
                "psnr_vs_gt": metrics_gt["psnr_db"],
                "ssim_vs_gt": metrics_gt["ssim"],
                "nrmse_vs_gt": metrics_gt["nrmse"],
            })
            logging.info(
                f"GT metrics:    PSNR={metrics_gt['psnr_db']:.2f} dB  "
                f"SSIM={metrics_gt['ssim']:.4f}  NRMSE={metrics_gt['nrmse']:.4f}"
            )
    except Exception as e:
        logging.warning(f"Final evaluation failed: {e}")
        import traceback
        traceback.print_exc()

    # ---- Save model ----
    wandb_logger.save_model(model, "model_best.pth", opt, steps, output_dir)

    wandb.run.summary["cart_eval_loss"] = best_cart_loss
    wandb.run.summary["final_train_loss"] = train_loss
    wandb.run.summary["total_steps"] = steps

    logging.info(f"Done. best_cart_eval_loss={best_cart_loss:.3e}")
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
