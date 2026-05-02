#!/usr/bin/env python
"""nik wandb trainer, sweeps"""
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

from nik_io import load_event
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
from nik_loss import (
    get_loss_fn,
    density_weighted_mse_loss,
    dc_consistency_loss,
    conjugate_symmetry_loss_from_model,
)
from nik_train import prepare_tensors
from nik_recon import (
    ifft1d_kz_to_z,
    make_fixed_frame_zslice_coil_dataset,
    split_points_by_spokes,
    nufft2d_recon,
)
from nik_metrics import compute_image_metrics
from wandb_logger import (
    make_spoke_figure,
    make_ring_figures,
    make_error_map_figure,
)


def load_data(config):
    """data once, sweep reusable"""
    data_cfg = config['data']
    file_path = data_cfg['file']
    t_frame = data_cfg['t_frame']
    coil_idx = data_cfg['coil_idx']
    z_slice_raw = data_cfg['z_slice_idx']
    val_frac = data_cfg['val_frac']
    seed = config['training']['seed']

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading data from {file_path} ...", flush=True)
    event = load_event(file_path, load_images=True)
    k_np, traj_np = event["k"], event["traj"]
    gt_img = event.get("gt_img")

    # tscro layout
    k_np = np.transpose(k_np, (0, 2, 1, 3))
    traj_np = np.transpose(traj_np, (0, 2, 1, 3))
    T, S, C, RO = k_np.shape
    print(f"k-space shape: T={T}, S={S}, C={C}, RO={RO}", flush=True)

    k_t, traj_t, scales, dims, k_scale = prepare_tensors(
        k_np, traj_np, data_device="cuda" if device == "cuda" else "cpu"
    )

    k_img_space, n_z_slices, n_ro_per_slice, kz_sort_order = ifft1d_kz_to_z(
        k_t, traj_t, t_frame=t_frame
    )
    print(f"z slices: {n_z_slices}, readouts/slice: {n_ro_per_slice}", flush=True)

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
    print(f"Dataset: {meta['N']} points", flush=True)

    train_idx, val_idx, train_spokes, val_spokes = split_points_by_spokes(
        spoke_id_all, val_frac=val_frac, seed=seed, mode="random",
    )

    img_size = gt_img.shape[2:4] if gt_img is not None else (128, 128)

    return {
        "k_img_space": k_img_space, "traj_t": traj_t, "scales": scales,
        "dims": dims, "k_scale": k_scale,
        "x_all": x_all, "y_all": y_all,
        "spoke_id_all": spoke_id_all, "ro_id_all": ro_id_all,
        "train_idx": train_idx, "val_idx": val_idx,
        "meta": meta, "n_z_slices": n_z_slices,
        "n_ro_per_slice": n_ro_per_slice,
        "T": T, "S": S, "C": C, "RO": RO,
        "z_slice_idx": z_slice_idx, "img_size": img_size,
        "gt_img": gt_img,
    }


def main(config_path, data):
    """single run, wandb logged"""
    random.seed()  # reset to OS entropy (undo previous run's deterministic seed)
    run_name = generate_slug(3) + "_nik"
    config = load_config(config_path)
    output_dir = unique_output_dir(config, run_name)
    copy_config_to_output(config_path, output_dir)
    setup_logging(config, output_dir)

    logging.info(f"Run: {run_name}")
    logging.info(f"Config: {config_path}")
    logging.info(f"Output: {output_dir}")
    for key, value in config.items():
        logging.info(f"  {key}: {value}")

    # flat config, sweep override
    train_config = {
        "model_family": config['model'].get('model_family', 'siren'),
        "hidden": config['model']['hidden'],
        "depth": config['model']['depth'],
        "w0": config['model'].get('w0', 15),
        "k_freq": config['model'].get('k_freq', 64),
        "k_sigma": config['model'].get('k_sigma', 6.0),
        "s0": config['model'].get('s0', 10.0),
        # polar
        "n_angular_modes": config['model'].get('n_angular_modes', 16),
        "radial_type": config['model'].get('radial_type', 'wire'),
        "dropout": config['model'].get('dropout', 0.0),
        # constraint weights
        "dc_weight": config['training'].get('dc_weight', 0.0),
        "density_weight": config['training'].get('density_weight', 0.0),
        "conj_weight": config['training'].get('conj_weight', 0.0),
        # standard
        "optimizer": config['training']['optimizer'],
        "lr": config['training']['lr'],
        "batch_size": config['training']['batch_size'],
        "steps": config['training']['steps'],
        "eval_every": config['training']['eval_every'],
        "grad_clip": config['training']['grad_clip'],
        "weight_decay": config['training'].get('weight_decay', 0.0),
        "loss_type": config['training'].get('loss_type', "mse"),
        "seed": config['training']['seed'],
        # lr scheduler
        "scheduler_patience": config['training'].get('scheduler_patience', 0),
        "scheduler_factor": config['training'].get('scheduler_factor', 0.5),
        "scheduler_min_lr": config['training'].get('scheduler_min_lr', 1e-6),
    }

    # wandb init
    wandb_logger = WandbLogger(
        config=train_config,
        output_dir=output_dir,
        run_name=run_name,
        job_type="training",
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
    dc_weight = float(getattr(wc, "dc_weight", 0.0))
    density_weight = float(getattr(wc, "density_weight", 0.0))
    conj_weight = float(getattr(wc, "conj_weight", 0.0))
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
    scheduler_patience = int(getattr(wc, "scheduler_patience", 0))
    scheduler_factor = float(getattr(wc, "scheduler_factor", 0.5))
    scheduler_min_lr = float(getattr(wc, "scheduler_min_lr", 1e-6))

    # skip redundant combos
    def _skip(reason: str):
        logging.info(f"Skipping: {reason}")
        wandb.run.summary["skipped"] = True
        wandb.run.summary["best_val_loss"] = float("inf")
        wandb_logger.finish()

    # family relevance
    if model_family in ("relu", "elu") and w0 != 15:
        _skip(f"w0={w0} is irrelevant for model_family={model_family}")
        return
    if model_family in ("relu", "elu", "siren") and (k_sigma != 6.0 or k_freq != 64):
        _skip(f"k_sigma={k_sigma}/k_freq={k_freq} irrelevant for model_family={model_family}")
        return
    if model_family in ("ff_relu", "ff_elu") and w0 != 15:
        _skip(f"w0={w0} is irrelevant for model_family={model_family}")
        return
    if model_family == "polar" and (k_sigma != 6.0 or k_freq != 64):
        _skip(f"k_sigma/k_freq irrelevant for model_family=polar")
        return

    # family valid ranges
    family_params = config.get("family_params", {}).get(model_family, {})
    valid_hidden = family_params.get("hidden")
    valid_depth = family_params.get("depth")
    if valid_hidden and hidden not in valid_hidden:
        _skip(f"hidden={hidden} not in valid set {valid_hidden} for {model_family}")
        return
    if valid_depth and depth not in valid_depth:
        _skip(f"depth={depth} not in valid set {valid_depth} for {model_family}")
        return

    plot_every = config['training']['plot']['plot_every']
    log_scale = config['training']['plot']['log_scale']
    console_every = config['logging']['console_every']

    # reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # unpack data
    k_img_space = data["k_img_space"]
    traj_t = data["traj_t"]
    scales = data["scales"]
    k_scale = data["k_scale"]
    x_all = data["x_all"]
    y_all = data["y_all"]
    spoke_id_all = data["spoke_id_all"]
    ro_id_all = data["ro_id_all"]
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    meta = data["meta"]
    n_z_slices = data["n_z_slices"]
    n_ro_per_slice = data["n_ro_per_slice"]
    T, S, C, RO = data["T"], data["S"], data["C"], data["RO"]
    z_slice_idx = data["z_slice_idx"]
    img_size = data["img_size"]
    t_frame = config['data']['t_frame']
    coil_idx = config['data']['coil_idx']

    wandb.config.update({
        "T": T, "S": S, "C": C, "RO": RO,
        "n_z_slices": n_z_slices,
        "n_ro_per_slice": n_ro_per_slice,
        "z_slice_idx": z_slice_idx,
    }, allow_val_change=True)

    x_val = x_all[val_idx][:, :2].to(device)
    y_val = y_all[val_idx].to(device)

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
        model = ReLU_MLP_KXY_REIM(
            in_dim=2, hidden=hidden, depth=depth,
        ).to(device)
    elif model_family == "elu":
        model = ELU_MLP_KXY_REIM(
            in_dim=2, hidden=hidden, depth=depth,
        ).to(device)
    elif model_family == "ff_relu":
        model = FF_ReLU_MLP_KXY_REIM(
            in_dim=2, k_freq=k_freq, k_sigma=k_sigma,
            hidden=hidden, depth=depth,
        ).to(device)
    elif model_family == "ff_elu":
        model = FF_ELU_MLP_KXY_REIM(
            in_dim=2, k_freq=k_freq, k_sigma=k_sigma,
            hidden=hidden, depth=depth,
        ).to(device)
    elif model_family == "ff_siren":
        model = NIK_SIREN_KXY_FF_REIM(
            x_dim=2, k_freq=k_freq, k_sigma=k_sigma,
            hidden=hidden, depth=depth, w0=w0,
        ).to(device)
    elif model_family == "polar":
        model = PolarKSpaceNet(
            n_angular_modes=n_angular_modes,
            radial_depth=depth,
            radial_width=hidden,
            radial_type=radial_type,
            omega_0=w0,
            s_0=s0,
            s_max=s_max,
        ).to(device)
    elif model_family == "wire":
        model = WIRE_KXY_REIM(
            in_dim=2, hidden=hidden, depth=depth, w0=w0, s0=s0, dropout=dropout,
        ).to(device)
    else:
        model = NIK_SIREN_KXY_REIM(
            in_dim=2, hidden=hidden, depth=depth, w0=w0,
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    wandb_extra = {
        "model_family": model_family, "n_params": n_params,
        "k_freq": k_freq, "k_sigma": k_sigma,
        "dc_weight": dc_weight, "density_weight": density_weight,
        "conj_weight": conj_weight,
    }
    if model_family == "polar":
        wandb_extra.update({
            "n_angular_modes": n_angular_modes, "radial_type": radial_type,
            "s_max": s_max,
        })
    wandb.config.update(wandb_extra, allow_val_change=True)
    logging.info(
        f"Model: family={model_family}, hidden={hidden}, depth={depth}, "
        f"w0={w0}, s0={s0 if model_family in ('wire', 'polar') else 'n/a'}, "
        f"k_freq={k_freq}, k_sigma={k_sigma}, wd={weight_decay}, loss={loss_type}, params={n_params}"
    )
    if conj_weight > 0 or density_weight > 0:
        logging.info(
            f"  Constraints: conj_weight={conj_weight}, density_weight={density_weight}"
        )
    if model_family == "polar":
        logging.info(
            f"  Polar: radial_type={radial_type}, n_angular_modes={n_angular_modes}, "
            f"s_max={s_max:.4f}, dc_weight={dc_weight}"
        )

    # watch grads
    watch_interval = config['wandb'].get('watch_interval', 0)
    if watch_interval > 0:
        wandb_logger.run.watch(model, log_freq=watch_interval)
        logging.info(f"wandb watching model at interval {watch_interval}")

    # build optimizer
    if optimizer_name == "Adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "SGD":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # lr scheduler
    scheduler = None
    if scheduler_patience > 0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=scheduler_factor,
            patience=scheduler_patience, min_lr=scheduler_min_lr,
        )
        logging.info(
            f"LR scheduler: ReduceLROnPlateau(patience={scheduler_patience}, "
            f"factor={scheduler_factor}, min_lr={scheduler_min_lr})"
        )

    # plot metadata
    train_spoke_show = int(torch.unique(spoke_id_all[train_idx])[0].item())
    val_spoke_show = int(torch.unique(spoke_id_all[val_idx])[0].item())
    RO_total = int(ro_id_all.max().item()) + 1
    ro_list = [RO_total // 4, RO_total // 2, int(0.8 * RO_total)]

    plot_steps = {1, steps}
    s = plot_every
    while s <= steps:
        plot_steps.add(s)
        s += plot_every

    # training loop
    x_all_2d = x_all[:, :2].to(device)
    y_all_dev = y_all.to(device)
    x_train = x_all_2d[train_idx]
    y_train = y_all_dev[train_idx]
    N_train = x_train.shape[0]

    model.train()
    best_val_loss = float("inf")
    best_state = None
    last_val_loss = None

    logging.info(f"Training for {steps} steps, optimizer={optimizer_name}, lr={lr}")
    logging.info(f"Train points: {N_train}, Val points: {x_val.shape[0]}")

    for step in range(1, steps + 1):
        # train step
        idx = torch.randint(0, N_train, (batch_size,), device=device)
        x = x_train[idx]
        y = y_train[idx]

        opt.zero_grad(set_to_none=True)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        # constraint losses
        if dc_weight > 0 and hasattr(model, "dc_predictions"):
            loss = loss + dc_weight * dc_consistency_loss(model)
        if density_weight > 0:
            loss = loss + density_weight * density_weighted_mse_loss(y_pred, y, x)
        if conj_weight > 0:
            loss = loss + conj_weight * conjugate_symmetry_loss_from_model(model, x)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        train_loss = float(loss.item())

        # validation
        if step % eval_every == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                val_pred = model(x_val)
                last_val_loss = float(loss_fn(val_pred, y_val).item())
            model.train()

            if last_val_loss < best_val_loss:
                best_val_loss = last_val_loss
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }

            # scheduler step
            if scheduler is not None:
                scheduler.step(last_val_loss)

        # scalar logging
        log_dict = {"train/train_loss": train_loss}
        if last_val_loss is not None and (step % eval_every == 0 or step == steps):
            log_dict["train/val_loss"] = last_val_loss
        if scheduler is not None:
            log_dict["train/lr"] = opt.param_groups[0]["lr"]
        wandb_logger.log(log_dict, step=step)

        # figure logging
        if step in plot_steps:
            model.eval()
            with torch.no_grad():
                figures = {}

                # spoke plots
                for sp_id, label in [
                    (train_spoke_show, "train"),
                    (val_spoke_show, "val"),
                ]:
                    fig = make_spoke_figure(
                        model,
                        x_all=x_all, y_all=y_all,
                        spoke_id_all=spoke_id_all,
                        ro_id_all=ro_id_all,
                        spoke_id=sp_id,
                        y_scale=k_scale,
                        n_s=4096,
                        title_prefix=f"[{label}] step {step}",
                        log_scale=log_scale,
                    )
                    figures[f"plots/spoke_{label}"] = fig

                # ring plots
                ring_figs = make_ring_figures(
                    model,
                    x_all=x_all, y_all=y_all,
                    spoke_id_all=spoke_id_all,
                    ro_id_all=ro_id_all,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    ro_list=ro_list,
                    y_scale=k_scale,
                    n_theta=1024,
                    title_prefix=f"step {step}",
                    log_scale=log_scale,
                )
                for i, fig in enumerate(ring_figs):
                    ro_idx_i = ro_list[i // 3]
                    component = ["Re", "Im", "Mag"][i % 3]
                    figures[f"plots/ring_ro{ro_idx_i}_{component}"] = fig

                # error maps
                fig_err_train = make_error_map_figure(
                    model,
                    x_sub=x_all[train_idx],
                    y_sub=y_all[train_idx],
                    y_scale=k_scale,
                    title_prefix=f"[train] step {step}",
                )
                figures["plots/error_map_train"] = fig_err_train

                fig_err_val = make_error_map_figure(
                    model,
                    x_sub=x_all[val_idx],
                    y_sub=y_all[val_idx],
                    y_scale=k_scale,
                    title_prefix=f"[val] step {step}",
                )
                figures["plots/error_map_val"] = fig_err_val

            wandb_logger.log_figures(figures, step=step)
            model.train()

        # console logging
        if step % console_every == 0:
            msg = f"step {step:6d}  train {train_loss:.3e}"
            if last_val_loss is not None:
                msg += f"  val {last_val_loss:.3e}"
            logging.info(msg)

    # restore best, recon, metrics
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    gt_img = data.get("gt_img")

    try:
        with torch.no_grad():
            y_pred_all = model(x_all_2d) * k_scale
            k_pred = torch.complex(y_pred_all[:, 0], y_pred_all[:, 1])
            k_pred_slice = k_pred.reshape(n_ro_per_slice, RO)

            k_img_space_pred = torch.zeros_like(k_img_space)
            k_img_space_pred[t_frame, :, coil_idx, z_slice_idx, :] = k_pred_slice

            img_pred = nufft2d_recon(
                k_img_space_pred, traj_t,
                t_frame=t_frame, coil_idx=coil_idx,
                z_slice_idx=z_slice_idx,
                scales=scales, img_size=img_size, n_slices=n_z_slices,
            )
        wandb_logger.log({"recon/prediction": wandb.Image(img_pred)}, step=steps)

        img_measured = nufft2d_recon(
            k_img_space, traj_t,
            t_frame=t_frame, coil_idx=coil_idx,
            z_slice_idx=z_slice_idx,
            scales=scales, img_size=img_size, n_slices=n_z_slices,
        )
        wandb_logger.log({"recon/measured": wandb.Image(img_measured)}, step=steps)

        # image metrics, vs nufft
        metrics_vs_measured = compute_image_metrics(img_pred, img_measured)
        wandb_logger.log({
            "metrics/psnr_vs_measured": metrics_vs_measured["psnr_db"],
            "metrics/ssim_vs_measured": metrics_vs_measured["ssim"],
            "metrics/nrmse_vs_measured": metrics_vs_measured["nrmse"],
        }, step=steps)
        wandb.run.summary.update({
            "psnr_vs_measured": metrics_vs_measured["psnr_db"],
            "ssim_vs_measured": metrics_vs_measured["ssim"],
            "nrmse_vs_measured": metrics_vs_measured["nrmse"],
        })
        logging.info(
            f"Metrics vs measured:  PSNR={metrics_vs_measured['psnr_db']:.2f} dB  "
            f"SSIM={metrics_vs_measured['ssim']:.4f}  "
            f"NRMSE={metrics_vs_measured['nrmse']:.4f}"
        )

        # vs ground truth
        if gt_img is not None:
            gt_slice = gt_img[t_frame, z_slice_idx, :, :]
            # resize match gt
            if img_pred.shape != gt_slice.shape:
                from scipy.ndimage import zoom
                zoom_factors = (
                    gt_slice.shape[0] / img_pred.shape[0],
                    gt_slice.shape[1] / img_pred.shape[1],
                )
                img_pred_resized = zoom(img_pred, zoom_factors, order=3)
            else:
                img_pred_resized = img_pred

            metrics_vs_gt = compute_image_metrics(img_pred_resized, gt_slice)
            wandb_logger.log({
                "metrics/psnr_vs_gt": metrics_vs_gt["psnr_db"],
                "metrics/ssim_vs_gt": metrics_vs_gt["ssim"],
                "metrics/nrmse_vs_gt": metrics_vs_gt["nrmse"],
            }, step=steps)
            wandb.run.summary.update({
                "psnr_vs_gt": metrics_vs_gt["psnr_db"],
                "ssim_vs_gt": metrics_vs_gt["ssim"],
                "nrmse_vs_gt": metrics_vs_gt["nrmse"],
            })
            logging.info(
                f"Metrics vs GT:       PSNR={metrics_vs_gt['psnr_db']:.2f} dB  "
                f"SSIM={metrics_vs_gt['ssim']:.4f}  "
                f"NRMSE={metrics_vs_gt['nrmse']:.4f}"
            )
    except Exception as e:
        logging.warning(f"NUFFT reconstruction failed: {e}")

    # save model
    wandb_logger.save_model(model, "model_best.pth", opt, steps, output_dir)

    # summary
    wandb.run.summary["best_val_loss"] = best_val_loss
    wandb.run.summary["final_train_loss"] = train_loss
    wandb.run.summary["total_steps"] = steps

    logging.info(f"Done. best_val_loss={best_val_loss:.3e}")
    wandb_logger.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NIK DCE-MRI training with wandb sweeps."
    )
    parser.add_argument('config_path', type=str,
                        help='Path to the configuration TOML file.')
    parser.add_argument('--sweep-id', type=str, default=None,
                        help='Existing wandb sweep ID to join (ENTITY/PROJECT/ID).')
    parser.add_argument('--count', type=int, default=50,
                        help='Number of sweep runs (default: 50).')
    args = parser.parse_args()

    # config, data once
    config = load_config(args.config_path)
    data = load_data(config)

    if args.sweep_id:
        # join existing
        wandb.agent(
            args.sweep_id,
            function=lambda: main(args.config_path, data),
            count=args.count,
        )
    else:
        # new sweep
        sweep_config = config['sweep']
        sweep_id = wandb.sweep(sweep=sweep_config)
        wandb.agent(
            sweep_id,
            function=lambda: main(args.config_path, data),
            count=args.count,
        )
