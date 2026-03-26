#!/usr/bin/env python
"""
train_multicoil.py -- Multi-coil NIK training with wandb logging.

Trains a multi-coil model (FiLM or Concat) on all C coils jointly.
Based on train_wandb.py but with multi-coil data loading and model.

Usage:
    # Single run:
    python train_multicoil.py config/step15_multicoil.toml

    # Sweep:
    python train_multicoil.py config/step15_multicoil.toml --sweep-id ENTITY/PROJECT/ID --count 10
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

from nik_io import load_event
from nik_model import (
    MultiCoilWIRE,
    MultiCoilSIREN,
    MultiCoilConcat,
)
from nik_loss import get_loss_fn
from nik_train import prepare_tensors
from nik_recon import (
    ifft1d_kz_to_z,
    make_fixed_frame_zslice_multicoil_dataset,
    split_points_by_spokes,
    split_points_by_angular_sector,
    verify_spoke_holdout,
    verify_multicoil_data,
    nufft2d_recon,
    nufft2d_recon_multicoil_sos,
)
from nik_metrics import compute_image_metrics, compute_perceptual_metrics


def load_data(config):
    """Load and preprocess multi-coil data."""
    data_cfg = config['data']
    file_path = data_cfg['file']
    t_frame = data_cfg['t_frame']
    z_slice_raw = data_cfg['z_slice_idx']
    val_frac = data_cfg['val_frac']
    seed = config['training']['seed']

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading data from {file_path} ...", flush=True)
    event = load_event(file_path, load_images=True)
    k_np, traj_np = event["k"], event["traj"]
    gt_img = event.get("gt_img")

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

    # Multi-coil dataset: all C coils
    x_all, y_all, coil_id_all, spoke_id_all, ro_id_all, meta = \
        make_fixed_frame_zslice_multicoil_dataset(
            k_img_space, traj_t, scales, dims,
            y_scale=k_scale,
            t_fixed=t_frame,
            z_slice_idx=z_slice_idx,
            n_slices=n_z_slices,
            compute_device=device,
        )
    print(f"Multi-coil dataset: {meta['N_total']} points "
          f"({meta['n_coils']} coils × {meta['N_per_coil']} pts/coil)", flush=True)

    # Spoke-based train/val split (identical across coils)
    train_idx, val_idx, train_spokes, val_spokes = split_points_by_spokes(
        spoke_id_all, val_frac=val_frac, seed=seed, mode="random",
    )

    # ---- Verification checks ----
    N_per_coil = meta["N_per_coil"]
    print("Verifying multi-coil data loader...", flush=True)
    verify_multicoil_data(x_all, y_all, coil_id_all, spoke_id_all,
                          val_idx, C, N_per_coil)
    verify_spoke_holdout(spoke_id_all, val_idx, train_idx,
                         n_coils=C, coil_id_all=coil_id_all)
    print(f"  Val spokes: {len(val_spokes)}, Train spokes: {len(train_spokes)}")
    print(f"  Val points: {len(val_idx)}, Train points: {len(train_idx)}")
    print("All multi-coil data loader checks passed.", flush=True)

    # Angular sector split (additional eval diagnostic — not used for training)
    # Compute spoke angles from trajectory
    sx, sy, _ = scales
    indices = torch.arange(0, S, n_z_slices, device=traj_t.device)
    kx_sp = traj_t[t_frame, indices, 0, RO // 2] / sx
    ky_sp = traj_t[t_frame, indices, 1, RO // 2] / sy
    theta_sp = torch.atan2(ky_sp, kx_sp)

    # Hold out sector 0 (of 4 sectors) for angular validation
    sector_train_idx, sector_val_idx, sector_val_spokes = \
        split_points_by_angular_sector(
            spoke_id_all, theta_sp, n_sectors=4, val_sector=0,
        )
    print(f"Angular sector holdout: {len(sector_val_spokes)} spokes, "
          f"{len(sector_val_idx)} points in sector 0", flush=True)

    img_size = gt_img.shape[2:4] if gt_img is not None else (128, 128)

    return {
        "k_img_space": k_img_space, "traj_t": traj_t, "scales": scales,
        "dims": dims, "k_scale": k_scale,
        "x_all": x_all, "y_all": y_all,
        "coil_id_all": coil_id_all,
        "spoke_id_all": spoke_id_all, "ro_id_all": ro_id_all,
        "train_idx": train_idx, "val_idx": val_idx,
        "sector_val_idx": sector_val_idx,
        "meta": meta, "n_z_slices": n_z_slices,
        "n_ro_per_slice": n_ro_per_slice,
        "T": T, "S": S, "C": C, "RO": RO,
        "z_slice_idx": z_slice_idx, "img_size": img_size,
        "gt_img": gt_img,
    }


def main(config_path, data):
    """Single multi-coil training run."""
    random.seed()
    run_name = generate_slug(3) + "_mc"
    config = load_config(config_path)
    output_dir = unique_output_dir(config, run_name)
    copy_config_to_output(config_path, output_dir)
    setup_logging(config, output_dir)

    logging.info(f"Run: {run_name}")
    logging.info(f"Config: {config_path}")
    logging.info(f"Output: {output_dir}")

    # Build training config
    train_config = {
        "model_family": config['model'].get('model_family', 'wire'),
        "multicoil_mode": config['model'].get('multicoil_mode', 'film'),
        "hidden": config['model']['hidden'],
        "depth": config['model']['depth'],
        "w0": config['model'].get('w0', 60),
        "s0": config['model'].get('s0', 10.0),
        "coil_embed_dim": config['model'].get('coil_embed_dim', 32),
        "optimizer": config['training']['optimizer'],
        "lr": config['training']['lr'],
        "batch_size": config['training']['batch_size'],
        "steps": config['training']['steps'],
        "eval_every": config['training']['eval_every'],
        "grad_clip": config['training']['grad_clip'],
        "weight_decay": config['training'].get('weight_decay', 0.0),
        "loss_type": config['training'].get('loss_type', "mse"),
        "seed": config['training']['seed'],
        "scheduler_patience": config['training'].get('scheduler_patience', 0),
        "scheduler_factor": config['training'].get('scheduler_factor', 0.5),
        "scheduler_min_lr": config['training'].get('scheduler_min_lr', 1e-6),
    }

    wandb_logger = WandbLogger(
        config=train_config,
        output_dir=output_dir,
        run_name=run_name,
        job_type="training",
    )
    wandb_logger.initialize()

    wc = wandb.config
    model_family = str(getattr(wc, "model_family", "wire"))
    multicoil_mode = str(getattr(wc, "multicoil_mode", "film"))
    hidden = int(wc.hidden)
    depth = int(wc.depth)
    w0 = float(getattr(wc, "w0", 60))
    s0 = float(getattr(wc, "s0", 10.0))
    coil_embed_dim = int(getattr(wc, "coil_embed_dim", 32))
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

    console_every = config['logging']['console_every']

    # Reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Unpack data
    k_img_space = data["k_img_space"]
    traj_t = data["traj_t"]
    scales = data["scales"]
    k_scale = data["k_scale"]
    x_all = data["x_all"]
    y_all = data["y_all"]
    coil_id_all = data["coil_id_all"]
    spoke_id_all = data["spoke_id_all"]
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    sector_val_idx = data["sector_val_idx"]
    meta = data["meta"]
    n_z_slices = data["n_z_slices"]
    n_ro_per_slice = data["n_ro_per_slice"]
    T, S, C, RO = data["T"], data["S"], data["C"], data["RO"]
    z_slice_idx = data["z_slice_idx"]
    img_size = data["img_size"]
    t_frame = config['data']['t_frame']
    N_per_coil = meta["N_per_coil"]

    # Build model
    backbone_kwargs = dict(hidden=hidden, depth=depth, w0=w0, s0=s0)

    if multicoil_mode == "film":
        if model_family == "wire":
            model = MultiCoilWIRE(
                in_dim=2, n_coils=C, coil_embed_dim=coil_embed_dim,
                **backbone_kwargs,
            ).to(device)
        elif model_family == "siren":
            model = MultiCoilSIREN(
                in_dim=2, n_coils=C, coil_embed_dim=coil_embed_dim,
                hidden=hidden, depth=depth, w0=w0,
            ).to(device)
        else:
            raise ValueError(f"FiLM not implemented for {model_family}")
    elif multicoil_mode == "concat":
        model = MultiCoilConcat(
            backbone_family=model_family,
            backbone_kwargs=backbone_kwargs,
            n_coils=C,
            coil_embed_dim=coil_embed_dim,
        ).to(device)
    else:
        raise ValueError(f"Unknown multicoil_mode: {multicoil_mode}")

    n_params = sum(p.numel() for p in model.parameters())

    # Param count breakdown: backbone vs coil-specific
    if multicoil_mode == "concat":
        backbone_params = sum(p.numel() for p in model.backbone.parameters())
        coil_params = sum(p.numel() for p in model.coil_embed.parameters())
    else:
        # FiLM: backbone = linears + head, coil = film modules
        backbone_params = sum(p.numel() for n, p in model.named_parameters()
                              if 'film' not in n and 'coil' not in n)
        coil_params = n_params - backbone_params

    wandb.config.update({
        "n_params": n_params, "n_coils": C,
        "backbone_params": backbone_params, "coil_params": coil_params,
        "N_per_coil": N_per_coil, "N_total": meta["N_total"],
    }, allow_val_change=True)
    logging.info(
        f"Model: multicoil_{multicoil_mode}_{model_family}, "
        f"hidden={hidden}, depth={depth}, w0={w0}, s0={s0}, "
        f"coil_embed_dim={coil_embed_dim}, params={n_params} "
        f"(backbone={backbone_params}, coil={coil_params})"
    )

    # Optimizer
    if optimizer_name == "Adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # LR scheduler
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

    # Prepare training tensors
    x_all_2d = x_all[:, :2].to(device)
    y_all_dev = y_all.to(device)
    coil_all_dev = coil_id_all.to(device)

    x_train = x_all_2d[train_idx]
    y_train = y_all_dev[train_idx]
    coil_train = coil_all_dev[train_idx]
    N_train = x_train.shape[0]

    x_val = x_all_2d[val_idx]
    y_val = y_all_dev[val_idx]
    coil_val = coil_all_dev[val_idx]

    # Angular sector val tensors (additional diagnostic)
    x_sector_val = x_all_2d[sector_val_idx]
    y_sector_val = y_all_dev[sector_val_idx]
    coil_sector_val = coil_all_dev[sector_val_idx]

    model.train()
    best_val_loss = float("inf")
    best_state = None
    last_val_loss = None

    logging.info(f"Training for {steps} steps, optimizer={optimizer_name}, lr={lr}")
    logging.info(f"Train points: {N_train}, Val points (spokes): {x_val.shape[0]}, "
                 f"Val points (sector): {x_sector_val.shape[0]}")

    # Per-coil val loss tracking
    val_coil_ids = coil_id_all[val_idx]

    for step in range(1, steps + 1):
        # --- Training step: random batch from all coils ---
        idx = torch.randint(0, N_train, (batch_size,), device=device)
        x = x_train[idx]
        y = y_train[idx]
        c = coil_train[idx]

        opt.zero_grad(set_to_none=True)
        y_pred = model(x, c)
        loss = loss_fn(y_pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        train_loss = float(loss.item())

        # --- Validation (PRIMARY: k-space MSE on held-out whole spokes) ---
        if step % eval_every == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                val_pred = model(x_val, coil_val)
                last_val_loss = float(loss_fn(val_pred, y_val).item())

                # Per-coil val loss + angular sector (every 10 evals or at end)
                if step == steps or step % (eval_every * 10) == 0:
                    per_coil_val = {}
                    for ci in range(C):
                        mask = val_coil_ids == ci
                        if mask.any():
                            coil_loss = float(F.mse_loss(
                                val_pred[mask], y_val[mask]
                            ).item())
                            per_coil_val[f"val_coil_{ci}"] = coil_loss
                    wandb_logger.log(
                        {f"train/{k}": v for k, v in per_coil_val.items()},
                        step=step,
                    )

                    # Angular sector validation (harder test)
                    sector_pred = model(x_sector_val, coil_sector_val)
                    sector_val_loss = float(loss_fn(sector_pred, y_sector_val).item())
                    wandb_logger.log({
                        "train/val_loss_angular_sector": sector_val_loss,
                    }, step=step)

            model.train()

            if last_val_loss < best_val_loss:
                best_val_loss = last_val_loss
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }

            if scheduler is not None:
                scheduler.step(last_val_loss)

        # --- Logging ---
        log_dict = {"train/train_loss": train_loss}
        if last_val_loss is not None and (step % eval_every == 0 or step == steps):
            log_dict["train/val_loss"] = last_val_loss
        if scheduler is not None:
            log_dict["train/lr"] = opt.param_groups[0]["lr"]
        wandb_logger.log(log_dict, step=step)

        if step % console_every == 0:
            msg = f"step {step:6d}  train {train_loss:.3e}"
            if last_val_loss is not None:
                msg += f"  val {last_val_loss:.3e}"
            logging.info(msg)

    # ---- Restore best model and evaluate ----
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    gt_img = data.get("gt_img")

    # ---- PRIMARY: k-space validation metrics ----
    with torch.no_grad():
        val_pred_final = model(x_val, coil_val)
        val_kspace_mse = float(F.mse_loss(val_pred_final, y_val).item())

        sector_pred_final = model(x_sector_val, coil_sector_val)
        val_sector_mse = float(F.mse_loss(sector_pred_final, y_sector_val).item())

        # Per-coil breakdown
        per_coil_mse = []
        for ci in range(C):
            mask = val_coil_ids == ci
            if mask.any():
                per_coil_mse.append(float(F.mse_loss(
                    val_pred_final[mask], y_val[mask]).item()))

    val_train_ratio = val_kspace_mse / max(train_loss, 1e-12)

    wandb.run.summary.update({
        "val_kspace_mse_spokes": val_kspace_mse,
        "val_kspace_mse_angular_sector": val_sector_mse,
        "val_train_ratio": val_train_ratio,
    })
    logging.info(f"--- PRIMARY METRICS (k-space) ---")
    logging.info(f"  Val MSE (whole spokes):    {val_kspace_mse:.3e}")
    logging.info(f"  Val MSE (angular sector):  {val_sector_mse:.3e}")
    logging.info(f"  Final train MSE:           {train_loss:.3e}")
    logging.info(f"  Val/train ratio:           {val_train_ratio:.1f}x")
    logging.info(f"  Per-coil val MSE:          {['%.3e' % m for m in per_coil_mse]}")

    # ---- SECONDARY: SOS image reconstruction (qualitative) ----
    try:
        recon = nufft2d_recon_multicoil_sos(
            model,
            x_all=x_all,
            coil_id_all=coil_id_all,
            y_scale=k_scale,
            k_img_space=k_img_space,
            traj_t=traj_t,
            scales=scales,
            t_frame=t_frame,
            z_slice_idx=z_slice_idx,
            n_z_slices=n_z_slices,
            n_ro_per_slice=n_ro_per_slice,
            n_coils=C,
            N_per_coil=N_per_coil,
            RO=RO,
            img_size=img_size,
        )

        img_sos_pred = recon["img_sos_pred"]
        img_sos_meas = recon["img_sos_measured"]

        wandb_logger.log({
            "recon/sos_predicted": wandb.Image(img_sos_pred),
            "recon/sos_measured": wandb.Image(img_sos_meas),
        }, step=steps)

        # Log per-coil images
        for ci, img_c in enumerate(recon["imgs_per_coil"]):
            wandb_logger.log({
                f"recon/coil_{ci}": wandb.Image(img_c),
            }, step=steps)

        # ---- TERTIARY: image-space metrics vs NUFFT proxy ----
        # NOTE: metrics_vs_nufft_proxy compares against NUFFT reconstruction,
        # which itself has artifacts. These are PROXY metrics for relative
        # comparison between models, NOT absolute quality measures.
        metrics_sos = compute_image_metrics(img_sos_pred, img_sos_meas)
        wandb_logger.log({
            "metrics/psnr_sos_vs_nufft": metrics_sos["psnr_db"],
            "metrics/ssim_sos_vs_nufft": metrics_sos["ssim"],
            "metrics/nrmse_sos_vs_nufft": metrics_sos["nrmse"],
        }, step=steps)
        wandb.run.summary.update({
            "psnr_sos_vs_nufft": metrics_sos["psnr_db"],
            "ssim_sos_vs_nufft": metrics_sos["ssim"],
            "nrmse_sos_vs_nufft": metrics_sos["nrmse"],
        })
        logging.info(
            f"SOS vs NUFFT proxy:  PSNR={metrics_sos['psnr_db']:.2f} dB  "
            f"SSIM={metrics_sos['ssim']:.4f}  NRMSE={metrics_sos['nrmse']:.4f}"
        )

        # Perceptual metrics (end of training only)
        try:
            perc = compute_perceptual_metrics(img_sos_pred, img_sos_meas)
            wandb_logger.log({
                f"metrics/{k}": v for k, v in perc.items()
            }, step=steps)
            wandb.run.summary.update(perc)
            logging.info(
                f"Perceptual vs NUFFT proxy:  DISTS={perc['DISTS']:.4f}  "
                f"HaarPSI={perc['HaarPSI']:.4f}  VSI={perc['VSI']:.4f}"
            )
        except Exception as e:
            logging.warning(f"Perceptual metrics failed: {e}")

        # Coil embedding visualization
        try:
            if hasattr(model, 'first_film'):
                embeds = model.first_film.coil_embed.weight.detach().cpu().numpy()
            elif hasattr(model, 'coil_embed'):
                embeds = model.coil_embed.weight.detach().cpu().numpy()
            else:
                embeds = None

            if embeds is not None:
                # Log pairwise distances
                from scipy.spatial.distance import pdist, squareform
                dists = squareform(pdist(embeds))
                logging.info(f"Coil embedding pairwise distances:\n{np.array2string(dists, precision=3)}")
        except Exception as e:
            logging.warning(f"Coil embedding visualization failed: {e}")

    except Exception as e:
        logging.warning(f"Multi-coil reconstruction failed: {e}")
        import traceback
        traceback.print_exc()

    # Save model
    wandb_logger.save_model(model, "model_best.pth", opt, steps, output_dir)

    # Summary
    wandb.run.summary["best_val_loss"] = best_val_loss
    wandb.run.summary["final_train_loss"] = train_loss
    wandb.run.summary["total_steps"] = steps

    logging.info(f"Done. best_val_loss={best_val_loss:.3e}")
    wandb_logger.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-coil NIK training with wandb."
    )
    parser.add_argument('config_path', type=str,
                        help='Path to the configuration TOML file.')
    parser.add_argument('--sweep-id', type=str, default=None,
                        help='Existing wandb sweep ID to join.')
    parser.add_argument('--count', type=int, default=10,
                        help='Number of sweep runs (default: 10).')
    args = parser.parse_args()

    config = load_config(args.config_path)
    data = load_data(config)

    if args.sweep_id:
        wandb.agent(
            args.sweep_id,
            function=lambda: main(args.config_path, data),
            count=args.count,
        )
    else:
        sweep_cfg = config.get('sweep')
        if sweep_cfg:
            sweep_id = wandb.sweep(sweep=sweep_cfg)
            wandb.agent(
                sweep_id,
                function=lambda: main(args.config_path, data),
                count=args.count,
            )
        else:
            # Single run (no sweep)
            # Initialize wandb directly for non-sweep mode
            main(args.config_path, data)
