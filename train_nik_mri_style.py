#!/usr/bin/env python3
"""nikmri style trainer, dce slice"""

import argparse
import importlib.util
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
import wandb
from coolname import generate_slug

from nik_io import load_event
from nik_loss import HDRLossFF
from nik_model import NIK_MRI_SIREN_REIM
from nik_mri_style import (
    DynamicSliceDataset,
    build_nik_mri_dce_dataset,
    coil_combine,
    ifft2c,
    make_recon_videos,
    predict_cartesian_kspace,
    prepare_coil_sensitivity_maps,
)
from nik_recon import ifft1d_kz_to_z
from nik_train import prepare_tensors
from utils.io_utils import (
    copy_config_to_output,
    load_config,
    setup_logging,
    unique_output_dir,
)
from utils.wandb_utils import WandbLogger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        nargs="?",
        default="config/nik_mri_style_dce.toml",
        help="Path to TOML config",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Run name used for outputs and wandb. default: random slug.",
    )
    parser.add_argument("--single", action="store_true",
                        help="Single run, ignore [sweep] section.")
    parser.add_argument("--sweep-id", default=None,
                        help="Join an existing sweep ENTITY/PROJECT/ID.")
    parser.add_argument("--count", type=int, default=50,
                        help="Number of sweep runs (agent count).")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override training.num_steps (smoke test).")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_z_slice(z_slice_idx, n_slices):
    if z_slice_idx == -1:
        return n_slices // 2
    return int(max(0, min(int(z_slice_idx), n_slices - 1)))


def load_dynamic_slice(config):
    data_cfg = config["data"]
    file_path = data_cfg["file"]
    data_device = data_cfg.get("data_device", "cpu")
    reference_t_frame = int(data_cfg.get("reference_t_frame", 0))

    logging.info("Loading radial DCE data from %s", file_path)
    event = load_event(file_path, load_images=True, load_coil_maps=True)
    k_np = np.transpose(event["k"], (0, 2, 1, 3))
    traj_np = np.transpose(event["traj"], (0, 2, 1, 3))

    T, S, C, RO = k_np.shape
    logging.info("Loaded k-space with T=%d S=%d C=%d RO=%d", T, S, C, RO)

    k_t, traj_t, scales, dims, _ = prepare_tensors(
        k_np,
        traj_np,
        data_device=data_device,
    )

    reference_t_frame = min(reference_t_frame, T - 1)
    k_img_space, n_slices, n_ro_per_slice, _ = ifft1d_kz_to_z(
        k_t,
        traj_t,
        t_frame=reference_t_frame,
    )

    z_slice_idx = resolve_z_slice(int(data_cfg.get("z_slice_idx", -1)), n_slices)
    coords, targets, data_meta = build_nik_mri_dce_dataset(
        k_img_space,
        traj_t,
        scales,
        z_slice_idx=z_slice_idx,
        n_slices=n_slices,
        target_device="cpu",
    )
    dataset = DynamicSliceDataset(coords, targets)

    coil_maps = prepare_coil_sensitivity_maps(
        event.get("coil_maps"),
        z_slice_idx,
        n_coils=C,
    )
    require_coil_maps = bool(data_cfg.get("require_coil_maps", True))
    if require_coil_maps and coil_maps is None:
        raise ValueError(
            "This training path requires coil maps in the input data."
        )

    recon_cfg = config.get("recon", {})
    if coil_maps is not None:
        nx, ny = coil_maps.shape[-2:]
    else:
        nx = int(recon_cfg["nx"])
        ny = int(recon_cfg["ny"])

    logging.info(
        "Prepared dynamic slice z=%d with %d frames, %d coils, %d samples",
        z_slice_idx,
        data_meta["n_frames"],
        data_meta["n_coils"],
        data_meta["n_total"],
    )

    return {
        "dataset": dataset,
        "coil_maps": coil_maps,
        "n_frames": data_meta["n_frames"],
        "n_coils": data_meta["n_coils"],
        "nx": int(nx),
        "ny": int(ny),
        "z_slice_idx": z_slice_idx,
        "dataset_meta": data_meta,
        "file_path": file_path,
        "n_slices": n_slices,
        "n_ro_per_slice": n_ro_per_slice,
        "scales": scales,
        "dims": dims,
    }


def build_model(config, device):
    model_cfg = config["model"]
    return NIK_MRI_SIREN_REIM(
        coord_dim=int(model_cfg.get("coord_dim", 4)),
        feature_dim=int(model_cfg.get("feature_dim", 512)),
        num_layers=int(model_cfg.get("num_layers", 8)),
        out_dim=int(model_cfg.get("out_dim", 1)),
        omega_0=float(model_cfg.get("omega_0", 30.0)),
        ff_scale=float(model_cfg.get("ff_scale", 1.0)),
        ff_seed=model_cfg.get("ff_seed"),
        dropout=float(model_cfg.get("dropout", 0.0)),
    ).to(device)


def build_optimizer(config, model):
    training_cfg = config["training"]
    optimizer_name = str(training_cfg.get("optimizer", "Adam"))
    if optimizer_name != "Adam":
        raise ValueError(
            f"NIK_MRI-style training expects Adam, got {optimizer_name}"
        )
    beta1 = float(training_cfg.get("adam_beta1", 0.9))
    beta2 = float(training_cfg.get("adam_beta2", 0.999))
    eps = float(training_cfg.get("adam_eps", 1e-8))
    return torch.optim.Adam(
        model.parameters(),
        lr=float(training_cfg["lr"]),
        betas=(beta1, beta2),
        eps=eps,
    )


def maybe_build_wandb(config, output_dir, run_name):
    wandb_cfg = config.get("wandb", {})
    if not bool(wandb_cfg.get("enabled", True)):
        return None

    # sweep override keys
    flat_config = {
        "data.file": config["data"]["file"],
        "data.z_slice_idx": config["data"].get("z_slice_idx", -1),
        "feature_dim":   config["model"].get("feature_dim", 512),
        "num_layers":    config["model"].get("num_layers", 8),
        "omega_0":       config["model"].get("omega_0", 30.0),
        "ff_scale":      config["model"].get("ff_scale", 1.0),
        "dropout":       config["model"].get("dropout", 0.0),
        "hdr_eps":       config["loss"].get("hdr_eps", 1e-2),
        "hdr_ff_sigma":  config["loss"].get("hdr_ff_sigma", 1.0),
        "hdr_ff_factor": config["loss"].get("hdr_ff_factor", 0.0),
        "lr":            config["training"]["lr"],
        "batch_size":    config["training"]["batch_size"],
        "num_steps":     config["training"].get("num_steps", config["training"].get("steps")),
        "seed":          config["training"].get("seed", 0),
        "adam_beta1":    config["training"].get("adam_beta1", 0.9),
        "adam_beta2":    config["training"].get("adam_beta2", 0.999),
        "adam_eps":      config["training"].get("adam_eps", 1e-8),
    }
    if "project" in wandb_cfg:
        flat_config["project"] = wandb_cfg["project"]
    if "entity" in wandb_cfg:
        flat_config["entity"] = wandb_cfg["entity"]
    logger = WandbLogger(
        config=flat_config,
        output_dir=output_dir,
        run_name=run_name,
        job_type="training",
    )
    logger.initialize()
    return logger


def wandb_video_supported():
    return importlib.util.find_spec("moviepy") is not None


def build_recon_media_log(videos, *, log_videos):
    if log_videos:
        return {
            "recon/k_mag": wandb.Video(videos["k_mag"], fps=4, format="gif"),
            "recon/img_mag": wandb.Video(videos["combined_mag"], fps=4, format="gif"),
            "recon/img_phase": wandb.Video(videos["combined_phase"], fps=4, format="gif"),
        }

    k_frame = videos["k_mag"][-1, 0]
    img_frame = videos["combined_mag"][-1, 0]
    phase_frame = np.moveaxis(videos["combined_phase"][-1], 0, -1)
    return {
        "recon/k_mag_frame": wandb.Image(k_frame),
        "recon/img_mag_frame": wandb.Image(img_frame),
        "recon/img_phase_frame": wandb.Image(phase_frame),
    }


def save_checkpoint(path, model, optimizer, epoch, extra=None):
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }
    if extra is not None:
        payload["meta"] = extra
    torch.save(payload, path)


def run_training(config_path, data, *, run_name=None, steps_override=None,
                 save_checkpoint_file=True):
    """one training run"""
    config = load_config(config_path)
    if run_name is None:
        run_name = generate_slug(2) + "_nikmri"
    output_dir = unique_output_dir(config, run_name)
    copy_config_to_output(config_path, output_dir)
    setup_logging(config, output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb_logger = maybe_build_wandb(config, output_dir, run_name)
    can_log_videos = wandb_logger is not None and wandb_video_supported()
    if wandb_logger is not None and not can_log_videos:
        logging.warning(
            "moviepy not installed; wandb video logging disabled, static images instead."
        )

    # hyperparam resolution
    mc, lc, tc = config["model"], config["loss"], config["training"]
    wc = wandb.config if wandb_logger is not None else {}
    g = (lambda name, default: getattr(wc, name) if hasattr(wc, name) else default)

    feature_dim   = int(g("feature_dim",   mc.get("feature_dim",   512)))
    num_layers    = int(g("num_layers",    mc.get("num_layers",    8)))
    omega_0       = float(g("omega_0",     mc.get("omega_0",       30.0)))
    ff_scale      = float(g("ff_scale",    mc.get("ff_scale",      1.0)))
    dropout       = float(g("dropout",     mc.get("dropout",       0.0)))
    coord_dim     = int(mc.get("coord_dim", 4))
    out_dim       = int(mc.get("out_dim",   1))
    ff_seed       = mc.get("ff_seed")

    hdr_eps       = float(g("hdr_eps",      lc.get("hdr_eps",      1e-2)))
    hdr_ff_sigma  = float(g("hdr_ff_sigma", lc.get("hdr_ff_sigma", 1.0)))
    hdr_ff_factor = float(g("hdr_ff_factor", lc.get("hdr_ff_factor", 0.0)))

    lr            = float(g("lr",          tc.get("lr",            3e-5)))
    batch_size    = int(g("batch_size",    tc["batch_size"]))
    num_steps     = int(g("num_steps",     tc.get("num_steps", tc.get("steps", 50))))
    if steps_override is not None:
        num_steps = int(steps_override)
    seed          = int(g("seed",          tc.get("seed", 0)))
    set_seed(seed)

    num_workers          = int(tc.get("num_workers", 0))
    recon_every          = int(tc.get("recon_every", 1))
    cartesian_chunk_size = int(tc.get("cartesian_chunk_size", 131072))
    console_every        = int(config["logging"].get("console_every", 100))

    model = NIK_MRI_SIREN_REIM(
        coord_dim=coord_dim,
        feature_dim=feature_dim,
        num_layers=num_layers,
        out_dim=out_dim,
        omega_0=omega_0,
        ff_scale=ff_scale,
        ff_seed=ff_seed,
        dropout=dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if wandb_logger is not None:
        wandb.config.update({"n_params": n_params}, allow_val_change=True)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(float(tc.get("adam_beta1", 0.9)), float(tc.get("adam_beta2", 0.999))),
        eps=float(tc.get("adam_eps", 1e-8)),
    )
    criterion = HDRLossFF(sigma=hdr_ff_sigma, eps=hdr_eps, factor=hdr_ff_factor)

    dataloader = DataLoader(
        data["dataset"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    logging.info("Training on %s", device)
    logging.info("Outputs will be written to %s", output_dir)
    logging.info("model: feature_dim=%d num_layers=%d omega_0=%g ff_scale=%g params=%d",
                 feature_dim, num_layers, omega_0, ff_scale, n_params)
    logging.info("Training for %d outer epochs", num_steps)

    best_val, best_step = float("inf"), -1

    for epoch in range(1, num_steps + 1):
        model.train()
        loss_epoch = 0.0
        reg_epoch = 0.0
        n_batches = 0

        for sample in dataloader:
            coords = sample["coords"].to(device, non_blocking=True)
            targets = sample["targets"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(coords)
            loss, reg = criterion(pred, targets, coords)
            loss.backward()
            optimizer.step()

            loss_epoch += float(loss.item())
            reg_epoch += float(reg.item())
            n_batches += 1

        mean_loss = loss_epoch / max(n_batches, 1)
        mean_reg = reg_epoch / max(n_batches, 1)
        log_dict = {
            "train/loss": mean_loss,
            "train/reg": mean_reg,
            "train/lr": optimizer.param_groups[0]["lr"],
        }

        if epoch % recon_every == 0 or epoch == num_steps:
            kpred = predict_cartesian_kspace(
                model,
                nt=data["n_frames"],
                nc=data["n_coils"],
                nx=data["nx"],
                ny=data["ny"],
                device=device,
                chunk_size=cartesian_chunk_size,
                time_coords=data["dataset_meta"]["time_coords"],
                coil_coords=data["dataset_meta"]["coil_coords"],
            )
            coil_imgs = ifft2c(kpred)
            combined = coil_combine(coil_imgs, data["coil_maps"])

            if wandb_logger is not None:
                videos = make_recon_videos(kpred, combined)
                log_dict["recon/k_hist"] = wandb.Histogram(
                    torch.view_as_real(kpred).detach().cpu().numpy().reshape(-1)
                )
                log_dict.update(build_recon_media_log(videos, log_videos=can_log_videos))

        if mean_loss < best_val:
            best_val, best_step = mean_loss, epoch

        if wandb_logger is not None:
            wandb_logger.log(log_dict, step=epoch)

        if epoch % console_every == 0 or epoch == 1 or epoch == num_steps:
            logging.info(
                "epoch %6d/%6d  loss %.4e  reg %.4e",
                epoch, num_steps, mean_loss, mean_reg,
            )

    if save_checkpoint_file:
        checkpoint_path = os.path.join(output_dir, "nik_mri_style_final.pt")
        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            num_steps,
            extra={
                "dataset_meta": data["dataset_meta"],
                "z_slice_idx": data["z_slice_idx"],
                "file_path": data["file_path"],
            },
        )
        logging.info("Saved final checkpoint to %s", checkpoint_path)

    if wandb_logger is not None:
        wandb.run.summary.update({
            "best_train_loss": best_val,
            "best_step": best_step,
            "n_params": n_params,
        })
        wandb_logger.finish()
    logging.info("done. best_train_loss=%.3e @ epoch %d", best_val, best_step)


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["training"].get("seed", 0)))
    data = load_dynamic_slice(config)

    if args.single or args.sweep_id is None and "sweep" not in config:
        run_training(args.config, data,
                     run_name=args.run_name, steps_override=args.steps,
                     save_checkpoint_file=True)
        return

    if args.sweep_id:
        wandb.agent(
            args.sweep_id,
            function=lambda: run_training(args.config, data,
                                          run_name=args.run_name,
                                          steps_override=args.steps,
                                          save_checkpoint_file=False),
            count=args.count,
        )
    else:
        sweep_id = wandb.sweep(sweep=config["sweep"])
        print(f"sweep id: {sweep_id}", flush=True)
        wandb.agent(
            sweep_id,
            function=lambda: run_training(args.config, data,
                                          run_name=args.run_name,
                                          steps_override=args.steps,
                                          save_checkpoint_file=False),
            count=args.count,
        )


if __name__ == "__main__":
    main()
