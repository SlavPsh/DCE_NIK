#!/usr/bin/env python3
"""
run_polar_ablation.py -- Systematic ablation: Cartesian vs Polar k-space models.

Compares 6 configurations on one slice, one coil, one frame:
  A. Baseline: Cartesian SIREN (depth 7, width 256)
  B. Polar + SIREN radial (depth 4, width 128, N=16)
  C. Polar + WIRE radial  (depth 4, width 128, N=16)
  D. Polar + WIRE + DC consistency loss
  E. Polar + WIRE + DC + density-weighted loss
  F. Polar + WIRE + DC + density + conjugate symmetry

Usage:
    python run_polar_ablation.py                    # default settings
    python run_polar_ablation.py --steps 30000      # more training
    python run_polar_ablation.py --device cpu        # force CPU
"""
import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nik_io import load_event
from nik_model import (
    NIK_SIREN_KXY_REIM,
    PolarKSpaceNet,
)
from nik_loss import (
    mse_loss,
    density_weighted_mse_loss,
    dc_consistency_loss,
    conjugate_symmetry_loss_from_model,
)
from nik_metrics import per_spoke_mse
from nik_train import prepare_tensors
from nik_recon import (
    ifft1d_kz_to_z,
    make_fixed_frame_zslice_coil_dataset,
    split_points_by_spokes,
    nufft2d_recon,
)
from nik_metrics import compute_image_metrics


# ---------------------------------------------------------------------------
# Data loading (reuse from train_wandb.py pattern)
# ---------------------------------------------------------------------------

def load_data(data_file, t_frame=0, coil_idx=0, z_slice_idx=-1, val_frac=0.1, seed=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading data from {data_file} ...", flush=True)
    event = load_event(data_file, load_images=True)
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

    z_idx = n_z_slices // 2 if z_slice_idx == -1 else int(z_slice_idx)

    x_all, y_all, kx_all, ky_all, spoke_id_all, ro_id_all, meta = \
        make_fixed_frame_zslice_coil_dataset(
            k_img_space, traj_t, scales, dims,
            y_scale=k_scale,
            t_fixed=t_frame,
            coil_fixed=coil_idx,
            z_slice_idx=z_idx,
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
        "z_slice_idx": z_idx, "img_size": img_size,
        "gt_img": gt_img,
        "t_frame": t_frame, "coil_idx": coil_idx,
    }


# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------

def make_configs():
    """Return list of (name, model_builder, loss_builder) tuples."""

    # Compute s_max from data at runtime; placeholder here
    configs = []

    # A. Baseline SIREN
    configs.append({
        "name": "A_siren_baseline",
        "label": "A: Cartesian SIREN (d=7, h=256)",
        "model_fn": lambda s_max: NIK_SIREN_KXY_REIM(
            in_dim=2, hidden=256, depth=7, w0=30.0,
        ),
        "loss_type": "mse_only",
        "lr": 2e-4,
        "weight_decay": 0.0,
    })

    # B. Polar + SIREN radial
    configs.append({
        "name": "B_polar_siren",
        "label": "B: Polar + SIREN (d=4, h=128, N=16)",
        "model_fn": lambda s_max: PolarKSpaceNet(
            n_angular_modes=16, radial_depth=4, radial_width=128,
            radial_type="siren", omega_0=30.0, s_max=s_max,
        ),
        "loss_type": "mse_only",
        "lr": 2e-4,
        "weight_decay": 0.0,
    })

    # C. Polar + WIRE radial
    configs.append({
        "name": "C_polar_wire",
        "label": "C: Polar + WIRE (d=4, h=128, N=16)",
        "model_fn": lambda s_max: PolarKSpaceNet(
            n_angular_modes=16, radial_depth=4, radial_width=128,
            radial_type="wire", omega_0=30.0, s_0=10.0, s_max=s_max,
        ),
        "loss_type": "mse_only",
        "lr": 2e-4,
        "weight_decay": 0.0,
    })

    # D. Polar + WIRE + DC consistency
    configs.append({
        "name": "D_polar_wire_dc",
        "label": "D: Polar + WIRE + DC loss",
        "model_fn": lambda s_max: PolarKSpaceNet(
            n_angular_modes=16, radial_depth=4, radial_width=128,
            radial_type="wire", omega_0=30.0, s_0=10.0, s_max=s_max,
        ),
        "loss_type": "mse_dc",
        "lr": 2e-4,
        "weight_decay": 0.0,
    })

    # E. Polar + WIRE + DC + density
    configs.append({
        "name": "E_polar_wire_dc_density",
        "label": "E: Polar + WIRE + DC + density",
        "model_fn": lambda s_max: PolarKSpaceNet(
            n_angular_modes=16, radial_depth=4, radial_width=128,
            radial_type="wire", omega_0=30.0, s_0=10.0, s_max=s_max,
        ),
        "loss_type": "mse_dc_density",
        "lr": 2e-4,
        "weight_decay": 0.0,
    })

    # F. Polar + WIRE + DC + density + conjugate symmetry
    configs.append({
        "name": "F_polar_wire_dc_density_conj",
        "label": "F: Polar + WIRE + DC + density + conj",
        "model_fn": lambda s_max: PolarKSpaceNet(
            n_angular_modes=16, radial_depth=4, radial_width=128,
            radial_type="wire", omega_0=30.0, s_0=10.0, s_max=s_max,
        ),
        "loss_type": "mse_dc_density_conj",
        "lr": 2e-4,
        "weight_decay": 0.0,
    })

    return configs


# ---------------------------------------------------------------------------
# Loss dispatcher
# ---------------------------------------------------------------------------

def compute_loss(loss_type, y_pred, y_true, k_coords, model,
                 dc_weight=0.01, density_weight=1.0, conj_weight=0.01):
    """Compute combined loss based on loss_type string."""
    loss = mse_loss(y_pred, y_true)

    if "dc" in loss_type:
        loss = loss + dc_weight * dc_consistency_loss(model)

    if "density" in loss_type:
        loss = loss + density_weight * density_weighted_mse_loss(y_pred, y_true, k_coords)

    if "conj" in loss_type:
        loss = loss + conj_weight * conjugate_symmetry_loss_from_model(model, k_coords)

    return loss


# ---------------------------------------------------------------------------
# Training loop for one config
# ---------------------------------------------------------------------------

def train_one(config, data, steps=20000, batch_size=4096, eval_every=50,
              grad_clip=1.0, seed=0, device="cuda"):
    """Train a single model configuration and return results dict."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    name = config["name"]
    label = config["label"]
    print(f"\n{'='*70}")
    print(f"  Training: {label}")
    print(f"{'='*70}")

    # Unpack data
    x_all = data["x_all"]
    y_all = data["y_all"]
    spoke_id_all = data["spoke_id_all"]
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    k_scale = data["k_scale"]

    x_all_2d = x_all[:, :2].to(device)
    y_all_dev = y_all.to(device)

    x_train = x_all_2d[train_idx]
    y_train = y_all_dev[train_idx]
    x_val = x_all_2d[val_idx]
    y_val = y_all_dev[val_idx]
    N_train = x_train.shape[0]

    # Compute s_max from training data
    kx = x_train[:, 0]
    ky = x_train[:, 1]
    theta = torch.atan2(ky, kx)
    theta0 = torch.remainder(theta + 0.5 * np.pi, np.pi) - 0.5 * np.pi
    c = torch.cos(theta0)
    s = torch.sin(theta0)
    s_coord = kx * c + ky * s
    s_max = float(s_coord.abs().max().item())
    print(f"  s_max = {s_max:.4f}")

    # Build model
    model = config["model_fn"](s_max).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Optimizer
    lr = config["lr"]
    weight_decay = config["weight_decay"]
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_type = config["loss_type"]

    # Training history
    train_losses = []
    val_losses = []
    val_steps = []
    best_val_loss = float("inf")
    best_state = None
    best_step = 0
    overfit_onset_step = None  # first step where val > 1.05 * best_val

    t_start = time.time()
    step_times = []

    model.train()
    for step in range(1, steps + 1):
        t_step = time.time()

        idx = torch.randint(0, N_train, (batch_size,), device=device)
        x = x_train[idx]
        y = y_train[idx]

        opt.zero_grad(set_to_none=True)
        y_pred = model(x)
        loss = compute_loss(loss_type, y_pred, y, x, model)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        step_times.append(time.time() - t_step)
        train_losses.append(float(loss.item()))

        # Validation
        if step % eval_every == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                val_pred = model(x_val)
                val_loss = float(F.mse_loss(val_pred, y_val).item())
            model.train()

            val_losses.append(val_loss)
            val_steps.append(step)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = step
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }

            # Detect overfitting onset
            if overfit_onset_step is None and val_loss > 1.05 * best_val_loss and step > eval_every * 5:
                overfit_onset_step = step

            if step % 2000 == 0 or step == steps:
                print(f"  step {step:6d}  train {train_losses[-1]:.3e}  val {val_loss:.3e}  best {best_val_loss:.3e}")

    total_time = time.time() - t_start
    mean_step_time = np.mean(step_times)

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # Per-spoke validation MSE
    spoke_results = per_spoke_mse(
        model, x_all, y_all, spoke_id_all, idx=val_idx,
    )

    # Image reconstruction metrics
    img_metrics = {}
    try:
        with torch.no_grad():
            y_pred_all = model(x_all_2d) * k_scale
            k_pred = torch.complex(y_pred_all[:, 0], y_pred_all[:, 1])
            n_ro_per_slice = data["n_ro_per_slice"]
            RO = data["RO"]
            k_pred_slice = k_pred.reshape(n_ro_per_slice, RO)

            k_img_space = data["k_img_space"]
            traj_t = data["traj_t"]
            scales = data["scales"]
            img_size = data["img_size"]
            n_z_slices = data["n_z_slices"]
            t_frame = data["t_frame"]
            coil_idx = data["coil_idx"]
            z_slice_idx = data["z_slice_idx"]

            k_img_space_pred = torch.zeros_like(k_img_space)
            k_img_space_pred[t_frame, :, coil_idx, z_slice_idx, :] = k_pred_slice

            img_pred = nufft2d_recon(
                k_img_space_pred, traj_t,
                t_frame=t_frame, coil_idx=coil_idx,
                z_slice_idx=z_slice_idx,
                scales=scales, img_size=img_size, n_slices=n_z_slices,
            )

            img_measured = nufft2d_recon(
                k_img_space, traj_t,
                t_frame=t_frame, coil_idx=coil_idx,
                z_slice_idx=z_slice_idx,
                scales=scales, img_size=img_size, n_slices=n_z_slices,
            )

            img_metrics = compute_image_metrics(img_pred, img_measured)
            print(f"  Image metrics: PSNR={img_metrics['psnr_db']:.2f} dB  "
                  f"SSIM={img_metrics['ssim']:.4f}  NRMSE={img_metrics['nrmse']:.4f}")
    except Exception as e:
        print(f"  NUFFT reconstruction failed: {e}")

    result = {
        "name": name,
        "label": label,
        "n_params": n_params,
        "best_val_loss": best_val_loss,
        "best_step": best_step,
        "overfit_onset_step": overfit_onset_step,
        "total_time_s": total_time,
        "mean_step_time_ms": mean_step_time * 1000,
        "steps": steps,
        "lr": lr,
        "weight_decay": weight_decay,
        "loss_type": loss_type,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_steps": val_steps,
        "per_spoke_mse": spoke_results["per_spoke"],
        "per_spoke_mean": spoke_results["mean"],
        "per_spoke_std": spoke_results["std"],
        "worst_spoke": spoke_results["worst_spoke"],
        "best_spoke": spoke_results["best_spoke"],
        **{f"img_{k}": v for k, v in img_metrics.items()},
    }

    return result


# ---------------------------------------------------------------------------
# Summary and plotting
# ---------------------------------------------------------------------------

def print_summary_table(results):
    """Print ASCII summary table."""
    print(f"\n{'='*100}")
    print(f"  ABLATION SUMMARY")
    print(f"{'='*100}")

    header = (
        f"{'Config':<42s} | {'Params':>8s} | {'Val Loss':>10s} | "
        f"{'PSNR':>7s} | {'SSIM':>6s} | {'Overfit':>7s} | {'ms/step':>7s}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        overfit = str(r["overfit_onset_step"]) if r["overfit_onset_step"] else "none"
        psnr_str = f"{r.get('img_psnr_db', 0):.2f}" if "img_psnr_db" in r else "--"
        ssim_str = f"{r.get('img_ssim', 0):.4f}" if "img_ssim" in r else "--"
        print(
            f"{r['label']:<42s} | {r['n_params']:>8,d} | {r['best_val_loss']:>10.3e} | "
            f"{psnr_str:>7s} | {ssim_str:>6s} | {overfit:>7s} | {r['mean_step_time_ms']:>7.2f}"
        )
    print()


def plot_training_curves(results, output_dir):
    """Create comparison training curve plot."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Val loss curves ---
    ax = axes[0]
    for r in results:
        ax.semilogy(r["val_steps"], r["val_losses"], label=r["name"], alpha=0.85)
    ax.set_xlabel("Step")
    ax.set_ylabel("Validation MSE (log)")
    ax.set_title("Validation Loss Curves")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Per-spoke MSE bar chart ---
    ax = axes[1]
    n_models = len(results)
    # Get union of all spoke IDs
    all_spokes = sorted(set().union(*(r["per_spoke_mse"].keys() for r in results)))
    n_spokes = len(all_spokes)
    bar_width = 0.8 / n_models
    x_pos = np.arange(n_spokes)

    for i, r in enumerate(results):
        mse_vals = [r["per_spoke_mse"].get(sp, 0) for sp in all_spokes]
        ax.bar(x_pos + i * bar_width, mse_vals, bar_width,
               label=r["name"], alpha=0.75)

    ax.set_xlabel("Spoke index")
    ax.set_ylabel("MSE")
    ax.set_title("Per-Spoke Validation MSE")
    ax.legend(fontsize=7)
    ax.set_xticks(x_pos + bar_width * (n_models - 1) / 2)
    ax.set_xticklabels([str(s) for s in all_spokes], fontsize=6, rotation=45)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig_path = output_dir / "ablation_curves.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved training curves to {fig_path}")


def save_results_json(results, output_dir):
    """Save results to JSON (strip large arrays for readability)."""
    # Make a compact version without full loss histories
    compact = []
    for r in results:
        c = {k: v for k, v in r.items()
             if k not in ("train_losses", "val_losses", "val_steps")}
        # Convert any non-serializable types
        for k, v in c.items():
            if isinstance(v, (np.integer,)):
                c[k] = int(v)
            elif isinstance(v, (np.floating,)):
                c[k] = float(v)
        compact.append(c)

    json_path = output_dir / "ablation_results.json"
    with open(json_path, "w") as f:
        json.dump(compact, f, indent=2)
    print(f"Saved results to {json_path}")

    # Also save full histories for later analysis
    full_path = output_dir / "ablation_results_full.json"
    full = []
    for r in results:
        fr = {}
        for k, v in r.items():
            if isinstance(v, (np.integer,)):
                fr[k] = int(v)
            elif isinstance(v, (np.floating,)):
                fr[k] = float(v)
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], (float, int)):
                fr[k] = v
            elif isinstance(v, dict):
                fr[k] = {str(kk): vv for kk, vv in v.items()}
            else:
                fr[k] = v
        full.append(fr)

    with open(full_path, "w") as f:
        json.dump(full, f, indent=2)
    print(f"Saved full results to {full_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Polar k-space ablation study")
    parser.add_argument("--data-file", type=str,
                        default="/scratch/rnga/vvpshenov/XCAT-ERIC/results/simulation_results_20260115T150400.mat")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None,
                        help="Force device (default: auto)")
    parser.add_argument("--output-dir", type=str,
                        default="/scratch/rnga/vvpshenov/DCE_NIK/runs/polar_ablation")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data once
    data = load_data(args.data_file)

    # Run all configs
    configs = make_configs()
    results = []

    for cfg in configs:
        result = train_one(
            cfg, data,
            steps=args.steps,
            batch_size=args.batch_size,
            eval_every=args.eval_every,
            seed=args.seed,
            device=device,
        )
        results.append(result)
        # Save incrementally in case of crash
        save_results_json(results, output_dir)

    # Final outputs
    print_summary_table(results)
    plot_training_curves(results, output_dir)
    save_results_json(results, output_dir)

    print(f"\nDone. Results in {output_dir}")


if __name__ == "__main__":
    main()
