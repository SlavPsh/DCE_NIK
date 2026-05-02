#!/usr/bin/env python
"""multicoil cart trainer, sense recon"""
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

from utils.io_utils import load_config, setup_logging, unique_output_dir, copy_config_to_output
from utils.wandb_utils import WandbLogger

from nik_io import load_event, load_cartesian_kspace
from nik_model import WIRE_KXY_REIM
from nik_train import prepare_tensors
from kspace_normalization import compute_dcf_radial, compute_radius, KSpaceNormalizer
from losses import weighted_complex_mse
from nik_recon import (
    ifft1d_kz_to_z,
    ifft1d_kz_to_z_cartesian,
    make_multicoil_radial_dataset,
    make_multicoil_cartesian_dataset,
    coil_combine_rss,
    nufft2d_recon,
)
from nik_metrics import compute_image_metrics, compute_perceptual_metrics


def load_data(config):
    """multicoil radial, cart"""
    data_cfg = config['data']
    t_frame = data_cfg['t_frame']
    z_slice_raw = data_cfg['z_slice_idx']
    subsample_frac = data_cfg.get('subsample_frac', 1.0)
    seed = config['training']['seed']
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # radial
    print("Loading radial data ...", flush=True)
    event = load_event(data_cfg['radial_file'], load_images=True, load_coil_maps=True)
    k_np = np.transpose(event["k"], (0, 2, 1, 3))
    traj_np = np.transpose(event["traj"], (0, 2, 1, 3))
    T, S, C, RO = k_np.shape
    print(f"Radial: T={T}, S={S}, C={C}, RO={RO}", flush=True)

    k_t, traj_t, scales, dims, k_scale = prepare_tensors(k_np, traj_np, data_device=device)
    k_img_space, n_z_slices, n_ro_per_slice, _ = ifft1d_kz_to_z(k_t, traj_t, t_frame=t_frame)
    z_slice_idx = n_z_slices // 2 if z_slice_raw == -1 else int(z_slice_raw)

    # multicoil radial dataset
    x_all, y_all_raw, spoke_id_all, ro_id_all, meta = make_multicoil_radial_dataset(
        k_img_space, traj_t, scales, dims,
        t_fixed=t_frame, z_slice_idx=z_slice_idx,
        n_slices=n_z_slices, compute_device=device,
    )
    print(f"Radial multicoil: {meta['N']} points, {C} coils", flush=True)

    # subsample spokes
    n_unique = int(spoke_id_all.max().item()) + 1
    n_train = max(1, int(n_unique * subsample_frac))
    g = torch.Generator(device=spoke_id_all.device).manual_seed(seed)
    perm = torch.randperm(n_unique, generator=g, device=spoke_id_all.device)
    train_idx = torch.where(torch.isin(spoke_id_all, perm[:n_train]))[0]
    print(f"Spokes: {n_train}/{n_unique} ({subsample_frac:.0%}), {train_idx.shape[0]} points", flush=True)

    # cartesian
    print("Loading Cartesian data ...", flush=True)
    cart_event = load_cartesian_kspace(data_cfg['cartesian_file'], load_images=True, load_coil_maps=True)
    k_cart_t = torch.from_numpy(cart_event['k_cart'].astype(np.complex64)).to(device)
    k_cart_z = ifft1d_kz_to_z_cartesian(k_cart_t)
    z_slice_cart = k_cart_z.shape[2] // 2 if z_slice_raw == -1 else int(z_slice_raw)

    x_cart, y_cart_raw, meta_cart = make_multicoil_cartesian_dataset(
        k_cart_z, t_fixed=min(t_frame, k_cart_z.shape[0]-1),
        z_slice_idx=z_slice_cart, scales_radial=scales, compute_device=device,
    )
    print(f"Cart multicoil: {meta_cart['N']} points, {meta_cart['n_coils']} coils", flush=True)

    # coil maps
    coil_maps_rad = event.get("coil_maps")
    coil_maps_cart = cart_event.get("coil_maps")

    return {
        "x_all": x_all, "y_all_raw": y_all_raw,
        "spoke_id_all": spoke_id_all, "ro_id_all": ro_id_all,
        "train_idx": train_idx,
        "k_img_space": k_img_space, "traj_t": traj_t,
        "T": T, "S": S, "C": C, "RO": RO,
        "n_ro_per_slice": n_ro_per_slice,
        "z_slice_idx": z_slice_idx, "n_z_slices": n_z_slices,
        "scales": scales, "dims": dims,
        "x_cart": x_cart, "y_cart_raw": y_cart_raw,
        "meta_cart": meta_cart,
        "coil_maps_rad": coil_maps_rad,
        "coil_maps_cart": coil_maps_cart,
        "z_slice_cart": z_slice_cart,
        "subsample_frac": subsample_frac,
        "n_unique_spokes": n_unique,
    }


def main(config_path, data):
    """single multicoil run"""
    random.seed()
    run_name = generate_slug(3) + "_mc_carteval"
    config = load_config(config_path)
    output_dir = unique_output_dir(config, run_name)
    copy_config_to_output(config_path, output_dir)
    setup_logging(config, output_dir)
    logging.info(f"Run: {run_name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = config['training']['seed']
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    # wandb early init
    mc_default = config['model']
    tc = config['training']
    init_config = {
        "hidden": mc_default['hidden'], "depth": mc_default['depth'],
        "w0": mc_default['w0'], "s0": mc_default['s0'],
        "dropout": mc_default.get('dropout', 0.0),
        "lr": tc['lr'], "weight_decay": tc.get('weight_decay', 0),
        "subsample_frac": data["subsample_frac"],
    }
    wandb_logger = WandbLogger(config=init_config, output_dir=output_dir,
                               run_name=run_name, job_type="multicoil_cart_eval")
    wandb_logger.initialize()
    wc = wandb.config

    # sweep overrides
    hidden = int(getattr(wc, "hidden", mc_default['hidden']))
    depth = int(getattr(wc, "depth", mc_default['depth']))
    w0 = float(getattr(wc, "w0", mc_default['w0']))
    s0 = float(getattr(wc, "s0", mc_default['s0']))
    dropout = float(getattr(wc, "dropout", mc_default.get('dropout', 0.0)))
    lr = float(getattr(wc, "lr", tc['lr']))
    weight_decay = float(getattr(wc, "weight_decay", tc.get('weight_decay', 0)))
    subsample_frac = float(getattr(wc, "subsample_frac", data["subsample_frac"]))

    # unpack
    x_all = data["x_all"]
    y_all_raw = data["y_all_raw"]
    spoke_id_all = data["spoke_id_all"]
    train_idx = data["train_idx"]
    C = data["C"]; RO = data["RO"]
    x_cart = data["x_cart"]
    y_cart_raw = data["y_cart_raw"]
    meta_cart = data["meta_cart"]
    nky, nkx = meta_cart["nky"], meta_cart["nkx"]

    # resubsample if needed
    if subsample_frac != data["subsample_frac"]:
        n_unique = data["n_unique_spokes"]
        n_train = max(1, int(n_unique * subsample_frac))
        g = torch.Generator(device=spoke_id_all.device).manual_seed(seed)
        perm = torch.randperm(n_unique, generator=g, device=spoke_id_all.device)
        train_spokes = perm[:n_train]
        train_mask = torch.isin(spoke_id_all, train_spokes)
        train_idx = torch.where(train_mask)[0]
        logging.info(f"Re-subsampled: {n_train}/{n_unique} spokes ({subsample_frac:.0%})")

    # normalization
    norm_cfg = config.get('normalization', {})
    use_envelope = norm_cfg.get('use_envelope', True)
    dcf_power = norm_cfg.get('dcf_power', 0.0)
    kcoords = x_all[:, :2]
    dcf = compute_dcf_radial(kcoords) if norm_cfg.get('use_dcf', True) else torch.ones(kcoords.shape[0], device=kcoords.device)

    kcoords_train = kcoords[train_idx]
    y_train_raw = y_all_raw[train_idx]
    dcf_train_for_norm = dcf[train_idx]

    normalizer = KSpaceNormalizer()
    if use_envelope:
        normalizer.fit(kcoords_train, y_train_raw, dcf=dcf_train_for_norm,
                       envelope_bins=norm_cfg.get('envelope_bins', 128),
                       envelope_smooth_width=norm_cfg.get('envelope_smooth_width', 5),
                       envelope_floor_fraction=norm_cfg.get('envelope_floor_fraction', 1e-3))
    else:
        from kspace_normalization import compute_global_scale, _to_complex, _rss_magnitude, RadialEnvelope
        y_c = _to_complex(y_train_raw)
        mag = _rss_magnitude(y_c)
        normalizer.global_scale = compute_global_scale(mag.to(torch.complex64), dcf=dcf_train_for_norm)
        r_max = float(compute_radius(kcoords_train).max().item())
        normalizer.envelope = RadialEnvelope(
            bin_centers=torch.linspace(0, r_max, 128),
            raw_shell_values=torch.ones(128), smoothed_shell_values=torch.ones(128),
            floor_value=1.0, r_max=r_max, statistic="flat", smooth_method="none")
        normalizer._fitted = True

    y_all = normalizer.normalize(kcoords, y_all_raw)
    y_cart = normalizer.normalize(x_cart[:, :2], y_cart_raw)
    logging.info(f"Normalizer: global_scale={normalizer.global_scale:.4f}, C={C}")

    # build model
    model = WIRE_KXY_REIM(
        in_dim=2, hidden=hidden, depth=depth,
        w0=w0, s0=s0, out_dim=2*C, dropout=dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model: WIRE multicoil h={hidden} d={depth} w0={w0} s0={s0} out_dim={2*C} params={n_params}")

    wandb.config.update({
        "n_coils": C, "n_params": n_params,
        "subsample_frac_actual": subsample_frac,
        "use_envelope": use_envelope, "dcf_power": dcf_power,
    }, allow_val_change=True)

    # optimizer
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = None
    if tc.get('scheduler_patience', 0) > 0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=tc['scheduler_factor'],
            patience=tc['scheduler_patience'], min_lr=tc['scheduler_min_lr'])

    # prepare tensors
    x_all_2d = x_all[:, :2].to(device)
    y_all_dev = y_all.to(device)
    x_train = x_all_2d[train_idx]
    y_train = y_all_dev[train_idx]
    N_train = x_train.shape[0]
    dcf_dev = dcf.to(device)
    dcf_train = dcf_dev[train_idx]
    x_cart_dev = x_cart.to(device)
    y_cart_dev = y_cart.to(device)

    # heldout spokes
    train_spoke_set = torch.unique(spoke_id_all[train_idx])
    heldout_mask = ~torch.isin(spoke_id_all, train_spoke_set)
    heldout_idx = torch.where(heldout_mask)[0]
    has_heldout = heldout_idx.numel() > 0
    if has_heldout:
        x_heldout = x_all_2d[heldout_idx]
        y_heldout = y_all_dev[heldout_idx]

    # cart coil maps
    coil_maps = data["coil_maps_cart"]
    if coil_maps is not None:
        z_cart = data["z_slice_cart"]
        # cart shape match
        sens = coil_maps[:, z_cart, :, :].astype(np.complex64)
        logging.info(f"Coil sensitivity maps: {sens.shape}")
    else:
        sens = None
        logging.warning("No coil sensitivity maps — will use RSS combination")

    # training
    steps = tc['steps']
    batch_size = tc['batch_size']
    grad_clip = tc['grad_clip']
    eval_every = tc['eval_every']
    warmup_steps = tc.get('warmup_steps', steps // 5)
    plot_every = tc['plot']['plot_every']
    console_every = config['logging']['console_every']

    model.train()
    best_cart_loss = float("inf")
    best_state = None
    last_cart_loss = last_heldout_loss = None

    logging.info(f"Training {steps} steps, {N_train} points, {C} coils")

    for step in range(1, steps + 1):
        idx = torch.randint(0, N_train, (batch_size,), device=device)
        x = x_train[idx]; y = y_train[idx]; w = dcf_train[idx]

        opt.zero_grad(set_to_none=True)
        y_pred = model(x)
        loss = weighted_complex_mse(y_pred, y, weights=w, power=dcf_power)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        train_loss = float(loss.item())

        if step % eval_every == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                cart_pred = model(x_cart_dev)
                last_cart_loss = float(F.mse_loss(cart_pred, y_cart_dev).item())
                if has_heldout:
                    last_heldout_loss = float(F.mse_loss(model(x_heldout), y_heldout).item())
            model.train()

            if step >= warmup_steps and last_cart_loss < best_cart_loss:
                best_cart_loss = last_cart_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if scheduler:
                scheduler.step(last_cart_loss)

        log_dict = {"train/train_loss": train_loss}
        if step % eval_every == 0 or step == steps:
            log_dict["train/cart_eval_loss"] = last_cart_loss
            if last_heldout_loss is not None:
                log_dict["train/heldout_loss"] = last_heldout_loss
        if scheduler:
            log_dict["train/lr"] = opt.param_groups[0]["lr"]
        wandb_logger.log(log_dict, step=step)

        if step % console_every == 0:
            msg = f"step {step:6d}  train {train_loss:.3e}"
            if last_cart_loss is not None: msg += f"  cart {last_cart_loss:.3e}"
            if last_heldout_loss is not None: msg += f"  heldout {last_heldout_loss:.3e}"
            logging.info(msg)

    # restore best, eval
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    try:
        with torch.no_grad():
            # predict, denormalize
            cart_pred_norm = model(x_cart_dev)
            cart_pred_denorm = normalizer.denormalize(x_cart[:, :2].to(device), cart_pred_norm)
            y_meas_denorm = normalizer.denormalize(x_cart[:, :2].to(device), y_cart_dev)

        # per coil ifft
        pred_coil_imgs = []
        meas_coil_imgs = []
        for c in range(C):
            # predicted
            k_pred_c = torch.complex(
                cart_pred_denorm[:, 2*c], cart_pred_denorm[:, 2*c+1]
            ).reshape(nky, nkx)
            img_pred_c = torch.fft.fftshift(torch.fft.ifft2(k_pred_c)).cpu().numpy().T
            pred_coil_imgs.append(img_pred_c)

            # measured
            k_meas_c = torch.complex(
                y_meas_denorm[:, 2*c], y_meas_denorm[:, 2*c+1]
            ).reshape(nky, nkx)
            img_meas_c = torch.fft.ifft2(k_meas_c).cpu().numpy().T
            meas_coil_imgs.append(img_meas_c)

        pred_coil_imgs = np.array(pred_coil_imgs)
        meas_coil_imgs = np.array(meas_coil_imgs)

        # coil combine
        if sens is not None:
            assert sens.shape == pred_coil_imgs.shape, \
                f"sens shape {sens.shape} != coil imgs {pred_coil_imgs.shape}"
            # sense formula
            denom = np.sum(np.abs(sens) ** 2, axis=0) + 1e-10
            img_pred_combined = np.abs(np.sum(np.conj(sens) * pred_coil_imgs, axis=0) / denom)
            img_meas_combined = np.abs(np.sum(np.conj(sens) * meas_coil_imgs, axis=0) / denom)
        else:
            img_pred_combined = coil_combine_rss(pred_coil_imgs)
            img_meas_combined = coil_combine_rss(meas_coil_imgs)

        # peak normalize
        img_pred_n = img_pred_combined / (img_pred_combined.max() or 1.0)
        img_meas_n = img_meas_combined / (img_meas_combined.max() or 1.0)

        # log images
        wandb_logger.log({
            "recon/model_combined": wandb.Image(img_pred_n),
            "recon/ref_combined": wandb.Image(img_meas_n),
        }, step=steps)

        # model vs reference
        m_metrics = compute_image_metrics(img_pred_n, img_meas_n)
        m_perceptual = compute_perceptual_metrics(img_pred_n, img_meas_n)
        wandb_logger.log({
            "model_vs_ref/psnr": m_metrics["psnr_db"],
            "model_vs_ref/ssim": m_metrics["ssim"],
            "model_vs_ref/dists": m_perceptual["DISTS"],
            "model_vs_ref/haarpsi": m_perceptual["HaarPSI"],
            "model_vs_ref/vsi": m_perceptual["VSI"],
        }, step=steps)
        wandb.run.summary.update({
            "model_psnr": m_metrics["psnr_db"], "model_ssim": m_metrics["ssim"],
            "model_dists": m_perceptual["DISTS"], "model_haarpsi": m_perceptual["HaarPSI"],
        })
        logging.info(
            f"Model vs Ref (combined):  PSNR={m_metrics['psnr_db']:.2f}  "
            f"SSIM={m_metrics['ssim']:.4f}  DISTS={m_perceptual['DISTS']:.4f}  "
            f"HaarPSI={m_perceptual['HaarPSI']:.4f}"
        )

        # nufft baseline
        k_img_sub = torch.zeros_like(data["k_img_space"])
        train_spoke_ids = torch.unique(spoke_id_all[train_idx])
        for sp_id in train_spoke_ids:
            k_img_sub[:, sp_id, :, :, :] = data["k_img_space"][:, sp_id, :, :, :]

        _t = config['data']['t_frame']
        _z = data["z_slice_idx"]
        nufft_coil_imgs = []
        for c in range(C):
            img_c = nufft2d_recon(
                k_img_sub, data["traj_t"],
                t_frame=_t, coil_idx=c, z_slice_idx=_z,
                scales=data["scales"], img_size=(312, 312), n_slices=data["n_z_slices"],
                return_complex=True,
            )
            nufft_coil_imgs.append(img_c)
        nufft_coil_imgs = np.array(nufft_coil_imgs)

        # center crop fov
        gt_h, gt_w = img_meas_combined.shape
        nh, nw = nufft_coil_imgs.shape[1], nufft_coil_imgs.shape[2]
        if nh != gt_h or nw != gt_w:
            y0 = (nh - gt_h) // 2; x0 = (nw - gt_w) // 2
            nufft_coil_imgs = nufft_coil_imgs[:, y0:y0+gt_h, x0:x0+gt_w]

        if sens is not None:
            assert sens.shape == nufft_coil_imgs.shape, \
                f"sens {sens.shape} != nufft coil imgs {nufft_coil_imgs.shape}"
            denom = np.sum(np.abs(sens) ** 2, axis=0) + 1e-10
            img_nufft_combined = np.abs(np.sum(np.conj(sens) * nufft_coil_imgs, axis=0) / denom)
        else:
            img_nufft_combined = coil_combine_rss(nufft_coil_imgs)

        img_nufft_n = img_nufft_combined / (img_nufft_combined.max() or 1.0)

        n_metrics = compute_image_metrics(img_nufft_n, img_meas_n)
        n_perceptual = compute_perceptual_metrics(img_nufft_n, img_meas_n)
        wandb_logger.log({
            "nufft_vs_ref/psnr": n_metrics["psnr_db"],
            "nufft_vs_ref/ssim": n_metrics["ssim"],
            "nufft_vs_ref/dists": n_perceptual["DISTS"],
            "nufft_vs_ref/haarpsi": n_perceptual["HaarPSI"],
            "recon/nufft_combined": wandb.Image(img_nufft_n),
        }, step=steps)

        # delta
        d_psnr = m_metrics["psnr_db"] - n_metrics["psnr_db"]
        d_dists = n_perceptual["DISTS"] - m_perceptual["DISTS"]
        wandb.run.summary.update({
            "nufft_psnr": n_metrics["psnr_db"], "nufft_dists": n_perceptual["DISTS"],
            "delta_psnr": d_psnr, "delta_dists": d_dists,
        })
        logging.info(
            f"NUFFT vs Ref (RSS):  PSNR={n_metrics['psnr_db']:.2f}  "
            f"DISTS={n_perceptual['DISTS']:.4f}"
        )
        logging.info(f"Delta: dPSNR={d_psnr:+.2f}  dDISTS={d_dists:+.4f}")

    except Exception as e:
        logging.warning(f"Final evaluation failed: {e}")
        import traceback; traceback.print_exc()

    wandb_logger.save_model(model, "model_best.pth", opt, steps, output_dir)
    wandb.run.summary.update({"cart_eval_loss": best_cart_loss, "total_steps": steps})
    logging.info(f"Done. best_cart_eval_loss={best_cart_loss:.3e}")
    wandb_logger.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    parser.add_argument('--single', action='store_true')
    args = parser.parse_args()

    config = load_config(args.config_path)
    data = load_data(config)

    if args.single:
        main(args.config_path, data)
    else:
        sweep_config = config.get('sweep', {})
        sweep_id = wandb.sweep(sweep=sweep_config)
        wandb.agent(sweep_id, function=lambda: main(args.config_path, data),
                    count=sweep_config.get('run_cap', 50))
