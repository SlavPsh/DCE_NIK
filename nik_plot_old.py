import matplotlib.pyplot as plt
import numpy as np
import torch

from nik_recon import ifft1d_kz_to_z, nufft2d_recon

@torch.no_grad()
def plot_error_maps_kxy(
    model,
    *,
    kxy_sp, y_sp,
    spokes,
    y_scale=1.0,
    max_points=250_000,
    title_prefix="",
):
    device = next(model.parameters()).device
    model.eval()

    spokes = spokes.to(kxy_sp.device)
    x = kxy_sp[spokes].reshape(-1, 2).to(device)
    y = y_sp[spokes].reshape(-1, 2).to(device)

    N = x.shape[0]
    if N > max_points:
        idx = torch.randperm(N, device=device)[:max_points]
        x = x[idx]
        y = y[idx]

    yp = model(x)

    ys = float(y_scale.detach().cpu().item()) if isinstance(y_scale, torch.Tensor) else float(y_scale)
    y0 = (y * ys).detach().cpu().numpy()
    yp0 = (yp * ys).detach().cpu().numpy()

    kx = x[:, 0].detach().cpu().numpy()
    ky = x[:, 1].detach().cpu().numpy()

    dRe = yp0[:, 0] - y0[:, 0]
    dIm = yp0[:, 1] - y0[:, 1]
    mag_y  = np.sqrt(y0[:,0]**2 + y0[:,1]**2)
    mag_yp = np.sqrt(yp0[:,0]**2 + yp0[:,1]**2)
    dMag = mag_yp - mag_y

    def scatter(ax, c, ttl):
        sc = ax.scatter(kx, ky, c=np.log1p(np.abs(c)), s=2)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(ttl)
        ax.set_xlabel("kx"); ax.set_ylabel("ky")
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    scatter(ax[0], dRe,  f"{title_prefix} log1p(|ΔRe|)")
    scatter(ax[1], dIm,  f"{title_prefix} log1p(|ΔIm|)")
    scatter(ax[2], dMag, f"{title_prefix} log1p(|Δ|y||)")
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def plot_spoke_zoom_kxy_dense(
    model,
    *,
    kxy_sp, y_sp,
    spoke_id: int,
    y_scale=1.0,
    n_s: int = 4096,        # dense samples along the spoke ordering
    title_prefix="",
):
    """
    Plot in the native spoke ordering: RO index increases from one edge,
    through the center (~mid index), to the opposite edge.

    Dense curve is constructed by interpolating (kx,ky) along RO.
    """
    device = next(model.parameters()).device
    model.eval()

    # measured spoke (RO,2) and measurements (RO,2)
    x_m = kxy_sp[spoke_id].to(device)    # (RO,2)
    y_m = y_sp[spoke_id].to(device)      # (RO,2)
    RO = x_m.shape[0]

    ys = float(y_scale.detach().cpu().item()) if isinstance(y_scale, torch.Tensor) else float(y_scale)
    y_m0 = (y_m * ys).detach().cpu().numpy()  # (RO,2)

    # x-axis: native RO index
    idx_m = np.arange(RO)

    # Dense parameter s in [0, RO-1] following the same ordering
    s = torch.linspace(0, RO - 1, n_s, device=device)  # (n_s,)
    s0 = torch.floor(s).long().clamp(0, RO - 1)
    s1 = (s0 + 1).clamp(0, RO - 1)
    w = (s - s0.float()).unsqueeze(1)  # (n_s,1)

    # interpolate kx,ky along the trajectory in index-space
    x0 = x_m[s0]  # (n_s,2)
    x1 = x_m[s1]  # (n_s,2)
    x_d = (1.0 - w) * x0 + w * x1       # (n_s,2)

    # predict dense along spoke
    yp_d = model(x_d)                   # (n_s,2)
    yp_d0 = (yp_d * ys).detach().cpu().numpy()

    # components
    re_m, im_m = y_m0[:, 0], y_m0[:, 1]
    mag_m = np.sqrt(re_m**2 + im_m**2)

    re_p, im_p = yp_d0[:, 0], yp_d0[:, 1]
    mag_p = np.sqrt(re_p**2 + im_p**2)

    s_np = s.detach().cpu().numpy()

    # plots
    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    ax[0].plot(s_np, re_p, label="pred Re (dense)")
    ax[0].scatter(idx_m, re_m, s=10, label="meas Re (RO)")
    ax[0].axvline(RO//2, linestyle="--", linewidth=1)
    ax[0].set_title(f"{title_prefix} spoke {spoke_id}: Re vs RO index")
    ax[0].legend()

    ax[1].plot(s_np, im_p, label="pred Im (dense)")
    ax[1].scatter(idx_m, im_m, s=10, label="meas Im (RO)")
    ax[1].axvline(RO//2, linestyle="--", linewidth=1)
    ax[1].set_title(f"{title_prefix} spoke {spoke_id}: Im vs RO index")
    ax[1].legend()

    ax[2].plot(s_np, mag_p, label="pred |y| (dense)")
    ax[2].scatter(idx_m, mag_m, s=10, label="meas |y| (RO)")
    ax[2].axvline(RO//2, linestyle="--", linewidth=1)
    ax[2].set_title(f"{title_prefix} spoke {spoke_id}: |y| vs RO index")
    ax[2].set_xlabel("RO index (edge → center → edge)")
    ax[2].legend()

    plt.tight_layout()
    plt.show()


@torch.no_grad()
def plot_rings_kxy_train_vs_val_dense(
    model,
    *,
    kxy_sp, y_sp, theta,
    train_spokes, val_spokes,
    ro_list,                 # list of ro_idx values, e.g. [RO//4, RO//2, int(0.8*RO)]
    y_scale=1.0,
    n_theta: int = 1024,     # dense in angle
    title_prefix="",
):
    device = next(model.parameters()).device
    model.eval()

    ys = float(y_scale.detach().cpu().item()) if isinstance(y_scale, torch.Tensor) else float(y_scale)

    def get_points(spokes, ro_idx):
        spokes = spokes.to(kxy_sp.device)
        th_m = theta[spokes].detach().cpu().numpy()
        y_m = (y_sp[spokes, ro_idx, :].detach().cpu().numpy()) * ys
        o = np.argsort(th_m)
        th_m = th_m[o]
        re_m, im_m = y_m[o,0], y_m[o,1]
        mag_m = np.sqrt(re_m**2 + im_m**2)
        return th_m, re_m, im_m, mag_m

    for ro_idx in ro_list:
        # radius estimate from median across all spokes at this ro_idx
        r_all = torch.sqrt((kxy_sp[:, ro_idx, 0]**2 + kxy_sp[:, ro_idx, 1]**2)).detach().cpu().numpy()
        r0 = float(np.median(r_all))

        # dense prediction around ring
        th = torch.linspace(-np.pi, np.pi, n_theta, device=device)
        x_dense = torch.stack([torch.tensor(r0, device=device) * torch.cos(th),
                               torch.tensor(r0, device=device) * torch.sin(th)], dim=1).float()
        yp = (model(x_dense).detach().cpu().numpy()) * ys
        re_p, im_p = yp[:,0], yp[:,1]
        mag_p = np.sqrt(re_p**2 + im_p**2)
        th_np = th.detach().cpu().numpy()

        # measured points on that ring for train/val spokes
        th_tr, re_tr, im_tr, mag_tr = get_points(train_spokes, ro_idx)
        th_va, re_va, im_va, mag_va = get_points(val_spokes, ro_idx)

        # Re
        plt.figure(figsize=(10,4))
        plt.plot(th_np, re_p, label="pred Re (dense ring)")
        plt.scatter(th_tr, re_tr, s=18, label="train points Re")
        plt.scatter(th_va, re_va, s=18, label="val points Re")
        plt.title(f"{title_prefix} ro={ro_idx} r={r0:.3f}: Re vs theta")
        plt.xlabel("theta [rad]")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Im
        plt.figure(figsize=(10,4))
        plt.plot(th_np, im_p, label="pred Im (dense ring)")
        plt.scatter(th_tr, im_tr, s=18, label="train points Im")
        plt.scatter(th_va, im_va, s=18, label="val points Im")
        plt.title(f"{title_prefix} ro={ro_idx} r={r0:.3f}: Im vs theta")
        plt.xlabel("theta [rad]")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Mag
        plt.figure(figsize=(10,4))
        plt.plot(th_np, mag_p, label="pred |y| (dense ring)")
        plt.scatter(th_tr, mag_tr, s=18, label="train points |y|")
        plt.scatter(th_va, mag_va, s=18, label="val points |y|")
        plt.title(f"{title_prefix} ro={ro_idx} r={r0:.3f}: |y| vs theta")
        plt.xlabel("theta [rad]")
        plt.legend()
        plt.tight_layout()
        plt.show()


def make_plot_callback_all(
    *,
    kxy_sp, y_sp, theta,
    train_spokes, val_spokes,
    y_scale,
    ro_idx_ring=None,
    train_spoke_show=None,
    val_spoke_show=None,

    # --- add these for recon snapshots ---
    pred_imgs=None,
    pred_steps=None,
    k_img_space=None,
    traj_t=None,
    t_frame=None,
    coil_idx=None,
    z_slice_idx=None,
    scales=None,
    img_size=None,
    n_z_slices=None,
    n_ro_per_slice=None,   # number of spokes in this kz plane (S_kz)
    RO_full=None,          # RO length
):
    S_kz, RO, _ = kxy_sp.shape
    if ro_idx_ring is None:
        ro_idx_ring = RO // 2

    if train_spoke_show is None:
        train_spoke_show = int(train_spokes[0].item())
    if val_spoke_show is None:
        val_spoke_show = int(val_spokes[0].item())

    do_recon = (
        pred_imgs is not None and pred_steps is not None and
        k_img_space is not None and traj_t is not None and
        t_frame is not None and coil_idx is not None and z_slice_idx is not None and
        scales is not None and img_size is not None and n_z_slices is not None
    )

    def cb(step, model):
        plot_spoke_zoom_kxy_dense(
            model,
            kxy_sp=kxy_sp, y_sp=y_sp,
            spoke_id=train_spoke_show,
            y_scale=y_scale,
            n_s=4096,
            title_prefix=f"[train] step {step}",
        )

        plot_spoke_zoom_kxy_dense(
            model,
            kxy_sp=kxy_sp, y_sp=y_sp,
            spoke_id=val_spoke_show,
            y_scale=y_scale,
            n_s=4096,
            title_prefix=f"[val] step {step}",
        )

        RO_local = kxy_sp.shape[1]
        plot_rings_kxy_train_vs_val_dense(
            model,
            kxy_sp=kxy_sp, y_sp=y_sp, theta=theta,
            train_spokes=train_spokes, val_spokes=val_spokes,
            ro_list=[RO_local // 4, RO_local // 2, int(0.8 * RO_local)],
            y_scale=y_scale,
            n_theta=1024,
            title_prefix=f"step {step}",
        )

        if not do_recon:
            return cb

        # Build predicted k-space for this (t_frame, coil_idx, z_slice_idx) plane

        model.eval()
        with torch.no_grad():
            device = next(model.parameters()).device

            # kxy_sp is (S_kz, RO, 2) for the selected kz plane
            x_all_2d = kxy_sp.reshape(-1, 2).to(device)  # (S_kz*RO, 2)

            yp = model(x_all_2d)  # (N,2) scaled
            ys = float(y_scale.detach().cpu().item()) if isinstance(y_scale, torch.Tensor) else float(y_scale)
            yp = yp * ys  # unscale to original k-space units

            k_pred = torch.complex(yp[:, 0], yp[:, 1]).reshape(S_kz, RO)  # (spokes, RO)

            # create a k-space tensor with only this slice filled
            k_img_space_pred = torch.zeros_like(k_img_space)
            k_img_space_pred[t_frame, :, coil_idx, z_slice_idx, :] = k_pred

            img_pred = nufft2d_recon(
                k_img_space_pred, traj_t,
                t_frame=t_frame, coil_idx=coil_idx, z_slice_idx=z_slice_idx,
                scales=scales, img_size=img_size, n_slices=n_z_slices
            )

            pred_imgs.append(img_pred)
            pred_steps.append(step)

        model.train()

    return cb

