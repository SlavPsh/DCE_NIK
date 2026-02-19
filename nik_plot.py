import numpy as np
import matplotlib.pyplot as plt
import torch

@torch.no_grad()
def plot_error_maps_kxy_flat(
    model,
    *,
    x_sub, y_sub,                # (M,4), (M,2) scaled
    y_scale=1.0,
    max_points=250_000,
    title_prefix="",
    coord_adapter=None,
):
    device = next(model.parameters()).device
    model.eval()

    x = x_sub[:, :2].to(device)
    y = y_sub.to(device)

    M = x.shape[0]
    if M > max_points:
        g = torch.Generator(device=device).manual_seed(0)
        idx = torch.randperm(M, generator=g, device=device)[:max_points]

        x = x[idx]
        y = y[idx]

    xm = coord_adapter(x) if coord_adapter is not None else x
    yp = model(xm)

    ys = float(y_scale.detach().cpu().item()) if torch.is_tensor(y_scale) else float(y_scale)
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
def plot_spoke_zoom_kxy_dense_flat(
    model,
    *,
    x_all, y_all,               # (N,4), (N,2) scaled
    spoke_id_all, ro_id_all,     # (N,), (N,)
    spoke_id: int,
    y_scale=1.0,
    n_s: int = 4096,
    title_prefix="",
    coord_adapter=None,
    log_scale: bool = False,
):
    device = next(model.parameters()).device
    model.eval()

    # select this spoke
    m = (spoke_id_all == int(spoke_id))
    ro = ro_id_all[m]
    x_m = x_all[m][:, :2]
    y_m = y_all[m]

    # sort by RO index (native ordering)
    order = torch.argsort(ro)
    ro = ro[order]
    x_m = x_m[order].to(device)        # (RO,2)
    y_m = y_m[order].to(device)        # (RO,2)
    RO = x_m.shape[0]

    ys = float(y_scale.detach().cpu().item()) if torch.is_tensor(y_scale) else float(y_scale)
    y_m0 = (y_m * ys).detach().cpu().numpy()
    idx_m = ro.detach().cpu().numpy()

    # dense parameter s in [0, RO-1] and interpolate trajectory
    s = torch.linspace(0, RO - 1, n_s, device=device)
    s0 = torch.floor(s).long().clamp(0, RO - 1)
    s1 = (s0 + 1).clamp(0, RO - 1)
    w = (s - s0.float()).unsqueeze(1)

    x0 = x_m[s0]
    x1 = x_m[s1]
    x_d = (1.0 - w) * x0 + w * x1
    x_in = coord_adapter(x_d) if coord_adapter is not None else x_d

    yp_d = model(x_in)
    yp_d0 = (yp_d * ys).detach().cpu().numpy()

    re_m, im_m = y_m0[:, 0], y_m0[:, 1]
    mag_m = np.sqrt(re_m**2 + im_m**2)
    re_p, im_p = yp_d0[:, 0], yp_d0[:, 1]
    mag_p = np.sqrt(re_p**2 + im_p**2)

    s_np = s.detach().cpu().numpy()

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

    if log_scale:
        ax[0].set_yscale("symlog")
        ax[1].set_yscale("symlog")
        ax[2].set_yscale("log")

    plt.tight_layout()
    plt.show()

@torch.no_grad()
def plot_rings_kxy_train_vs_val_dense_flat(
    model,
    *,
    x_all, y_all,
    spoke_id_all, ro_id_all,
    train_idx, val_idx,          # point indices into x_all/y_all
    ro_list,
    y_scale=1.0,
    n_theta: int = 1024,
    title_prefix="",
    coord_adapter=None,
    log_scale: bool = False,
):
    device = next(model.parameters()).device
    model.eval()

    ys = float(y_scale.detach().cpu().item()) if torch.is_tensor(y_scale) else float(y_scale)

    def get_points(point_idx, ro_idx):
        # select subset points at this ring (ro == ro_idx)
        rid = ro_id_all[point_idx]
        m = (rid == int(ro_idx))
        idx = point_idx[m]

        x = x_all[idx][:, :2]              # (K,2)
        y = y_all[idx] * ys                # unscale

        th = torch.atan2(x[:, 1], x[:, 0])
        order = torch.argsort(th)

        y = y[order].detach().cpu().numpy()
        th = th[order].detach().cpu().numpy()

        re = y[:, 0]
        im = y[:, 1]
        mag = np.sqrt(re**2 + im**2)
        return th, re, im, mag

    for ro_idx in ro_list:
        # estimate radius from ALL spokes at this ro_idx (median)
        m_all = (ro_id_all == int(ro_idx))
        x_ring = x_all[m_all][:, :2]
        r0 = torch.median(torch.sqrt(x_ring[:,0]**2 + x_ring[:,1]**2)).item()

        # dense prediction around ring
        th = torch.linspace(-np.pi, np.pi, n_theta, device=device)
        x_dense = torch.stack([
            torch.tensor(r0, device=device) * torch.cos(th),
            torch.tensor(r0, device=device) * torch.sin(th),
        ], dim=1).float()

        x_in = coord_adapter(x_dense) if coord_adapter is not None else x_dense
        
        yp = (model(x_in) * ys).detach().cpu().numpy()
        re_p, im_p = yp[:, 0], yp[:, 1]
        mag_p = np.sqrt(re_p**2 + im_p**2)
        th_np = th.detach().cpu().numpy()

        # measured points at this ring for train/val subsets
        th_tr, re_tr, im_tr, mag_tr = get_points(train_idx, ro_idx)
        th_va, re_va, im_va, mag_va = get_points(val_idx, ro_idx)

        # Re
        plt.figure(figsize=(10,4))
        plt.plot(th_np, re_p, label="pred Re (dense ring)")
        plt.scatter(th_tr, re_tr, s=18, label="train points Re")
        plt.scatter(th_va, re_va, s=18, label="val points Re")
        plt.title(f"{title_prefix} ro={ro_idx} r={r0:.3f}: Re vs theta")
        plt.xlabel("theta [rad]")
        if log_scale:
            plt.yscale("symlog")
        plt.legend(); plt.tight_layout(); plt.show()

        # Im
        plt.figure(figsize=(10,4))
        plt.plot(th_np, im_p, label="pred Im (dense ring)")
        plt.scatter(th_tr, im_tr, s=18, label="train points Im")
        plt.scatter(th_va, im_va, s=18, label="val points Im")
        plt.title(f"{title_prefix} ro={ro_idx} r={r0:.3f}: Im vs theta")
        plt.xlabel("theta [rad]")
        if log_scale:
            plt.yscale("symlog")
        plt.legend(); plt.tight_layout(); plt.show()

        # Mag
        plt.figure(figsize=(10,4))
        plt.plot(th_np, mag_p, label="pred |y| (dense ring)")
        plt.scatter(th_tr, mag_tr, s=18, label="train points |y|")
        plt.scatter(th_va, mag_va, s=18, label="val points |y|")
        plt.title(f"{title_prefix} ro={ro_idx} r={r0:.3f}: |y| vs theta")
        plt.xlabel("theta [rad]")
        if log_scale:
            plt.yscale("log")
        plt.legend(); plt.tight_layout(); plt.show()


def make_plot_callback_all_flat(
    *,
    x_all, y_all,
    spoke_id_all, ro_id_all,
    train_idx, val_idx,
    y_scale,
    ro_list=None,
    train_spoke_show=None,
    val_spoke_show=None,
    coord_adapter=None,
    log_scale: bool = False,
):
    # choose defaults
    if ro_list is None:
        RO = int(ro_id_all.max().item()) + 1
        ro_list = [RO//4, RO//2, int(0.8*RO)]

    if train_spoke_show is None:
        train_spoke_show = int(torch.unique(spoke_id_all[train_idx])[0].item())
    if val_spoke_show is None:
        val_spoke_show = int(torch.unique(spoke_id_all[val_idx])[0].item())

    def cb(step, model):
        with torch.no_grad():
            model.eval()
            # error maps
            plot_error_maps_kxy_flat(model, x_sub=x_all[train_idx], y_sub=y_all[train_idx],
                                     y_scale=y_scale, title_prefix=f"[train] step {step}", coord_adapter=coord_adapter)
            plot_error_maps_kxy_flat(model, x_sub=x_all[val_idx], y_sub=y_all[val_idx],
                                     y_scale=y_scale, title_prefix=f"[val] step {step}", coord_adapter=coord_adapter)
    
            # spoke zoom
            plot_spoke_zoom_kxy_dense_flat(model,
                x_all=x_all, y_all=y_all,
                spoke_id_all=spoke_id_all, ro_id_all=ro_id_all,
                spoke_id=train_spoke_show,
                y_scale=y_scale, n_s=4096,
                title_prefix=f"[train] step {step}",
                coord_adapter=coord_adapter,
                log_scale=log_scale,
            )
            plot_spoke_zoom_kxy_dense_flat(model,
                x_all=x_all, y_all=y_all,
                spoke_id_all=spoke_id_all, ro_id_all=ro_id_all,
                spoke_id=val_spoke_show,
                y_scale=y_scale, n_s=4096,
                title_prefix=f"[val] step {step}",
                coord_adapter=coord_adapter,
                log_scale=log_scale,
            )

            plot_rings_kxy_train_vs_val_dense_flat(
                model,
                x_all=x_all, y_all=y_all,
                spoke_id_all=spoke_id_all, ro_id_all=ro_id_all,
                train_idx=train_idx, val_idx=val_idx,
                ro_list=ro_list,
                y_scale=y_scale, n_theta=1024,
                title_prefix=f"step {step}",
                coord_adapter=coord_adapter,
                log_scale=log_scale,
            )

            model.train()

    return cb

