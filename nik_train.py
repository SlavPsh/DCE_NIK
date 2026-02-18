# nik_train.py
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

from contextlib import nullcontext
import matplotlib.pyplot as plt


@torch.no_grad()
def compute_scale1(y_in):
    return torch.abs(y_in).amax() + 1e-8

@torch.no_grad()
def compute_scale(k_t, q: float = 0.95, probe_size: int = 200_000) -> torch.Tensor:
    """
    Compute quantile scale by sampling from k-space.
    
    Args:
        k_t: complex k-space tensor of any shape
        q: quantile (default 0.95)
        probe_size: number of random samples to draw
    
    Returns:
        scalar tensor: q-th percentile of magnitude
    """
    # Random sampling from full k-space
    shape = k_t.shape
    total_elements = np.prod(shape)
    probe_size = min(probe_size, total_elements)
    
    k_flat = k_t.reshape(-1)
    idx = torch.randperm(k_flat.numel(), device=k_t.device)[:probe_size]
    y_probe = k_flat[idx]
    
    y_abs = torch.abs(y_probe)
    return torch.quantile(y_abs, q) + 1e-8

class EvalEarlyStopper:
    def __init__(self, min_rel_improve=5e-4, patience=10):
        self.min_rel_improve = min_rel_improve
        self.patience = patience
        self.best = float("inf")
        self.bad = 0
        self.last_rel = None

    def step(self, eval_loss: float) -> bool:
        if self.best == float("inf"):
            self.best = eval_loss
            self.bad = 0
            self.last_rel = None
            return False

        rel = (self.best - eval_loss) / max(self.best, 1e-12)
        self.last_rel = rel

        if eval_loss < self.best and rel >= self.min_rel_improve:
            self.best = eval_loss
            self.bad = 0
        else:
            self.bad += 1

        return self.bad >= self.patience

    def postfix(self):
        return {"eval_rel↓": f"{self.last_rel:.2e}" if self.last_rel is not None else "NA",
                "eval_stall": f"{self.bad}/{self.patience}",
                "best_eval": f"{self.best:.2e}"}



class PlateauStopper:
    def __init__(self, window=50, min_rel_improve=5e-4, patience=8):
        self.window = window
        self.min_rel_improve = min_rel_improve
        self.patience = patience
        self.hist = []
        self.bad = 0
        self.last_rel = None

    def step(self, loss):
        self.hist.append(loss)
        if len(self.hist) < 2 * self.window:
            self.last_rel = None
            return False
        a = float(np.mean(self.hist[-2 * self.window:-self.window]))
        b = float(np.mean(self.hist[-self.window:]))
        rel = (a - b) / max(a, 1e-12)
        self.last_rel = rel
        if rel < self.min_rel_improve:
            self.bad += 1
        else:
            self.bad = 0
        return self.bad >= self.patience

    def postfix(self):
        if self.last_rel is None:
            return {"warmup": f"{len(self.hist)}/{2*self.window}", "stall": f"{self.bad}/{self.patience}"}
        return {"rel↓": f"{self.last_rel:.2e}", "stall": f"{self.bad}/{self.patience}"}


def prepare_tensors(k_np, traj_np, data_device="cpu"):
    """
    k_np:    (T,S,C,RO) complex64
    traj_np: (T,S,3,RO) float32

    Returns k_t, traj_t, scales (sx,sy,sz), dims (T,S,C,RO).
    """
    device = torch.device(data_device)
    k_t = torch.from_numpy(k_np)
    traj_t = torch.from_numpy(traj_np)

    if device.type == "cpu":
        k_t = k_t.pin_memory()
        traj_t = traj_t.pin_memory()
    else:
        k_t = k_t.to(device, non_blocking=True)
        traj_t = traj_t.to(device, non_blocking=True)

    # keep as tensors 
    sx = traj_t[:, :, 0, :].abs().amax() + 1e-8
    sy = traj_t[:, :, 1, :].abs().amax() + 1e-8
    sz = traj_t[:, :, 2, :].abs().amax() + 1e-8

    T, S, C, RO = k_np.shape
    k_scale = compute_scale(k_t)
    return k_t, traj_t, (sx, sy, sz), (T, S, C, RO), k_scale


def make_sampler(k_t, traj_t, scales, dims, y_scale, compute_device):
    """
    Returns a callable sample_batch(batch_size) -> (x, coil_idx, y_meas)
    x: (B,4) float, coil_idx: (B,) long, y_meas: (B,2) float
    """
    sx, sy, sz = scales
    T, S, C, RO = dims
    idx_dev = k_t.device
    compute_device = torch.device(compute_device)

    def sample_batch(batch_size: int):
        t_idx  = torch.randint(T, (batch_size,), device=idx_dev)
        s_idx  = torch.randint(S, (batch_size,), device=idx_dev)
        c_idx  = torch.randint(C, (batch_size,), device=idx_dev)
        ro_idx = torch.randint(RO, (batch_size,), device=idx_dev)

        y = k_t[t_idx, s_idx, c_idx, ro_idx]
        y_ri = torch.view_as_real(y).float()  # (B,2)
        y_ri = y_ri / y_scale

        kx = traj_t[t_idx, s_idx, 0, ro_idx] / sx
        ky = traj_t[t_idx, s_idx, 1, ro_idx] / sy
        kz = traj_t[t_idx, s_idx, 2, ro_idx] / sz
        t_norm = (t_idx.to(traj_t.dtype) / (T - 1 + 1e-8)) * 2.0 - 1.0
        x = torch.stack([kx, ky, kz, t_norm], dim=1).float()

        non_block = (idx_dev.type == "cuda" and compute_device.type == "cuda")
        return (x.to(compute_device, non_blocking=non_block),
                c_idx.to(compute_device, non_blocking=non_block),
                y_ri.to(compute_device, non_blocking=non_block))

    return sample_batch


def make_spoke_sampler_kxy(kxy_sp, y_sp, spokes, *, batch_size, device="cuda"):
    device = torch.device(device)
    kxy_sp = kxy_sp.to(device)
    y_sp = y_sp.to(device)
    spokes = spokes.to(device)

    _, RO, _ = kxy_sp.shape

    def sample():
        sp = spokes[torch.randint(spokes.numel(), (batch_size,), device=device)]
        ro = torch.randint(RO, (batch_size,), device=device)
        x = kxy_sp[sp, ro, :]   # (B,2)
        y = y_sp[sp, ro, :]     # (B,2)
        return x, y
    return sample

@torch.no_grad()
def make_fixed_eval_set_kxy(kxy_sp, y_sp, spokes, *, n=200_000, device="cuda"):
    device = torch.device(device)
    kxy_sp = kxy_sp.to(device)
    y_sp = y_sp.to(device)
    spokes = spokes.to(device)

    _, RO, _ = kxy_sp.shape
    sp = spokes[torch.randint(spokes.numel(), (n,), device=device)]
    ro = torch.randint(RO, (n,), device=device)
    x = kxy_sp[sp, ro, :]
    y = y_sp[sp, ro, :]
    return x, y



def make_fixed_frame_kslice_coil_dataset(
    k_t, traj_t, scales, dims,
    *,
    y_scale,
    t_fixed: int = 0,
    coil_fixed: int = 0,
    kz_index: int = 0,
    compute_device: str = "cuda",
):
    """
    Build a deterministic dataset for ONE frame, ONE kz-plane, ONE coil.

    k_t    : (T,S,C,RO) complex
    traj_t : (T,S,3,RO) float
    returns:
      x_all   : (N,4) float  [kx,ky,kz,t_norm]
      y_all   : (N,2) float  [Re,Im]
      kx_all, ky_all : (N,) float (for plotting)
    """
    sx, sy, sz = scales
    T, _, C, RO = dims
    dev_data = k_t.device
    dev_compute = torch.device(compute_device)

    assert 0 <= t_fixed < T
    assert 0 <= coil_fixed < C

    # kz per spoke (constant along RO, so take ro=0)
    kz_sp = traj_t[t_fixed, :, 2, 0]  # (S,)
    kz_bins = torch.sort(torch.unique(kz_sp)).values
    kz_index = int(max(0, min(int(kz_index), int(len(kz_bins) - 1))))
    kz_target = kz_bins[kz_index]

    # select spokes belonging to this kz plane
    sp_mask = (kz_sp == kz_target)
    sp_idx = torch.where(sp_mask)[0]  # (S_kz,)


    # Gather coords for all (spoke, ro)
    # kx,ky,kz: (S_kz, RO)
    kx = traj_t[t_fixed, sp_idx, 0, :] / sx
    ky = traj_t[t_fixed, sp_idx, 1, :] / sy
    kz = traj_t[t_fixed, sp_idx, 2, :] / sz

    # time is fixed (normalized)
    t_norm = (torch.tensor(t_fixed, device=dev_data, dtype=traj_t.dtype) / (T - 1 + 1e-8)) * 2.0 - 1.0
    t_col = torch.full((sp_idx.numel(), RO), t_norm, device=dev_data, dtype=traj_t.dtype)

    # Flatten to (N,)
    kx_all = kx.reshape(-1)
    ky_all = ky.reshape(-1)
    kz_all = kz.reshape(-1)
    t_all  = t_col.reshape(-1)

    x_all = torch.stack([kx_all, ky_all, kz_all, t_all], dim=1).float()  # (N,4)

    # Measured k-space for this coil: (S_kz, RO) -> (N,)
    y = k_t[t_fixed, sp_idx, coil_fixed, :].reshape(-1)
    y_ri = torch.view_as_real(y).float()  # (N,2)

    non_block = (dev_data.type == "cuda" and dev_compute.type == "cuda")
    x_all = x_all.to(dev_compute, non_blocking=non_block)
    y_ri = y_ri.to(dev_compute, non_blocking=non_block)
    kx_all = kx_all.to(dev_compute, non_blocking=non_block)
    ky_all = ky_all.to(dev_compute, non_blocking=non_block)

    y_ri = y_ri / y_scale

    meta = {
        "t_fixed": t_fixed,
        "coil_fixed": coil_fixed,
        "kz_index": kz_index,
        "kz_target": float(kz_target.item()),
        "n_spokes_in_plane": int(sp_idx.numel()),
        "N": int(x_all.shape[0]),
        "y_scale": float(y_scale.item()),
    }
    return x_all, y_ri, kx_all, ky_all, meta


def split_points_by_spokes(spoke_id_all, *, val_frac=0.2, seed=0, mode="random"):
    """
    spoke_id_all: (N,) long, values in {0..S_kz-1} identifying which spoke each point came from.

    Returns:
      train_idx, val_idx: 1D Long tensors of point indices into x_all/y_all
      train_spokes, val_spokes: 1D Long tensors of spoke ids
    """
    device = spoke_id_all.device
    S_kz = int(spoke_id_all.max().item()) + 1
    n_val = max(1, int(round(S_kz * val_frac)))

    g = torch.Generator(device=device)
    g.manual_seed(int(seed))

    if mode == "random":
        perm = torch.randperm(S_kz, generator=g, device=device)
        val_spokes = perm[:n_val]
        train_spokes = perm[n_val:]
    else:
        raise ValueError("mode must be 'random'")

    # point indices
    train_mask = torch.isin(spoke_id_all, train_spokes)
    val_mask   = torch.isin(spoke_id_all, val_spokes)

    train_idx = torch.where(train_mask)[0]
    val_idx   = torch.where(val_mask)[0]
    return train_idx, val_idx, train_spokes, val_spokes



def fit_one_frame_slice_coil(
    model,
    *,
    x_all,
    y_all,
    coil_fixed = None,
    steps: int = 5000,
    loss_function = F.mse_loss,
    lr: float = 1e-3,
    grad_clip: float = 1.0,
    amp: bool = False,
    device: str = "cuda",
    batch_size: int = 65536,
    use_tqdm: bool = True,
    log_every: int = 50,
    callback=None,
    callback_every: int = 0,
):
    """
    Overfit on a fixed dataset (x_all,y_all) from one frame/slice/coil.
    """
    device = torch.device(device)
    model = model.to(device)
    model.train()
    x_all = x_all.to(device, non_blocking=True)
    y_all = y_all.to(device, non_blocking=True)
    
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    scaler = GradScaler("cuda" if device.type == "cuda" else "cpu", enabled=amp)
    amp_dtype = torch.float16 if (device.type == "cuda" and not torch.cuda.is_bf16_supported()) else torch.bfloat16

    N = x_all.shape[0]
    if coil_fixed is None:
        coil_idx_full = None
    else:
        coil_idx_full = torch.full((batch_size,), int(coil_fixed), device=device, dtype=torch.long)

    if use_tqdm:
        from tqdm.auto import trange
        it = trange(1, steps + 1, desc="Overfitting fixed (t,kz,coil)", leave=True, dynamic_ncols=True)
    else:
        it = range(1, steps + 1)

    loss_hist = []
    for step in it:

        idx = torch.randint(0, N, (batch_size,), device=device)
        x = x_all[idx]
        y = y_all[idx]

        if coil_idx_full is not None:
            coil_idx = coil_idx_full[: batch_size]  # reuse
        else:
            coil_idx = None

        opt.zero_grad(set_to_none=True)

        ac = (autocast("cuda" if device.type == "cuda" else "cpu", dtype=amp_dtype, enabled=True)
              if amp else nullcontext())

        with ac:
            if coil_idx is not None:
                y_pred = model(x, coil_idx)
            else:
                y_pred = model(x)
            loss = loss_function(y_pred, y)

        if amp:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

        L = float(loss.item())
        loss_hist.append(L)

        if use_tqdm:
            it.set_postfix_str(f"mse={L:.2e}")
        elif step % log_every == 0:
            print(f"step {step:6d}  mse {L:.3e}")

        # callback hook
        if callback and callback_every > 0:
            if (step % callback_every == 0) or (step == 1) or (step == steps):
                callback(step, model)
        
    return model, {"loss_hist": loss_hist}



def fit_one_scan(
    model,
    sampler,
    *,
    steps=20000,
    batch_size=131072,
    lr=1e-3,
    grad_clip=1.0,
    amp=False,
    device="cuda",
    log_every=50,
    use_tqdm=False,
    eval_size=200_000,
    eval_every=300,
    eval_min_rel=5e-4,
    eval_patience=10,

):
    device = torch.device(device)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    scaler = GradScaler("cuda" if device.type == "cuda" else "cpu", enabled=amp)
    amp_dtype = torch.float16 if (device.type == "cuda" and not torch.cuda.is_bf16_supported()) else torch.bfloat16

    #  fixed eval set
    @torch.no_grad()
    def make_eval_set(n=eval_size):
        model.eval()
        x, coil, y = sampler(n)
        return x, coil, y

    x_eval, c_eval, y_eval = make_eval_set(eval_size)

    @torch.no_grad()
    def eval_mse():
        model.eval()
        ac = (autocast("cuda" if device.type == "cuda" else "cpu", dtype=amp_dtype, enabled=True)
              if amp else nullcontext())
        with ac:
            y_pred = model(x_eval, c_eval)
            return F.mse_loss(y_pred, y_eval).item()

    stopper = EvalEarlyStopper(min_rel_improve=eval_min_rel, patience=eval_patience)

    #stopper = PlateauStopper(window=plateau_window, min_rel_improve=plateau_min_rel, patience=plateau_patience)

    best_state = None
    loss_hist = []
    eval_hist = []

    if use_tqdm:
        from tqdm.auto import trange
        from tqdm.auto import tqdm
        it = trange(1, steps + 1, desc="Fitting NIK", leave=True, dynamic_ncols=True)
    else:
        it = range(1, steps + 1)

    t0 = time.time()
    for step in it:
        model.train()
        x, coil_idx, y_meas = sampler(batch_size)

        opt.zero_grad(set_to_none=True)

        ac = (autocast("cuda" if device.type == "cuda" else "cpu", dtype=amp_dtype, enabled=True)
              if amp else nullcontext())

        with ac:
            y_pred = model(x, coil_idx)
            loss = F.mse_loss(y_pred, y_meas)

        if amp:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

        L = float(loss.item())
        loss_hist.append(L)

        #  early stop on eval plateau
        if step % eval_every == 0:
            e = eval_mse()
            eval_hist.append((step, e))

            # save best-on-eval state
            if e < stopper.best:
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

            stop = stopper.step(e)

            if use_tqdm:
                it.set_postfix_str(f"train={L:.2e}")
                tqdm.write(
                    f"step {step:6d}  train {L:.3e}  eval {e:.3e}  "
                    f"best {stopper.best:.3e}  rel {stopper.last_rel:.2e}  "
                    f"stall {stopper.bad}/{stopper.patience}"
                )

            else:
                elapsed = time.time() - t0
                it_s = step / max(elapsed, 1e-6)
                print(f"step {step:6d}  train {L:.3e}  eval {e:.3e}  it/s {it_s:.2f}  {stopper.postfix()}")

            if stop:
                if not use_tqdm:
                    print(f"Early stop on eval plateau at step {step}. best_eval={stopper.best:.3e}")
                break
        else:

            if (not use_tqdm) and (step % log_every == 0):
                elapsed = time.time() - t0
                it_s = step / max(elapsed, 1e-6)
                print(f"step {step:6d}  train {L:.3e}  it/s {it_s:.2f}")

    # restore best model 
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {
        "loss_hist": loss_hist,
        "eval_hist": eval_hist,
        "best_eval": stopper.best
    }
    


@torch.no_grad()
def plot_measured_vs_pred_kspace(
    model,
    x_all,
    y_all,
    kx_all,
    ky_all,
    coil_fixed,
    y_scale=1.0,
    max_points=300_000,
    show_magphase=True,
    phase_mask_percentile=60,
):
    """
    Plot measured vs predicted k-space on the (kx,ky) plane.


    Outputs:
      - Figure 1: Re/Im (measured vs predicted)  
      - Figure 2: magnitude/phase (optional)     
    """
    device = x_all.device
    model.eval()

    N = x_all.shape[0]
    idx = torch.randperm(N, device=device)[:min(N, max_points)] if N > max_points else torch.arange(N, device=device)

    x = x_all[idx]
    y = y_all[idx]
    kx = kx_all[idx].detach().cpu().numpy()
    ky = ky_all[idx].detach().cpu().numpy()

    # ensure y_scale is float
    ys = float(y_scale.detach().cpu().item()) if isinstance(y_scale, torch.Tensor) else float(y_scale)

    coil_idx = torch.full((x.shape[0],), int(coil_fixed), device=device, dtype=torch.long)

    # model outputs Re/Im 
    y_pred = model(x, coil_idx)  # (B,2)

    # undo normalization for plotting 
    y_plot = (y * ys).detach().cpu().numpy()
    yp_plot = (y_pred * ys).detach().cpu().numpy()

    y_c  = y_plot[:, 0] + 1j * y_plot[:, 1]
    yp_c = yp_plot[:, 0] + 1j * yp_plot[:, 1]

    #  Figure 1: Re/Im  
    def scatter_component(ax, vals_complex, title, component="re"):
        def clip_vals(v, p=99):
            lo, hi = np.percentile(v, [100-p, p])
            return np.clip(v, lo, hi)
        
        c = clip_vals(vals_complex.real, p=97) if component == "re" else clip_vals(vals_complex.imag, p=97)
        sc = ax.scatter(kx, ky, c=c, s=2)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        ax.set_xlabel("kx")
        ax.set_ylabel("ky")
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=("Re" if component == "re" else "Im"))

    _, axes = plt.subplots(2, 2, figsize=(12, 12))
    scatter_component(axes[0, 0], y_c,  "Measured Re", component="re")
    scatter_component(axes[0, 1], yp_c, "Predicted Re", component="re")
    scatter_component(axes[1, 0], y_c,  "Measured Im", component="im")
    scatter_component(axes[1, 1], yp_c, "Predicted Im", component="im")
    plt.tight_layout()
    plt.show()

    #  Figure 2: magnitude/phase 
    if not show_magphase:
        return

    def scatter_mag(ax, vals_complex, title):
        c = np.log1p(np.abs(vals_complex))
        sc = ax.scatter(kx, ky, c=c, s=2)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        ax.set_xlabel("kx")
        ax.set_ylabel("ky")
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="log(1+|y|)")

    def scatter_phase(ax, vals_complex, title, mask=None):
        if mask is not None:
            kx_m, ky_m, v_m = kx[mask], ky[mask], vals_complex[mask]
        else:
            kx_m, ky_m, v_m = kx, ky, vals_complex
        c = np.angle(v_m)
        sc = ax.scatter(kx_m, ky_m, c=c, s=2)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        ax.set_xlabel("kx")
        ax.set_ylabel("ky")
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="phase [rad]")

    # mask low-magnitude points for phase visibility
    mag = np.abs(y_c)
    thr = np.percentile(mag, phase_mask_percentile) if phase_mask_percentile is not None else 0.0
    mask = mag >= thr if phase_mask_percentile is not None else None

    _, axes = plt.subplots(2, 2, figsize=(12, 12))
    scatter_mag(axes[0, 0], y_c,  "Measured |y|")
    scatter_mag(axes[0, 1], yp_c, "Predicted |y|")
    scatter_phase(axes[1, 0], y_c,  f"Measured phase (mask p{phase_mask_percentile})", mask=mask)
    scatter_phase(axes[1, 1], yp_c, f"Predicted phase (mask p{phase_mask_percentile})", mask=mask)
    plt.tight_layout()
    plt.show()



def overfit_fixed_subset(
    model,
    x_all,
    y_all,
    coil_fixed,
    *,
    n_points=8192,
    steps=2000,
    lr=1e-4,
    device="cuda",
):
    """
     sanity check:
    Overfit a  fixed subset of points.
    """
    device = torch.device(device)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # pick subset 
    N = x_all.shape[0]
    idx = torch.randperm(N, device=device)[:n_points]
    x_fix = x_all[idx]
    y_fix = y_all[idx]
    coil_idx = torch.full((n_points,), int(coil_fixed), device=device, dtype=torch.long)

    print(f"[DEBUG] Overfitting {n_points} fixed points")

    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        y_pred = model(x_fix, coil_idx)   # Re/Im model
        loss = F.mse_loss(y_pred, y_fix)
        loss.backward()
        opt.step()

        if step % 100 == 0 or step == 1:
            print(f"[DEBUG] step {step:4d}  mse {loss.item():.3e}")

    return model


def fit_spoke_holdout_kxy(
    model,
    *,
    kxy_sp, y_sp, theta,
    train_spokes, val_spokes,
    steps=1000,
    batch_size=20000,
    lr=1e-3,
    grad_clip=1.0,
    device="cuda",
    eval_size=20000,
    eval_every=300,
    eval_min_rel=5e-4,
    eval_patience=10,
    plot_every=250,
    plot_callback=None,
):
    do_eval = val_spokes.numel() > 0
    device = torch.device(device)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    train_sampler = make_spoke_sampler_kxy(
        kxy_sp, y_sp, train_spokes,
        batch_size=batch_size, device=str(device)
    )

    if do_eval:
        x_val, y_val = make_fixed_eval_set_kxy(
            kxy_sp, y_sp, val_spokes,
            n=eval_size, device=str(device)
        )
    else:
        x_val = y_val = None


    stopper = EvalEarlyStopper(min_rel_improve=eval_min_rel, patience=eval_patience)
    best_state = None

    loss_hist = []          # train loss per step
    eval_hist = []          # list of (step, eval_loss)
    best_step = None
    stopped_step = None

    t0 = time.time()
    for step in range(1, steps + 1):
        model.train()
        x, y = train_sampler()

        opt.zero_grad(set_to_none=True)
        yp = model(x)
        loss = F.mse_loss(yp, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        loss_hist.append(float(loss.item()))

        # validation / early stopping
        if do_eval and step % eval_every == 0:
            model.eval()
            with torch.no_grad():
                e = float(F.mse_loss(model(x_val), y_val).item())
            eval_hist.append((step, e))

            if e < stopper.best:
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                best_step = step

            if stopper.step(e):
                stopped_step = step
                break

        # plotting callback
        if plot_callback is not None and plot_every and (step % plot_every == 0 or step == 1):
            plot_callback(step, model)

    if stopped_step is None:
        stopped_step = step  # last step reached

    # restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    info = {
        "loss_hist": loss_hist,
        "eval_hist": eval_hist,
        "best_eval": stopper.best,
        "best_step": best_step,
        "stopped_step": stopped_step,
        "eval_every": eval_every,
        "eval_size": eval_size,
        "train_spokes": int(train_spokes.numel()),
        "val_spokes": int(val_spokes.numel()),
        "seconds": time.time() - t0,
    }
    return model, info









