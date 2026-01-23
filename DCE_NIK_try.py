import itertools
import argparse
import math
import os
import time
import h5py
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


PATH_K_DCE   = "/results/kspace/DCE"
PATH_TRAJ    = "/results/kspace/trajDCE"
PATH_GT_IMG  = "/results/images/GroundTruth/img"
PATH_RC_IMG  = "/results/images/Recon/img"
PATH_GT_TIM  = "/results/images/GroundTruth/timing"
PATH_RC_TIM  = "/results/images/Recon/timing"


def h5_tree(file_path, max_items=250):
    items = []
    with h5py.File(file_path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                items.append(("D", name, obj.shape, str(obj.dtype)))
            else:
                items.append(("G", name, None, None))
        f.visititems(visit)
    for t, name, shape, dtype in items[:max_items]:
        if t == "D":
            print(f"DATASET  {name:50s}  shape={shape}  dtype={dtype}")
        else:
            print(f"GROUP    {name}")
    if len(items) > max_items:
        print(f"... ({len(items)-max_items} more)")


def _as_complex(arr: np.ndarray) -> np.ndarray:
    if arr.dtype.fields and ("real" in arr.dtype.fields) and ("imag" in arr.dtype.fields):
        return arr["real"] + 1j * arr["imag"]
    return arr


def h5_load(file_path: str, path_in_file: str) -> np.ndarray:
    with h5py.File(file_path, "r") as f:
        arr = f[path_in_file][()]
    return _as_complex(arr)


def h5_exists(file_path: str, path_in_file: str) -> bool:
    with h5py.File(file_path, "r") as f:
        return path_in_file in f


def canonicalize_k_and_traj(k: np.ndarray, traj: np.ndarray, coil_max: int = 64):
    """
    Find consistent permutations so that:
      traj -> (RO, S, 3, T)
      k    -> (RO, S, C, T)
    Then, return tensors re-ordered to:
      traj_ts3ro -> (T, S, 3, RO)
      k_tscro    -> (T, S, C, RO)
    """
    best = None
    for p_tr in itertools.permutations(range(4)):
        trp = np.transpose(traj, p_tr)
        if trp.shape[2] != 3:
            continue
        ROc, Spc, _, Fc = trp.shape
        for p_k in itertools.permutations(range(4)):
            kp = np.transpose(k, p_k)
            if kp.shape[0] == ROc and kp.shape[1] == Spc and kp.shape[3] == Fc:
                coils = kp.shape[2]
                if coils <= coil_max:
                    best = (trp, kp)
                    break
        if best is not None:
            break
    if best is None:
        raise RuntimeError(f"Could not find consistent permutations. k={k.shape}, traj={traj.shape}")

    trp, kp = best  # trp: (RO,S,3,T), kp: (RO,S,C,T)

    # Reorder to (T,S,*,RO) for intuitive sampling with t first
    traj_ts3ro = np.transpose(trp, (3, 1, 2, 0))  # (T, S, 3, RO)
    k_tscro    = np.transpose(kp,  (3, 1, 2, 0))  # (T, S, C, RO)
    return k_tscro, traj_ts3ro


class FourierFeatures(nn.Module):
    def __init__(self, in_dim, n_freq=64, sigma=6.0):
        super().__init__()
        B = torch.randn(in_dim, n_freq) * sigma
        self.register_buffer("B", B)

    def forward(self, x):
        proj = 2.0 * np.pi * (x @ self.B)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, w0=30.0, is_first=False):
        super().__init__()
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features)
        self.w0 = w0
        self.is_first = is_first
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                bound = math.sqrt(6 / self.in_features) / self.w0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.w0 * self.linear(x))


class NIK_SIREN(nn.Module):
    def __init__(self, n_coils, k_freq=96, t_freq=16, k_sigma=6.0, t_sigma=3.0,
                 coil_emb=16, hidden=256, depth=7, w0=30.0):
        super().__init__()
        self.ff_k = FourierFeatures(3, n_freq=k_freq, sigma=k_sigma)
        self.ff_t = FourierFeatures(1, n_freq=t_freq, sigma=t_sigma)
        self.coil_emb = nn.Embedding(n_coils, coil_emb)

        in_dim = 2 * k_freq + 2 * t_freq + coil_emb
        layers = [SineLayer(in_dim, hidden, w0=w0, is_first=True)]
        for _ in range(depth - 2):
            layers.append(SineLayer(hidden, hidden, w0=w0, is_first=False))
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, 2)  # [log_mag, phase_raw]

    def forward(self, kxyz_t, coil_idx):
        # kxyz_t: (B,4) = [kx,ky,kz,t_norm]
        k = kxyz_t[:, :3]
        t = kxyz_t[:, 3:4]
        zk = self.ff_k(k)
        zt = self.ff_t(t)
        ec = self.coil_emb(coil_idx)
        h = torch.cat([zk, zt, ec], dim=-1)
        h = self.backbone(h)
        out = self.head(h)
        log_mag = out[:, 0:1]
        phase = np.pi * torch.tanh(out[:, 1:2])  # bounded
        return log_mag, phase


def magphase_to_ri(log_mag, phase):
    mag = torch.exp(log_mag)
    re = mag * torch.cos(phase)
    im = mag * torch.sin(phase)
    return torch.cat([re, im], dim=1)  # (B,2)


def main():
    parser = argparse.ArgumentParser(description="GPU-optimized DCE k-space trainer")
    parser.add_argument("--file", default="XCAT-ERIC/results/simulation_results_20260109T221333.mat")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=131072)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile if available")
    parser.add_argument("--data-device", choices=["cpu", "cuda"], default="cpu",
                        help="Where to store k-space and traj tensors")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--print-h5", action="store_true", help="Print HDF5 tree and exit")
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    if (args.device == "cuda" or args.data_device == "cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA requested but this PyTorch build has no CUDA support. "
            "Install a CUDA-enabled PyTorch or rerun with --device cpu --data-device cpu."
        )

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    if args.print_h5:
        h5_tree(args.file, max_items=500)
        return

    # Load arrays (NumPy)
    k_raw  = h5_load(args.file, PATH_K_DCE)
    tr_raw = h5_load(args.file, PATH_TRAJ)
    k_np, traj_np = canonicalize_k_and_traj(k_raw, tr_raw)
    k_np = k_np.astype(np.complex64, copy=False)
    traj_np = traj_np.astype(np.float32, copy=False)

    # Optional GT and recon for later evaluation (not used in training here)
    gt_img = h5_load(args.file, PATH_GT_IMG).astype(np.float32)
    rc_img = h5_load(args.file, PATH_RC_IMG).astype(np.float32)
    gt_tim = h5_load(args.file, PATH_GT_TIM).reshape(-1) if h5_exists(args.file, PATH_GT_TIM) else None
    rc_tim = h5_load(args.file, PATH_RC_TIM).reshape(-1) if h5_exists(args.file, PATH_RC_TIM) else None

    T, S, C, RO = k_np.shape
    print(f"k_dce: {k_np.shape} traj: {traj_np.shape} complex? {np.iscomplexobj(k_np)}")
    print(f"gt_img: {getattr(gt_img, 'shape', None)} rc_img: {getattr(rc_img, 'shape', None)}")

    # Move to Torch tensors
    on_cuda = (args.data_device == "cuda") and torch.cuda.is_available()
    pin = (args.data_device == "cpu")

    k_t = torch.from_numpy(k_np)
    traj_t = torch.from_numpy(traj_np)
    if pin:
        k_t = k_t.pin_memory()
        traj_t = traj_t.pin_memory()
    if on_cuda:
        k_t = k_t.to("cuda", non_blocking=True)
        traj_t = traj_t.to("cuda", non_blocking=True)

    # Scale factors from entire traj (vectorized)
    # traj_t: (T,S,3,RO)
    sx = (traj_t[:, :, 0, :].abs().max().item()) + 1e-8
    sy = (traj_t[:, :, 1, :].abs().max().item()) + 1e-8
    sz = (traj_t[:, :, 2, :].abs().max().item()) + 1e-8

    device = torch.device(args.device)
    model = NIK_SIREN(n_coils=C).to(device)

    if args.compile:
        try:
            model = torch.compile(model, mode="max-autotune")  # PyTorch 2.x
            print("Model compiled with torch.compile")
        except Exception as e:
            print(f"torch.compile unavailable or failed: {e}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    class PlateauStopper:
        def __init__(self, window=50, min_rel_improve=5e-4, patience=8):
            self.window = window
            self.min_rel_improve = min_rel_improve
            self.patience = patience
            self.hist = []
            self.bad = 0

        def step(self, loss):
            self.hist.append(loss)
            if len(self.hist) < 2 * self.window:
                return False
            a = float(np.mean(self.hist[-2 * self.window:-self.window]))
            b = float(np.mean(self.hist[-self.window:]))
            rel = (a - b) / max(a, 1e-12)
            if rel < self.min_rel_improve:
                self.bad += 1
            else:
                self.bad = 0
            return self.bad >= self.patience

    stopper = PlateauStopper(window=50, min_rel_improve=5e-4, patience=8)

    scaler = torch.amp.GradScaler('cuda' if args.device == 'cuda' else 'cpu', enabled=args.amp)
    amp_dtype = torch.bfloat16 if (args.amp and torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    # Torch-native sampler; indices live on same device as data tensors
    def sample_batch_torch(batch_size: int):
        idx_dev = k_t.device
        t_idx = torch.randint(T, (batch_size,), device=idx_dev)
        s_idx = torch.randint(S, (batch_size,), device=idx_dev)
        c_idx = torch.randint(C, (batch_size,), device=idx_dev)
        ro_idx = torch.randint(RO, (batch_size,), device=idx_dev)

        # k-space measurement (complex)
        y = k_t[t_idx, s_idx, c_idx, ro_idx]
        y_ri = torch.view_as_real(y)  # (B,2)

        # coordinates with normalization
        kx = traj_t[t_idx, s_idx, 0, ro_idx] / sx
        ky = traj_t[t_idx, s_idx, 1, ro_idx] / sy
        kz = traj_t[t_idx, s_idx, 2, ro_idx] / sz
        t_norm = (t_idx.to(traj_t.dtype) / (T - 1 + 1e-8)) * 2.0 - 1.0
        x = torch.stack([kx, ky, kz, t_norm], dim=1)  # (B,4)

        # Move to compute device if needed
        non_block = (idx_dev.type == "cuda" and device.type == "cuda")
        return (x.to(device, non_blocking=non_block),
                c_idx.to(device, non_blocking=non_block),
                y_ri.to(device, non_blocking=non_block))

    steps = int(args.steps)
    batch_size = int(args.batch_size)
    log_every = int(args.log_every)
    grad_clip = float(args.grad_clip)

    loss_hist = []
    best_loss = float('inf')
    best_model_state = None
    start = time.time()
    for step in range(1, steps + 1):
        x, coil_idx, y_meas = sample_batch_torch(batch_size)

        with torch.amp.autocast(device_type=args.device, dtype=amp_dtype, enabled=args.amp):
            log_mag, phase = model(x, coil_idx)
            y_pred = magphase_to_ri(log_mag, phase)
            loss = F.mse_loss(y_pred, y_meas)

        opt.zero_grad(set_to_none=True)
        if args.amp:
            scaler.scale(loss).backward()
            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

        L = float(loss.item())
        loss_hist.append(L)

        # Track best model (lowest loss seen so far)
        if L < best_loss:
            best_loss = L
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}

        if step % log_every == 0:
            elapsed = time.time() - start
            print(f"step {step:6d}  mse {L:.6e}  best_mse {best_loss:.6e}  time={elapsed:.1f}s")

        if stopper.step(L):
            print(f"Stopping on plateau at step {step}, mse {L:.6e}, best_mse {best_loss:.6e}")
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model with MSE {best_loss:.6e}")

    @torch.no_grad()
    def eval_random(n=200000):
        model.eval()
        x, coil_idx, y_meas = sample_batch_torch(n)
        with torch.amp.autocast(device_type=args.device, dtype=amp_dtype, enabled=args.amp):
            log_mag, phase = model(x, coil_idx)
            y_pred = magphase_to_ri(log_mag, phase)
            return F.mse_loss(y_pred, y_meas).item()

    print("MSE on random measured subset:", eval_random())


if __name__ == "__main__":
    main()

