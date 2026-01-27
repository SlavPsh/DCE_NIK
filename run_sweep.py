import os, json, hashlib, itertools
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt

from nik_model import NIK_SIREN_REIM   # add  NIK_SIREN2D_REIM  later
from nik_train import fit_one_frame_slice_coil
from nik_recon import nufft2d_recon, fft2d_uniform, to_plot, make_fixed_frame_zslice_coil_dataset

# ---------------------------
# helpers
# ---------------------------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def dict_product(d):
    keys = list(d.keys())
    vals = [d[k] if isinstance(d[k], list) else [d[k]] for k in keys]
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo))

def cfg_hash(obj) -> str:
    s = json.dumps(obj, sort_keys=True).encode("utf-8")
    return hashlib.sha1(s).hexdigest()[:10]

def mse(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.mean((a - b) ** 2))

def norm99(x):
    x = np.asarray(x, dtype=np.float32)
    p = np.percentile(x, 99)
    return x / (p + 1e-12)

@torch.no_grad()
def predict_on_points(model, x_all, coil_model_idx, k_scale):
    model.eval()
    coil_vec = torch.full((x_all.shape[0],), int(coil_model_idx), device=x_all.device, dtype=torch.long)
    y_pred = model(x_all, coil_vec) * k_scale
    k_pred = torch.complex(y_pred[:, 0], y_pred[:, 1])
    return k_pred

@torch.no_grad()
def predict_cartesian_kspace(model, x_all, coil_model_idx, k_scale, Ny, Nx, kx_max, ky_max, oversamp=1):
    # reuse z,t constants from x_all
    z_const = x_all[0, 2]
    t_const = x_all[0, 3]

    Ny_os, Nx_os = int(Ny*oversamp), int(Nx*oversamp)
    kx_lin = torch.linspace(-kx_max, kx_max, Nx_os, device=x_all.device)
    ky_lin = torch.linspace(-ky_max, ky_max, Ny_os, device=x_all.device)
    KY, KX = torch.meshgrid(ky_lin, kx_lin, indexing="ij")

    kx_flat = KX.reshape(-1)
    ky_flat = KY.reshape(-1)
    z_col = z_const.expand(Ny_os * Nx_os)
    t_col = t_const.expand(Ny_os * Nx_os)
    x_query = torch.stack([kx_flat, ky_flat, z_col, t_col], dim=1).float()

    model.eval()
    coil_vec = torch.full((x_query.shape[0],), int(coil_model_idx), device=x_all.device, dtype=torch.long)
    y_pred = model(x_query, coil_vec) * k_scale
    k_cart = torch.complex(y_pred[:, 0], y_pred[:, 1]).reshape(Ny_os, Nx_os)
    return k_cart, KX, KY

def crop_center(img, Ny, Nx):
    Ny_os, Nx_os = img.shape[-2], img.shape[-1]
    y0 = (Ny_os - Ny)//2
    x0 = (Nx_os - Nx)//2
    return img[..., y0:y0+Ny, x0:x0+Nx]

def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# ---------------------------
# main sweep
# ---------------------------
cfg = load_yaml("sweep.yaml")
out_dir = cfg["experiment"]["out_dir"]
ensure_dir(out_dir)

# read fixed settings
t_frame = cfg["data"]["t_frame"]
coil_idx_data = cfg["data"]["coil_idx_data"]
z_slice_idx = cfg["data"]["z_slice_idx"]
device = cfg["data"].get("compute_device", "cuda")

steps = cfg["train"]["steps"]
batch_size = cfg["train"]["batch_size"]
lr = cfg["train"]["lr"]
amp = cfg["train"].get("amp", False)
log_every = cfg["train"].get("log_every", 1000)

Ny, Nx = cfg["recon"]["img_size"]  # (Ny, Nx)
oversamp = cfg["recon"]["cartesian"].get("oversamp", 1)
use_disk_mask = cfg["recon"]["cartesian"].get("use_disk_mask", True)

# Build dataset 
# k_img_space, traj_t, scales, dims, k_scale must exist
x_all, y_all, kx_all, ky_all, meta = make_fixed_frame_zslice_coil_dataset(
    k_img_space, traj_t, scales, dims,
    y_scale=k_scale,
    t_fixed=t_frame,
    coil_fixed=coil_idx_data,
    z_slice_idx=z_slice_idx,
    n_slices=meta["n_slices"] if "n_slices" in locals() else None,
    compute_device=device,
)

# measured baseline image (NUFFT) for comparison
# img_measured = nufft2d_recon(k_img_space, traj_t, t_frame, coil_idx_data, z_slice_idx, scales, img_size=(Ny, Nx), n_slices=meta["n_slices"])
img_measured = img_measured  # if you already computed it
img_measured_p = norm99(to_plot(img_measured))

kx_max = float(kx_all.abs().max().item())
ky_max = float(ky_all.abs().max().item())

results = []

for model_block in cfg["sweep"]["models"]:
    ctor_name = model_block["ctor"]
    name = model_block["name"]
    param_grid = model_block["params"]

    for params in dict_product(param_grid):
        run_cfg = {
            "model": {"ctor": ctor_name, "params": params},
            "data": cfg["data"],
            "train": cfg["train"],
            "recon": cfg["recon"],
        }
        run_id = f"{name}_{cfg_hash(run_cfg)}"
        run_path = os.path.join(out_dir, run_id)
        ensure_dir(run_path)

        # skip if already done
        metrics_path = os.path.join(run_path, "metrics.json")
        ckpt_path = os.path.join(run_path, "model.pt")
        if os.path.exists(metrics_path) and os.path.exists(ckpt_path):
            with open(metrics_path, "r") as f:
                results.append(json.load(f))
            continue

        # ---- build model ----
        if ctor_name == "NIK_SIREN_REIM":
            model = NIK_SIREN_REIM(
                n_coils=1,  # you are fitting a single coil at a time
                **params
            ).to(device)
            coil_model_idx = 0
        else:
            raise ValueError(f"Unknown ctor: {ctor_name}")

        # ---- fit ----
        model, info = fit_one_frame_slice_coil(
            model,
            x_all=x_all,
            y_all=y_all,
            coil_fixed=0,        # model coil index (since n_coils=1)
            steps=steps,
            batch_size=batch_size,
            lr=lr,
            amp=amp,
            device=device,
            use_tqdm=False,
            log_every=log_every,
        )

        # ---- recon 1: model on measured points -> NUFFT ----
        k_pred_pts = predict_on_points(model, x_all, coil_model_idx, k_scale)  # (N,) complex
        k_pred_slice = k_pred_pts.reshape(meta["n_ro_per_slice"], dims[3])     # (n_ro_per_slice, RO)

        # create a copy of k_img_space with predicted slice filled if your nufft2d needs it
        k_img_space_pred = torch.zeros_like(k_img_space)
        k_img_space_pred[t_frame, :, coil_idx_data, z_slice_idx, :] = k_pred_slice

        # img_pred_nufft = nufft2d_recon(k_img_space_pred, traj_t, t_frame, coil_idx_data, z_slice_idx, scales, img_size=(Ny, Nx), n_slices=meta["n_slices"])
        img_pred_nufft = img_pred_nufft  # replace if you already have a direct function

        # ---- recon 2: model on Cartesian grid -> FFT ----
        k_cart, KX, KY = predict_cartesian_kspace(
            model, x_all, coil_model_idx, k_scale,
            Ny=Ny, Nx=Nx,
            kx_max=kx_max, ky_max=ky_max,
            oversamp=oversamp
        )
        if use_disk_mask:
            mask = ((KX / kx_max) ** 2 + (KY / ky_max) ** 2) <= 1.0
            k_cart = k_cart * mask

        img_cart = fft2d_uniform(k_cart, axes=(-2, -1), shift=True, return_magnitude=False)
        img_cart = crop_center(img_cart, Ny, Nx) if oversamp != 1 else img_cart
        img_pred_fft = torch.abs(img_cart)

        # ---- metrics (use normalized for comparability) ----
        img_pred_nufft_p = norm99(to_plot(img_pred_nufft))
        img_pred_fft_p = norm99(to_plot(img_pred_fft))

        m_nufft = mse(img_pred_nufft_p, img_measured_p)
        m_fft = mse(img_pred_fft_p, img_measured_p)

        metrics = {
            "run_id": run_id,
            "ctor": ctor_name,
            "params": params,
            "mse_vs_measured_nufft__pred_nufft": m_nufft,
            "mse_vs_measured_nufft__pred_fft": m_fft,
            "t_frame": t_frame,
            "z_slice_idx": z_slice_idx,
            "coil_idx_data": coil_idx_data,
            "Ny": Ny, "Nx": Nx,
            "oversamp": oversamp,
            "use_disk_mask": use_disk_mask,
        }

        # ---- plot comparison panel ----
        fig, ax = plt.subplots(1, 3, figsize=(14, 4))
        ax[0].imshow(img_measured_p, cmap="gray"); ax[0].set_title("Measured NUFFT (norm99)"); ax[0].axis("off")
        ax[1].imshow(img_pred_nufft_p, cmap="gray"); ax[1].set_title(f"Pred NUFFT\nMSE={m_nufft:.4e}"); ax[1].axis("off")
        ax[2].imshow(img_pred_fft_p, cmap="gray"); ax[2].set_title(f"Pred FFT\nMSE={m_fft:.4e}"); ax[2].axis("off")
        plt.tight_layout()
        fig_path = os.path.join(run_path, "compare.png")
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)

        # ---- save ----
        save_checkpoint(model, ckpt_path)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        results.append(metrics)
        print(run_id, "MSE(nufft)", m_nufft, "MSE(fft)", m_fft)

# Save aggregate summary
with open(os.path.join(out_dir, "summary.json"), "w") as f:
    json.dump(results, f, indent=2)

print("Done. Results saved to:", out_dir)
