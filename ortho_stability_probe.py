"""T1a early-step QR stability log. real full model + PCA warm start + ortho, CPU, ~80 steps.
watches rawGramCond (QR conditioning) and gradnorm. no recon (too slow on CPU)."""
import numpy as np, torch, sys, warnings; warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")
sys.path.insert(0, "/scratch/rnga/vvpshenov/grasp_pro_py")
import nik_adapter as A
from nik_model import WIRE_FF_SUBSPACE_KXY_COIL_T_REIM, warmstart_phi
from kspace_normalization import compute_dcf_radial, KSpaceNormalizer
from nik_focal_loss import composable_kspace_loss
from train_grasp_nik import compute_pca_phi

dev = torch.device("cpu"); REF = "/scratch/rnga/vvpshenov/grasp_pro_py/results_ref"
sh = A.load_shared(REF); ds = A.make_radial_dataset(REF, 13, compute_device=dev, shared=sh)
x, t, c, yraw = ds["x_all"], ds["t_all"], ds["coil_all"], ds["y_all_raw"]
spoke = ds["spoke_id_all"]; ncc = ds["meta"]["ncc"]
keep = torch.as_tensor(np.load("spoke_masks/keep_f100.npy"), dtype=spoke.dtype)
tr = torch.where(torch.isin(spoke, keep))[0]
dcf = torch.ones(x.shape[0])
nz = KSpaceNormalizer(); nz.fit(x[tr], yraw[tr], dcf=dcf[tr], envelope_exponent=0.75)
y = nz.normalize(x, yraw)
xt, tt, ct, yt = x[tr], t[tr], c[tr], y[tr]; N = xt.shape[0]

torch.manual_seed(0)
m = WIRE_FF_SUBSPACE_KXY_COIL_T_REIM(n_coils=ncc, rank=16, hidden=512, depth=12, w0=62., s0=15.,
                                     k_freq=256, k_sigma=2.5, t_freq=32, t_sigma=1.5, ortho=True).to(dev)
ft, phi = compute_pca_phi(REF, 13, sh, 16)
err = warmstart_phi(m, ft, phi, steps=800, lr=1e-3, device="cpu")
print(f"warmstart Phi<-PCA fit MSE {err:.3e}", flush=True)
opt = torch.optim.Adam(m.parameters(), lr=1e-5, weight_decay=3e-3)
print(f"{'step':>4} {'loss':>10} {'gradnorm':>10} {'rawGramCond':>12}", flush=True)
for step in range(1, 81):
    idx = torch.randint(0, N, (8192,))
    opt.zero_grad(set_to_none=True)
    yp = m(xt[idx], tt[idx], ct[idx])
    loss = composable_kspace_loss(yp, yt[idx], dcf=torch.ones(8192), use_dcf=False, dcf_power=0.0,
                                  use_focal=False, return_diagnostics=False)
    loss.backward()
    g = float(torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0))
    opt.step()
    if step <= 10 or step % 10 == 0:
        print(f"{step:4d} {float(loss):10.3e} {g:10.2e} {m.last_gram_cond:12.2e}", flush=True)
print("done", flush=True)
