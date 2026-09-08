"""does forcing b1-factorization cost NIK any data fit?

the render audit measured a 10.6 dB SENSE combine/re-expand loss: NIK's per-coil images do NOT
factor as b1_c x (one object). it labelled that "inherent model capacity". this tests whether it is
FREE to remove: project the per-coil k-space onto the factorizable set (SENSE combine, re-expand)
and measure how much WORSE the fit to the MEASURED spokes becomes.
  small degradation -> factorization costs nothing, SENSE-forward B has headroom
  large degradation -> the network needs non-factorization to fit data, B would hurt
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os, argparse
import numpy as np, torch
sys.path.insert(0, "/scratch/rnga/vvpshenov/DCE_NIK"); sys.path.insert(0, "/scratch/rnga/vvpshenov/grasp_pro_py")
import nik_adapter as A, train_grasp_nik as T
from fftc import ifft2c_mri, fft2c_mri

REF = "/scratch/rnga/vvpshenov/grasp_pro_py/results_ref"
OUT = "/scratch/rnga/vvpshenov/DCE_NIK/results_sl21_k80"
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sh = A.load_shared(REF); ds = A.make_radial_dataset(REF, 21, shared=sh)
m = ds["meta"]; ncc, bas, nx = m["ncc"], m["bas"], m["nx"]
b1 = ds["b1"].astype(np.complex64)
ck = torch.load(f"{OUT}/model_slice_21.pt", map_location=dev, weights_only=False)
args = argparse.Namespace(model=ck["model"], rank=ck["rank"], hidden=ck["hidden"], depth=ck["depth"],
                          w0=ck["w0"], s0=ck["s0"], coil_embed_dim=ck["coil_embed_dim"],
                          k_freq=ck["k_freq"], k_sigma=ck["k_sigma"], t_freq=ck["t_freq"],
                          t_sigma=ck["t_sigma"], ff_seed=0)
model = T.build_model(args, ncc).to(dev)
model.load_state_dict(ck["state_dict"]); model.eval()
print(f"model {ck['model']} loaded", flush=True)

# rebuild the normalizer exactly as the trainer does, fit on the SAME train spokes (v%10<8)
from kspace_normalization import compute_dcf_radial, KSpaceNormalizer
x = ds["x_all"]; y_raw = ds["y_all_raw"]; spoke = ds["spoke_id_all"]
keep = torch.as_tensor(np.load("/scratch/rnga/vvpshenov/DCE_NIK/spoke_masks/keep_f80match.npy"),
                       device=x.device, dtype=spoke.dtype)
tr_idx = torch.where(torch.isin(spoke, keep))[0]
dcf = compute_dcf_radial(x, method="simple_ramp")
nz = KSpaceNormalizer()
nz.fit(x[tr_idx], y_raw[tr_idx], dcf=dcf[tr_idx], envelope_exponent=0.75)
print(f"normalizer fit on {tr_idx.numel()} train samples", flush=True)
_cf = f"{OUT}/nik_cart_cache.npy"
if os.path.exists(_cf):
    cart = np.load(_cf); print("cart from cache", flush=True)
else:
    cart = A.reconstruct_cartesian(model, nz, OUT, device=dev.type, shared=sh, verbose=False)
    np.save(_cf, cart)
print(f"per-coil cartesian k-space {cart.shape}", flush=True)

img = ifft2c_mri(cart)                                        # [nx,nx,nt,ncc]
den = np.sum(np.abs(b1)**2, axis=2)[:, :, None] + 1e-12
rho = np.sum(img * np.conj(b1)[:, :, None, :], axis=3) / den  # SENSE combine -> one object
img_p = rho[:, :, :, None] * b1[:, :, None, :]                # re-expand: FORCED factorization
cart_p = fft2c_mri(img_p)

def db(a, b): return 20*np.log10(np.linalg.norm(b) / (np.linalg.norm(a-b) + 1e-30))
print(f"\nfactorization residual (per-coil k-space vs its factorized projection): {db(cart_p, cart):.2f} dB")

# fit to the MEASURED spokes, before vs after forcing factorization
nx_ = cart.shape[0]; nt = cart.shape[2]
T = np.asarray(sh["traj_grog"])                      # [nx, ntviews]
ixa = np.rint(np.real(T) + nx_/2).astype(int).clip(0, nx_-1)     # [nx, ntviews]
iya = np.rint(np.imag(T) + nx_/2).astype(int).clip(0, nx_-1)
fia = np.repeat(np.asarray(sh["frame_idx"]).astype(int)[None, :], T.shape[0], axis=0)  # per readout pt
sel = fia < nt
ix = ixa[sel]; iy = iya[sel]; fi = fia[sel]
Yc = ds["y_all_raw"].view(ncc, -1, 2)[..., 0] + 1j*ds["y_all_raw"].view(ncc, -1, 2)[..., 1]
Yc = Yc.cpu().numpy().T                                          # [M, C], M = nx*ntviews (fortran order)
meas = Yc.reshape(T.shape[0], T.shape[1], ncc, order="F")[sel]    # [P, C]
p0 = cart[ix, iy, fi, :]; p1 = cart_p[ix, iy, fi, :]
def nmse(p, y):
    sfac = np.vdot(p.ravel(), y.ravel()) / (np.vdot(p.ravel(), p.ravel()) + 1e-30)
    return float(np.linalg.norm(sfac*p - y) / (np.linalg.norm(y) + 1e-30))
n0, n1 = nmse(p0, meas), nmse(p1, meas)
print(f"data NMSE at measured spokes:  as-is {n0:.4f}   factorization-forced {n1:.4f}   "
      f"degradation {100*(n1-n0)/n0:+.1f}%")
print("\nREAD: small degradation -> factorization is nearly free, SENSE-forward B has headroom.")
print("      large degradation -> the network is using non-factorization to fit data.")
print("CHECK_DONE")
