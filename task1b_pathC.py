"""TASK 1B Path C: direct learned-amplitude extraction + coil-identifiability (F2 sl21).
Query model.amplitudes(k,coil) on the cartesian grid, denormalize with the (re-fit, deterministic)
normalizer, IFFT per coil, SENSE-combine per basis component -> theta_r^C. Verify theta_C == theta_B
(complex projection) and reconstruct-consistency; test whether per-coil amplitudes correspond to a
common coil-independent map. Read-only; no model/training/checkpoint change."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, os, sys, torch
from types import SimpleNamespace
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/net/beegfs/users/P101440/grasp_pro_py"); sys.path.insert(0, ".")
import consolidated as C, nik_adapter as A
from train_grasp_nik import build_model
from kspace_normalization import KSpaceNormalizer, compute_dcf_radial
from nik_output_recon import recon_nik_cart
D = "/net/beegfs/users/P101440/DCE_NIK"; REF = "/net/beegfs/users/P101440/grasp_pro_py/results_ref"; OUT = f"{D}/results/task1b_coefficient_extraction"
Z, F = 21, 2; dev = "cpu"
sh = np.load(f"{REF}/shared.npz"); TA = float(sh["TA"]); nx, nt, ncc = int(sh["nx"]), int(sh["nt"]), int(sh["ncc"]); bas = int(sh["bas"])
frame_t = torch.tensor((2.0 * sh["frame_time"] - 1.0).astype(np.float32))

ck = torch.load(f"{D}/results_batch/pk_f{F}_sl{Z}/model_slice_{Z:02d}.pt", map_location="cpu", weights_only=False)
args = SimpleNamespace(model="wire_ff_patlak", patlak_free=F, aif_file=f"{D}/aif_slice{Z}.npz",
    coil_embed_dim=ck["coil_embed_dim"], hidden=ck["hidden"], depth=ck["depth"], w0=ck["w0"], s0=ck["s0"],
    k_freq=ck["k_freq"], k_sigma=ck["k_sigma"], t_freq=ck["t_freq"], t_sigma=ck["t_sigma"], ff_seed=ck["ff_seed"],
    phi_hidden=64, phi_depth=3, phi_w0=30.0)
model = build_model(args, ncc); model.load_state_dict(ck["state_dict"], strict=True); model.eval()

# --- re-fit the deterministic normalizer exactly as train_one_slice (keep_f25, envelope 0.75) ---
ds = A.make_radial_dataset(REF, Z, compute_device=dev, shared=sh)
x, y_raw, spoke_id = ds["x_all"], ds["y_all_raw"], ds["spoke_id_all"]; b1 = ds["b1"]
kept = torch.as_tensor(np.load(f"{D}/spoke_masks/keep_f25.npy"), dtype=spoke_id.dtype)
tr = torch.where(torch.isin(spoke_id, kept))[0]
dcf = compute_dcf_radial(x, method="simple_ramp")
norm = KSpaceNormalizer(); norm.fit(x[tr], y_raw[tr], dcf=dcf[tr], envelope_exponent=0.75)
print(f"normalizer re-fit on {tr.numel()} kept-spoke samples", flush=True)

# --- Path C: per-component amplitude -> denorm -> coil images -> SENSE combine ---
coords = torch.from_numpy(A.cartesian_grid(nx)).to(dev); Pn = coords.shape[0]
R = model.rank
cart_r = np.zeros((nx, nx, R, ncc), np.complex64)
with torch.no_grad():
    for c in range(ncc):
        cc = torch.full((Pn,), c, dtype=torch.long)
        Amp = model.amplitudes(coords, cc)                      # [P,R,2]
        for r in range(R):
            praw = norm.denormalize(coords, Amp[:, r, :].contiguous())   # denorm this component's k-space
            v = (praw[:, 0] + 1j * praw[:, 1]).numpy().reshape(nx, nx)
            cart_r[:, :, r, c] = v
# support mask like the recon
rr = np.sqrt(A.cartesian_grid(nx)[:, 0] ** 2 + A.cartesian_grid(nx)[:, 1] ** 2).reshape(nx, nx)
cart_r[rr > 1.0] = 0
_, thetaC = recon_nik_cart(cart_r, b1.numpy() if torch.is_tensor(b1) else b1, bas, return_complex=True)  # [bas,bas,R] complex

# --- Path B (complex projection) for comparison ---
Ic = np.load(f"{D}/results_batch/pk_f{F}_sl{Z}/nik_slice_{Z}_cplx.npy").astype(np.complex64)
with torch.no_grad(): Pm = torch.view_as_complex(model.basis(frame_t).contiguous()).numpy()  # [nt,R]
thetaB = np.einsum("rt,xyt->xyr", np.linalg.pinv(Pm), Ic)

# --- validation: theta_C vs theta_B, and reconstruct-consistency ---
ctx = C.slice_ctx(Z); body = ctx["BODY"]
def nrmse(a, b, m): return float(np.linalg.norm(a[m] - b[m]) / (np.linalg.norm(b[m]) + 1e-12))
IhatC = np.einsum("xyr,tr->xyt", thetaC, Pm)
print(f"Path C vs Path B (theta, body): AIF-coeff NRMSE {nrmse(thetaC[...,0], thetaB[...,0], body):.3f} | intAIF NRMSE {nrmse(thetaC[...,1], thetaB[...,1], body):.3f}")
print(f"Path C reconstruct-consistency (complex, body): NRMSE(sum thetaC Phi, I_c) = {nrmse(IhatC, Ic, body):.3e}")

# --- identifiability: is per-coil amplitude image a_0,c ~ S_c * theta_0 (common map)? ---
# per-coil amplitude image for component r=0 (AIF)
from fftc import ifft2c_mri, crop_img
img0 = ifft2c_mri(cart_r[:, :, 0, :])                            # [nx,nx,ncc] per-coil amplitude image
a0c = crop_img(img0, bas, bas)                                   # [bas,bas,ncc]
b1c = crop_img(b1.numpy() if torch.is_tensor(b1) else b1, bas, bas)  # [bas,bas,ncc]
hi = body & (np.abs(thetaC[..., 0]) > np.percentile(np.abs(thetaC[..., 0])[body], 80))
# if identifiable: a0c[:,:,c] ~ b1c[:,:,c] * thetaC0  -> ratio a0c/(b1c*thetaC0) coil-consistent
pred = b1c * thetaC[..., 0][:, :, None]
coilcorr = [float(np.corrcoef(np.abs(a0c[..., c])[hi], np.abs(pred[..., c])[hi])[0, 1]) for c in range(ncc)]
print(f"identifiability: corr(|a_0,c|, |S_c*theta_0|) per coil (hi-signal) = {np.round(coilcorr,2)}  mean {np.mean(coilcorr):.2f}")

np.save(f"{OUT}/arrays/F2_sl21_thetaC_AIF.npy", np.abs(thetaC[..., 0]))
np.save(f"{OUT}/arrays/F2_sl21_thetaC_intAIF.npy", np.abs(thetaC[..., 1]))
# figure: per-coil amplitude images (r=0) + SENSE combine vs Path B
fig, ax = plt.subplots(2, 4, figsize=(15, 7.5))
for c in range(4):
    im = ax[0, c].imshow(np.rot90(np.abs(a0c[..., c])), cmap="magma"); ax[0, c].axis("off"); ax[0, c].set_title(f"|a_0,coil{c}| (per-coil ampl.)", fontsize=9)
ax[1, 0].imshow(np.rot90(np.abs(thetaC[..., 1])), cmap="viridis", vmax=np.percentile(np.abs(thetaC[..., 1]), 99)); ax[1, 0].axis("off"); ax[1, 0].set_title("Path C intAIF (SENSE)", fontsize=9)
ax[1, 1].imshow(np.rot90(np.abs(thetaB[..., 1])), cmap="viridis", vmax=np.percentile(np.abs(thetaB[..., 1]), 99)); ax[1, 1].axis("off"); ax[1, 1].set_title("Path B intAIF", fontsize=9)
d = np.abs(thetaC[..., 1]) - np.abs(thetaB[..., 1]); vv = np.percentile(np.abs(d), 99)
ax[1, 2].imshow(np.rot90(d), cmap="bwr", vmax=vv, vmin=-vv); ax[1, 2].axis("off"); ax[1, 2].set_title("C - B diff", fontsize=9)
ax[1, 3].bar(range(ncc), coilcorr); ax[1, 3].set_title("corr(|a_0,c|,|S_c theta_0|)/coil", fontsize=9); ax[1, 3].set_ylim(0, 1); ax[1, 3].grid(alpha=.3, axis="y")
fig.suptitle("TASK 1B Path C: per-coil amplitudes, SENSE combine == Path B, identifiability (F2 sl21)", fontweight="bold")
fig.tight_layout(); fig.savefig(f"{OUT}/figures/F2_sl21_pathC.png", dpi=120); plt.close(fig)
print("wrote Path C outputs")
