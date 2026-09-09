"""TASK 2 heavy step: query per-coil image-domain coefficient fields a_r,c(x) (Path C) for every
checkpoint/slice. rebuild model (read-only), re-fit the deterministic normalizer, query
model.amplitudes on the Cartesian grid per coil, denormalize, support-mask, IFFT, crop.
Saves complex a_rc [bas,bas,R,ncc], b1_crop, Phi(frame_t), and scale constants. No model change."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, os, sys, torch, json
from types import SimpleNamespace
sys.path.insert(0, "/net/beegfs/users/P101440/grasp_pro_py"); sys.path.insert(0, ".")
import nik_adapter as A
from train_grasp_nik import build_model
from kspace_normalization import KSpaceNormalizer, compute_dcf_radial
from fftc import ifft2c_mri, crop_img
D = "/net/beegfs/users/P101440/DCE_NIK"; REF = "/net/beegfs/users/P101440/grasp_pro_py/results_ref"
OUT = f"{D}/results/task2_scaling_coil_audit"; os.makedirs(f"{OUT}/arrays", exist_ok=True); os.makedirs(f"{OUT}/logs", exist_ok=True)
sh = np.load(f"{REF}/shared.npz"); TA = float(sh["TA"]); nx, nt, ncc, bas = int(sh["nx"]), int(sh["nt"]), int(sh["ncc"]), int(sh["bas"])
frame_t = torch.tensor((2.0 * sh["frame_time"] - 1.0).astype(np.float32)); dev = "cpu"
coords = torch.from_numpy(A.cartesian_grid(nx)); rr = np.sqrt(A.cartesian_grid(nx)[:, 0] ** 2 + A.cartesian_grid(nx)[:, 1] ** 2).reshape(nx, nx)
scales = {}
for F in [0, 2]:
    for Z in [18, 19, 21]:
        ck = torch.load(f"{D}/results_batch/pk_f{F}_sl{Z}/model_slice_{Z:02d}.pt", map_location="cpu", weights_only=False)
        args = SimpleNamespace(model="wire_ff_patlak", patlak_free=F, aif_file=f"{D}/aif_slice{Z}.npz",
            coil_embed_dim=ck["coil_embed_dim"], hidden=ck["hidden"], depth=ck["depth"], w0=ck["w0"], s0=ck["s0"],
            k_freq=ck["k_freq"], k_sigma=ck["k_sigma"], t_freq=ck["t_freq"], t_sigma=ck["t_sigma"], ff_seed=ck["ff_seed"],
            phi_hidden=64, phi_depth=3, phi_w0=30.0)
        model = build_model(args, ncc); model.load_state_dict(ck["state_dict"], strict=True); model.eval()
        R = model.rank
        ds = A.make_radial_dataset(REF, Z, compute_device=dev, shared=sh)
        x, y_raw, spoke_id, b1 = ds["x_all"], ds["y_all_raw"], ds["spoke_id_all"], ds["b1"]
        kept = torch.as_tensor(np.load(f"{D}/spoke_masks/keep_f25.npy"), dtype=spoke_id.dtype)
        tr = torch.where(torch.isin(spoke_id, kept))[0]
        norm = KSpaceNormalizer(); norm.fit(x[tr], y_raw[tr], dcf=compute_dcf_radial(x, method="simple_ramp")[tr], envelope_exponent=0.75)
        b1np = b1.numpy() if torch.is_tensor(b1) else b1
        cart_r = np.zeros((nx, nx, R, ncc), np.complex64)
        with torch.no_grad():
            for c in range(ncc):
                cc = torch.full((coords.shape[0],), c, dtype=torch.long)
                Amp = model.amplitudes(coords, cc)                       # [P,R,2]
                for r in range(R):
                    praw = norm.denormalize(coords, Amp[:, r, :].contiguous())
                    cart_r[:, :, r, c] = (praw[:, 0] + 1j * praw[:, 1]).numpy().reshape(nx, nx)
        cart_r[rr > 1.0] = 0
        a_rc = crop_img(ifft2c_mri(cart_r.reshape(nx, nx, R * ncc)).reshape(nx, nx, R, ncc), bas, bas)  # [bas,bas,R,ncc]
        b1c = crop_img(b1np, bas, bas)
        with torch.no_grad(): Phi = torch.view_as_complex(model.basis(frame_t).contiguous()).numpy()
        np.save(f"{OUT}/arrays/a_rc_F{F}_sl{Z}.npy", a_rc.astype(np.complex64))
        np.save(f"{OUT}/arrays/b1c_sl{Z}.npy", b1c.astype(np.complex64))
        np.save(f"{OUT}/arrays/Phi_F{F}_sl{Z}.npy", Phi.astype(np.complex64))
        az = np.load(f"{D}/aif_slice{Z}.npz"); tg = np.linspace(0, TA, nt)
        aiff = np.interp(tg, az["tC"], az["aif_frame"]); alpha = float(aiff.max())
        integ = np.concatenate([[0], np.cumsum(0.5 * (aiff[1:] + aiff[:-1]) * np.diff(tg))]); beta = float(integ.max())
        scales[f"F{F}_sl{Z}"] = dict(global_scale=float(norm.global_scale), envelope_exponent=0.75,
                                     alpha_aif_peak=alpha, beta_iaif_max=beta, R=R, ncc=ncc, dt_grid_s=float(np.diff(tg).mean()))
        print(f"F{F} sl{Z}: a_rc {a_rc.shape} | global_scale {norm.global_scale:.4g} alpha {alpha:.4g} beta {beta:.4g}", flush=True)
json.dump(scales, open(f"{OUT}/arrays/scales.json", "w"), indent=1)
print("QUERY DONE")
