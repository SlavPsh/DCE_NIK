"""XCAT sparse-sim (20260527) -> NIK per-slice 2D radial dataset.

XCAT-SPECIFIC. this sim is an ALIGNED stack-of-stars: all 11 partitions share the same 9
angles per frame, so a separable partition IFFT (z-FFT) is VALID here. the real in-vivo
data is ROTATED stack-of-stars (angle depends on partition), where this is INVALID. do NOT
fold this into the real-data nik_adapter path.

layout: kspace.DCE (F,C,99,RO), 99 = 9 angles x 11 partitions (angle-major, partition
cycles 6,7,5,8,4,9,3,10,2,11,1). z-fft across partitions -> per image-slice 2D radial.
validated vs images.Recon: geometry corr 0.965, aorta TTP 25.2s / FWHM 43.6s (true 27.7 / 46.8).
"""
import h5py
import numpy as np

SIM = "/scratch/rnga/vvpshenov/XCAT-ERIC/results/simulation_results_20260527T175428.mat"
AORTA_LABEL = 36
NPART = 11
NANG = 9


def _load(path=SIM):
    f = h5py.File(path, "r"); r = f["results"]; ks = r["kspace"]
    dce = np.array(ks["DCE"]["real"]) + 1j * np.array(ks["DCE"]["imag"])   # (F,C,99,RO)
    traj = np.array(ks["trajDCE"])                                         # (F,3,99,RO)
    coil = np.array(ks["coilMaps"])                                        # (C,11,RO,RO) real
    tim = np.array(r["images"]["Recon"]["timing"]).ravel()                # (F,) seconds
    lab = np.array(r["labelGT"])                                          # (11,152,220)
    ref = np.abs(np.array(r["images"]["Recon"]["img"]))                   # (F,11,RO,152)
    return dce, traj, coil, tim, lab, ref


def slice_radial(zi, path=SIM):
    """z-fft to image-slice zi -> (kdata [C,F,NANG,RO], kx [F,NANG,RO], ky, times [F])."""
    dce, traj, coil, tim, lab, ref = _load(path)
    F, C, S, RO = dce.shape
    kd = np.zeros((C, F, NANG, RO), np.complex64)
    kx = np.zeros((F, NANG, RO), np.float32); ky = np.zeros((F, NANG, RO), np.float32)
    for fr in range(F):
        blk = dce[fr].reshape(C, NANG, NPART, RO)
        prt = traj[fr, 2].reshape(NANG, NPART, RO)[:, :, 0].astype(int)    # partition idx per (angle,part)
        tx = traj[fr, 0].reshape(NANG, NPART, RO); ty = traj[fr, 1].reshape(NANG, NPART, RO)
        for a in range(NANG):
            srt = np.argsort(prt[a])                                       # partitions 1..11
            zf = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(blk[:, a, srt, :], axes=1), axis=1), axes=1)
            kd[:, fr, a, :] = zf[:, zi, :]
            kx[fr, a] = tx[a, srt][0]; ky[fr, a] = ty[a, srt][0]          # angle shared across partitions
    return dict(kdata=kd, kx=kx, ky=ky, times=tim, RO=RO, coil=coil[:, zi], lab=lab, ref=ref[:, zi])


def aorta_roi(d, bas=None):
    """label-36 mask in the recon grid. lab is (152,220) in sim space -> RO x RO center-embedded."""
    RO = d["RO"]; L = d["lab"][5] if d["lab"].ndim == 3 else d["lab"]
    L = L.T if L.shape[0] != RO else L                                    # -> (RO,152)
    roi = np.zeros((RO, RO), bool); s = (RO - L.shape[1]) // 2
    roi[:, s:s + L.shape[1]] = (L == AORTA_LABEL)
    return roi


def true_kinetics(path=SIM, zi=5):
    """true aorta TTP + first-pass FWHM from GroundTruth.img, label 36."""
    f = h5py.File(path, "r"); r = f["results"]
    img = np.abs(np.array(r["images"]["GroundTruth"]["img"])[:, zi])      # (1801,RO,152)
    tim = np.array(r["images"]["GroundTruth"]["timing"]).ravel()
    L = np.array(r["labelGT"])[zi]; L = L.T if L.shape != img.shape[1:] else L
    c = img[:, L == AORTA_LABEL].mean(1); b = c[tim < 12].mean()
    n = (c - b) / (c.max() - b + 1e-9); ttp = float(tim[np.argmax(n)])
    half = (n > 0.5) & (tim < ttp + 40)
    return dict(ttp=ttp, fwhm=float(tim[half].max() - tim[half].min()), t=tim, curve=n)


# --- NIK training interface (mirrors nik_adapter.make_radial_dataset / reconstruct_cartesian) ---
import sys, torch
sys.path.insert(0, "/scratch/rnga/vvpshenov/grasp_pro_py")
from nik_output_recon import recon_nik_cart


def make_xcat_dataset(zi=5, path=SIM, device="cpu"):
    """XCAT slice zi -> nik training dict (x_all[M,2], t_all, coil_all, y_all_raw[M,2],
    spoke_id_all, b1[RO,RO,C], frame_t, meta). traj [-0.5,0.5] -> model coord 2*traj."""
    d = slice_radial(zi, path)
    kd = d["kdata"]; C, F, NA, RO = kd.shape; M = F * NA * RO
    kx = (2.0 * d["kx"].reshape(M)).astype(np.float32)             # model coord in [-1,1]
    ky = (2.0 * d["ky"].reshape(M)).astype(np.float32)
    Tt = float(d["times"].max()); tnorm = (2.0 * d["times"] / Tt - 1.0).astype(np.float32)  # [-1,1]
    frame_idx = np.repeat(np.arange(F), NA * RO)
    ang_idx = np.tile(np.repeat(np.arange(NA), RO), F)
    t_base = tnorm[frame_idx]; sp_base = (frame_idx * NA + ang_idx).astype(np.int64)
    x = np.stack([kx, ky], 1)
    x_all = np.tile(x, (C, 1)); t_all = np.tile(t_base, C)
    coil_all = np.repeat(np.arange(C, dtype=np.int64), M)
    y = kd.reshape(C, M)                                            # complex
    y_all_raw = np.stack([y.real, y.imag], -1).reshape(C * M, 2).astype(np.float32)
    spoke_id_all = np.tile(sp_base, C)
    b1 = np.transpose(d["coil"], (1, 2, 0)).astype(np.complex64)    # [RO,RO,C]
    def T(a, dt): return torch.as_tensor(a, dtype=dt, device=device)
    return dict(x_all=T(x_all, torch.float32), t_all=T(t_all, torch.float32),
                coil_all=T(coil_all, torch.long), y_all_raw=T(y_all_raw, torch.float32),
                spoke_id_all=T(spoke_id_all, torch.long), b1=b1,
                frame_t=tnorm, meta=dict(nx=RO, ncc=C, nt=F, bas=RO, Ttot=Tt, zi=zi))


def xcat_reconstruct(model, normalizer, meta, b1, frame_t, device="cuda", chunk=262144):
    """query the trained model on the RO x RO cartesian grid, all frames+coils -> image series."""
    nx = meta["nx"]; ncc = meta["ncc"]; nt = len(frame_t); dev = torch.device(device)
    k = (np.arange(nx) - nx // 2) / (nx // 2)
    gx, gy = np.meshgrid(k, k, indexing="ij")
    coords = torch.from_numpy(np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32)).to(dev)
    P = coords.shape[0]; cart = np.zeros((nx, nx, nt, ncc), np.complex64)
    model.eval()
    with torch.no_grad():
        for f in range(nt):
            tf = torch.full((P,), float(frame_t[f]), device=dev)
            for c in range(ncc):
                cc = torch.full((P,), c, device=dev, dtype=torch.long)
                pr = torch.cat([model(coords[s:s+chunk], tf[s:s+chunk], cc[s:s+chunk])
                                for s in range(0, P, chunk)], 0)
                raw = normalizer.denormalize(coords, pr)
                cart[:, :, f, c] = (raw[:, 0] + 1j * raw[:, 1]).cpu().numpy().reshape(nx, nx)
    return recon_nik_cart(cart, b1, meta["bas"])
