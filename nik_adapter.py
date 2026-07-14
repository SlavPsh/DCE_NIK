"""adapter: feed grasp-pro precomputed twix data into the WIRE NIK pipeline.

reads the artifacts written by grasp_pro_py/precompute_ref.py (shared.npz + slice_XX.npz)
and emits training samples with the SAME conventions as make_multicoil_time_radial_dataset:
    x_all     (N,2)  kx,ky in [-0.5,0.5]   (traj_norm = GROG traj / nx)
    t_all     (N,)   time in [-1,1]         (per-spoke continuous: 2*view_time-1)
    coil_all  (N,)   long, 0..ncc-1         (compressed virtual coils; matches saved b1)
    y_all_raw (N,2)  Re/Im, RAW             (DCF + KSpaceNormalizer applied downstream)

the target lives in the same 8 virtual-coil space as the saved coil maps b1, so model
output and coil combination are consistent. train one 2D model per z-slice (the z-ifft
is already done in the precompute), i.e. use WIRE_KXY_COIL_T_REIM with n_coils=ncc.

reconstruct_cartesian() queries the trained model on the cartesian grid that matches
traj_norm and returns [nx,nx,nt,ncc] complex k-space; hand that to
grasp_pro_py/nik_recon.recon_nik_cart(nik_cart, b1, bas) for an image comparable to the
CS reference cs_recon.npy.
"""
import os
import numpy as np
import torch


def load_shared(out_dir):
    d = np.load(os.path.join(out_dir, 'shared.npz'))
    return {k: d[k] for k in d.files}


def load_slice(out_dir, slice_idx):
    d = np.load(os.path.join(out_dir, f'slice_{slice_idx:02d}.npz'))
    return {k: d[k] for k in d.files}


def make_radial_dataset(out_dir, slice_idx, *, compute_device='cuda', shared=None):
    """build the per-slice radial training set from precompute artifacts.
    returns dict with x_all,t_all,coil_all,y_all_raw (+ spoke/ro/frame ids, meta)."""
    sh = shared if shared is not None else load_shared(out_dir)
    sl = load_slice(out_dir, slice_idx)

    traj = sh['traj_norm']                       # (nx, ntviews) complex, [-0.5,0.5]
    view_time = sh['view_time'].astype(np.float32)
    frame_idx = sh['frame_idx'].astype(np.int64)
    krad = sl['kdata_radial']                    # (nx, ntviews, ncc) complex
    nx, ntviews, ncc = krad.shape
    M = nx * ntviews

    # coords in [-1,1] to match the sim convention (kx = traj / max|traj| = traj_norm*2)
    kx = (2.0 * np.real(traj)).reshape(-1).astype(np.float32)     # (M,) order (r,v) C-major
    ky = (2.0 * np.imag(traj)).reshape(-1).astype(np.float32)
    rr, vv = np.meshgrid(np.arange(nx), np.arange(ntviews), indexing='ij')
    ro_rv = rr.reshape(-1).astype(np.int64)
    sp_rv = vv.reshape(-1).astype(np.int64)
    t_rv = (2.0 * view_time - 1.0)[sp_rv]                        # per-spoke time in [-1,1]
    fr_rv = frame_idx[sp_rv]

    # tile across coils (coil outer), target ordered (c, r, v) to match
    x_all = np.tile(np.stack([kx, ky], 1), (ncc, 1))            # (ncc*M, 2)
    t_all = np.tile(t_rv, ncc)
    coil_all = np.repeat(np.arange(ncc, dtype=np.int64), M)
    spoke_id = np.tile(sp_rv, ncc)
    ro_id = np.tile(ro_rv, ncc)
    frame_id = np.tile(fr_rv, ncc)
    y = np.transpose(krad, (2, 0, 1)).reshape(ncc * M)          # (c,r,v) flattened
    y_all_raw = np.stack([y.real, y.imag], 1).astype(np.float32)

    dev = torch.device(compute_device)
    to = lambda a, d: torch.from_numpy(np.ascontiguousarray(a)).to(dev, dtype=d)
    meta = dict(N=ncc * M, nx=int(nx), ntviews=int(ntviews), ncc=int(ncc),
                nt=int(sh['nt']), nline=int(sh['nline']), bas=int(sh['bas']),
                slice_idx=int(slice_idx))
    return dict(
        x_all=to(x_all, torch.float32), t_all=to(t_all, torch.float32),
        coil_all=to(coil_all, torch.long), y_all_raw=to(y_all_raw, torch.float32),
        spoke_id_all=to(spoke_id, torch.long), ro_id_all=to(ro_id, torch.long),
        frame_id_all=to(frame_id, torch.long), meta=meta, b1=sl['b1'])


def cartesian_grid(nx):
    """cartesian k-grid in [-1,1) matching the training coords (kx = traj_norm*2).
    returns coords (nx*nx,2)."""
    k = (np.arange(nx) - nx // 2) / (nx // 2)
    kx, ky = np.meshgrid(k, k, indexing='ij')
    return np.stack([kx.reshape(-1), ky.reshape(-1)], 1).astype(np.float32)


@torch.no_grad()
def reconstruct_cartesian(model, normalizer, out_dir, *, device='cuda',
                          shared=None, chunk=262144, support_radius=1.0, verbose=True):
    """query the trained model on the cartesian grid for every frame and coil.
    returns nik_cart [nx,nx,nt,ncc] complex64 (raw, denormalized k-space).

    support_radius: radial k-space support of the spokes (normalized; readout reaches
    ~1.0). grid points with sqrt(kx^2+ky^2) > support_radius were never sampled by any
    spoke, so the model's prediction there is pure extrapolation; they are zeroed to
    match the GROG/CS arm (mask=0 outside the sampled disk). set None to disable.
    pass to grasp_pro_py/nik_recon.recon_nik_cart(nik_cart, b1, bas) for the image."""
    sh = shared if shared is not None else load_shared(out_dir)
    nx, nt, ncc = int(sh['nx']), int(sh['nt']), int(sh['ncc'])
    frame_t = (2.0 * sh['frame_time'] - 1.0).astype(np.float32)     # (nt,) in [-1,1]
    coords_np = cartesian_grid(nx)                                  # (P,2)
    P = coords_np.shape[0]
    dev = torch.device(device)
    coords = torch.from_numpy(coords_np).to(dev)
    cart = np.zeros((nx, nx, nt, ncc), dtype=np.complex64)

    model.eval()
    for f in range(nt):
        tf = torch.full((P,), float(frame_t[f]), device=dev, dtype=torch.float32)
        for c in range(ncc):
            cc = torch.full((P,), c, device=dev, dtype=torch.long)
            pred = torch.empty((P, 2), device=dev, dtype=torch.float32)
            for s in range(0, P, chunk):
                e = min(s + chunk, P)
                pred[s:e] = model(coords[s:e], tf[s:e], cc[s:e])
            praw = normalizer.denormalize(coords, pred)             # (P,2) raw Re/Im
            v = praw[:, 0].cpu().numpy() + 1j * praw[:, 1].cpu().numpy()
            cart[:, :, f, c] = v.reshape(nx, nx).astype(np.complex64)
        if verbose and (f % 20 == 0 or f == nt - 1):
            print(f'  cartesian query frame {f+1}/{nt}', flush=True)

    if support_radius is not None:
        r = np.sqrt(coords_np[:, 0] ** 2 + coords_np[:, 1] ** 2).reshape(nx, nx)
        cart[r > support_radius] = 0                                # zero unsampled corners
        if verbose:
            kept = float((r <= support_radius).mean())
            print(f'  support mask: kept {kept:.3f} of grid (radius<={support_radius})', flush=True)
    return cart
