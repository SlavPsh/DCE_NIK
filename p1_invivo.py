"""P1: MEASURED point-source round trip through the in-vivo render (nik_recon.nufft2d_recon), not a
code-read. Forward a delta at pixel p to a synthetic radial trajectory in the in-vivo convention
(kx_pi = (traj/scale)*pi), then run the actual nufft2d_recon adjoint and measure the peak location.
Centered + no-flip => peak at p. Centered 180-rot => peak at (N-p). Off-by-one => the bug family."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, cupy as cp, cufinufft, torch
import nik_recon
N = 128; p = (N//2 + 9, N//2 - 6)                         # even grid, delta offset from centre
# synthetic radial stack-of-stars: 201 golden-angle spokes, RO=N, kx_norm in [-1,1] (full Nyquist)
nsp = 201; RO = N; ga = np.pi*(3-np.sqrt(5))
r = (np.arange(RO) - RO//2) / (RO//2)                     # [-1, ~1)
ang = (np.arange(nsp)*ga) % np.pi
kx = np.outer(np.cos(ang), r).astype(np.float64)          # [nsp, RO], in [-1,1]
ky = np.outer(np.sin(ang), r).astype(np.float64)
# forward: delta image -> nonuniform k at the SAME kx_pi nufft2d_recon will use
delta = np.zeros((N, N), np.complex64); delta[p] = 1.0
kx_pi = (kx*np.pi).ravel(); ky_pi = (ky*np.pi).ravel()
p2 = cufinufft.Plan(nufft_type=2, n_modes=(N, N), eps=1e-6, isign=1, dtype="complex64")
p2.setpts(cp.asarray(kx_pi, cp.float32), cp.asarray(ky_pi, cp.float32))
ksynth = cp.asnumpy(p2.execute(cp.asarray(delta))).reshape(nsp, RO)
# pack into nik_recon's expected tensors and call the ACTUAL render
traj_t = torch.zeros((1, nsp, 3, RO))
traj_t[0, :, 0, :] = torch.from_numpy(kx); traj_t[0, :, 1, :] = torch.from_numpy(ky)  # kz=0 -> n_slices=1
k_img_space = torch.zeros((1, nsp, 1, 1, RO), dtype=torch.complex64)
k_img_space[0, :, 0, 0, :] = torch.from_numpy(ksynth)
img = nik_recon.nufft2d_recon(k_img_space, traj_t, t_frame=0, coil_idx=0, z_slice_idx=0,
                              scales=(1.0, 1.0, 1.0), img_size=(N, N), n_slices=1, return_complex=True)
pk = np.unravel_index(np.argmax(np.abs(img)), img.shape)
off_p = (pk[0]-p[0], pk[1]-p[1]); off_rot = (pk[0]-(N-p[0]), pk[1]-(N-p[1]))
print(f"in-vivo nufft2d_recon: delta at {p} (N={N}) -> peak {pk}")
print(f"  offset vs p (no-flip centred)      = {off_p}")
print(f"  offset vs N-p (180-rot centred)    = {off_rot}")
verdict = "CENTERED (peak==p)" if off_p == (0, 0) else ("CENTERED-180ROT (peak==N-p)" if off_rot == (0, 0) else f"OFF-CENTRE {min(off_p, off_rot, key=lambda o: abs(o[0])+abs(o[1]))}")
print(f"  VERDICT: {verdict}")
print("P1_INVIVO_DONE")
