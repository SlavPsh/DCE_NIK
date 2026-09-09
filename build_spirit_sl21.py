"""build a slice-21 SPIRiT kernel (coil-consistency G) from sl21's own ACS. per-coil static
NUFFT -> coil images -> FFT -> cartesian k-space center 24x24 -> calibrate_spirit. out: spirit_kernel_sl21.npz"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, finufft, json
from spirit import calibrate_spirit
REF = "/net/beegfs/users/P101440/grasp_pro_py/results_ref"; D = "/net/beegfs/users/P101440/DCE_NIK"; Z = 21; ACS = 24; KS = 5
sh = np.load(f"{REF}/shared.npz"); traj = np.asarray(sh["traj_norm"]).astype(np.complex64); nx = int(sh["nx"])
sl = np.load(f"{REF}/slice_{Z:02d}.npz"); kdata = np.asarray(sl["kdata_radial"]).astype(np.complex64); ncc = kdata.shape[2]
meta = json.load(open(f"{D}/results_nufft_slice{Z}/meta.json")); SIGN = meta["sign"]
w = np.maximum(np.abs(traj).astype(np.float64), (1.0/nx)/4.0)             # ramp dcf
x = (SIGN*2*np.pi*np.real(traj)).ravel().astype(np.float64); y = (SIGN*2*np.pi*np.imag(traj)).ravel().astype(np.float64)
cart = np.zeros((nx, nx, ncc), np.complex64)
for c in range(ncc):
    img = finufft.nufft2d1(x, y, (kdata[:, :, c]*w).astype(np.complex128).ravel(), (nx, nx), eps=1e-6, isign=1)  # coil image
    cart[:, :, c] = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img))).astype(np.complex64)                     # -> cartesian k-space
c0 = nx//2; blk = cart[c0-ACS//2:c0+ACS//2, c0-ACS//2:c0+ACS//2, :]; acs = np.transpose(blk, (2, 0, 1))          # [ncc,ACS,ACS]
G = calibrate_spirit(acs, ksize=KS, lam=1e-2)
np.savez(f"{D}/spirit_kernel_sl21.npz", G=G, ksize=KS, acs=ACS)
print(f"sl21 SPIRiT kernel G{G.shape} from ACS {ACS}x{ACS}, {ncc} coils -> spirit_kernel_sl21.npz")
