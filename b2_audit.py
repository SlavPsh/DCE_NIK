"""B2: point-source round trip through every render path's deterministic centering, plus Parseval.
Report the measured peak offset for each path independently (paths did NOT agree - that is the point).
No model needed: centering lives in the FFT/crop/rot/NUFFT ops, tested with a delta."""
import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, "/scratch/rnga/vvpshenov/grasp_pro_py")
import numpy as np, cupy as cp, cufinufft
import fftc, xph_pipeline as P, xph_common as X, xph_grasp_nufft as GN
import recon_asserts as RA
RO = 220; p = (RO//2 + 7, RO//2 - 5)                       # delta location (offset from centre)
print(f"grid N={RO} (even), delta at {p}, centre N/2={RO//2}\n")

def ifft2c(Xk): return fftc.ifft2c_mri(Xk)
def fft2c(x):                                               # numerical inverse of ifft2c_mri
    Xk = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(x, 0), axis=0), 0)*np.sqrt(x.shape[0])
    Xk = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(Xk, 1), axis=1), 1)*np.sqrt(x.shape[1])
    return Xk
delta = np.zeros((RO, RO), np.complex128); delta[p] = 1.0

# --- fftc round trip (used by all NIK image renders) ---
K = fft2c(delta); back = ifft2c(K); pk = np.unravel_index(np.argmax(np.abs(back)), back.shape)
print(f"[fftc ifft2c(fft2c(delta))]  peak {pk}  offset {(pk[0]-p[0], pk[1]-p[1])}  | parseval rel {abs(np.sum(np.abs(delta)**2)-np.sum(np.abs(K)**2))/np.sum(np.abs(delta)**2):.2e}")

# --- rot180 centering: OLD flip vs FIXED roll(flip,1). centered rotation about N/2 -> peak at (N-p) ---
exp_c = (RO - p[0], RO - p[1])                             # centred-about-N/2 expectation
old = delta[::-1, ::-1]; new = np.roll(delta[::-1, ::-1], (1, 1), axis=(0, 1))
pkold = np.unravel_index(np.argmax(np.abs(old)), old.shape); pknew = np.unravel_index(np.argmax(np.abs(new)), new.shape)
print(f"[rot OLD im[::-1,::-1]]      peak {pkold}  vs centred {exp_c}  offset {(pkold[0]-exp_c[0], pkold[1]-exp_c[1])}  <- the bug (-1,-1)")
print(f"[rot FIXED roll(flip,1)]     peak {pknew}  vs centred {exp_c}  offset {(pknew[0]-exp_c[0], pknew[1]-exp_c[1])}")

# --- crop_img centering (identity here since grid==RO; test a larger-grid crop for convention) ---
big = np.zeros((RO+40, RO+40), np.complex128); pbig = (p[0]+20, p[1]+20); big[pbig] = 1.0
cr = fftc.crop_img(big[:, :, None], RO, RO)[:, :, 0]; pkc = np.unravel_index(np.argmax(np.abs(cr)), cr.shape)
print(f"[crop_img {RO+40}->{RO}]       delta {pbig} -> peak {pkc}  expected {p}  offset {(pkc[0]-p[0], pkc[1]-p[1])}")

# --- GRASP NUFFT operator round trip (E^H E delta), same trajs/sign as the GRASP baseline ---
d = P.data(); kx = d["kx"]; ky = d["ky"]; tr = np.array(P.TRAIN_ANG); F = kx.shape[0]
dg = cp.asarray(delta, cp.complex128)
p2 = cufinufft.Plan(2, (RO, RO), 1, eps=1e-6, isign=-1, dtype="complex128")
p1 = cufinufft.Plan(1, (RO, RO), 1, eps=1e-6, isign=1, dtype="complex128")
acc = cp.zeros((RO, RO), cp.complex128)
for t in range(0, F, 8):                                   # subsample frames; centring is frame-independent
    cox = cp.asarray(GN.SIGN*2*np.pi*(-kx[t, tr]).ravel(), cp.float64); coy = cp.asarray(GN.SIGN*2*np.pi*(-ky[t, tr]).ravel(), cp.float64)
    w = cp.asarray(np.maximum(np.abs(kx[t, tr]+1j*ky[t, tr]), 1e-3).ravel(), cp.float64)     # ramp dcf
    p2.setpts(cox, coy); ks = p2.execute(dg.reshape(1, RO, RO))[0]
    p1.setpts(cox, coy); acc += p1.execute((ks*w).reshape(1, -1))[0]
acc = cp.asnumpy(acc); pkg = np.unravel_index(np.argmax(np.abs(acc)), acc.shape)
print(f"[GRASP NUFFT E^H E delta]    peak {pkg}  expected {p}  offset {(pkg[0]-p[0], pkg[1]-p[1])}")
print("\nB2_AUDIT_DONE")
