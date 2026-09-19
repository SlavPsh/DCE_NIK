#!/bin/bash
# where the 1e-5 image scale comes from: print the magnitude at every stage of one frame's adjoint
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH PYTHONWARNINGS=ignore
timeout 50 micromamba run -n torch29 python -u - <<'PY'
import numpy as np, finufft, sys
R = "/net/beegfs/users/P101440/dce_data/orig/gv2_DCE_Rerun"
sh = np.load(f"{R}/prep/shared.npz"); d = np.load(f"{R}/prep/slice_25.npz")
k = np.asarray(d["kdata_radial"]); b1 = np.asarray(d["b1"]).astype(np.complex128)
T = np.asarray(sh["traj_grog"]); NX, NGRID = T.shape[0], int(sh["nx"]); NL = 21
print(f"prep kdata_radial {k.shape}  b1 {b1.shape}  nx {NX} grid {NGRID}")
print(f"  |k| at readout centre (dc, mean over spokes/coils) {np.abs(k[NX//2]).mean():.4e}")
print(f"  |k| mean over everything                            {np.abs(k).mean():.4e}")
print(f"  |k| max                                             {np.abs(k).max():.4e}")
# one frame, exactly as the recon builds it
kd = k[:, :NL, :].astype(np.complex128); tj = T[:, :NL]
kc = tj / NGRID; w = np.maximum(np.abs(kc), 1.0 / NX / 4.0)
print(f"  ramp dcf w: min {w.min():.4e} mean {w.mean():.4e} max {w.max():.4e}; mean sqrt(w) {np.sqrt(w).mean():.4f}")
y = kd * np.sqrt(w)[:, :, None]
print(f"  after data x sqrt(w):      mean |y| {np.abs(y).mean():.4e}")
yy = y * np.sqrt(w)[:, :, None]                      # adjoint applies sqrt(w) again -> full dcf
omx = np.ascontiguousarray((2*np.pi*np.real(kc)).ravel(order="F")); omy = np.ascontiguousarray((2*np.pi*np.imag(kc)).ravel(order="F"))
c = np.ascontiguousarray(yy.transpose(2, 0, 1).reshape(b1.shape[2], -1))
im = finufft.nufft2d1(omx, omy, c, (NX, NX), isign=-1, eps=1e-6)
print(f"  type-1 sum over {omx.size} points:  max |coil img| {np.abs(im).max():.4e}")
scale = np.sqrt(NX * NX); usf = NX * np.pi / 2.0 / NL
im2 = im / scale;  print(f"  / sqrt(Nx*Ny) = /{scale:.0f}:         max {np.abs(im2).max():.4e}")
im3 = im2 * usf;   print(f"  x usf = Nx*pi/2/nspokes = x{usf:.1f}: max {np.abs(im3).max():.4e}")
den = (np.abs(b1)**2).sum(-1)
out = (im3 * np.conj(b1.transpose(2, 0, 1))).sum(0) / den
print(f"  coil combine / den (den {den.min():.3f}..{den.max():.3f}): max |image| {np.abs(out).max():.4e}")
print(f"  -> one frame, one slice, max {np.abs(out).max():.4e}   (aorta baseline in the recon ~9e-6)")
print(f"NOTE z-ifft in prep divided by lPartitions/slice_res = {int(sh['nz_recon'])} (numpy ifft 1/n), applied before all of this")
PY
