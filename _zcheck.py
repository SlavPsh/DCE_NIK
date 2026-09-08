import warnings; warnings.filterwarnings("ignore")
import numpy as np, h5py
import xph_common as X
zt = 15
d = X.load_slice(zt); RO = d["RO"]
kx = d["kx"]; ky = d["ky"]; kd = d["kdata"]; b1 = d["b1"]
print("traj range kx [%.3f,%.3f] ky [%.3f,%.3f]" % (kx.min(), kx.max(), ky.min(), ky.max()))
import finufft
C, F, nang, _ = kd.shape
kxa = kx.reshape(-1).astype(np.float64); kya = ky.reshape(-1).astype(np.float64)
sc = np.pi / max(abs(kx).max(), abs(ky).max())
kxa *= sc; kya *= sc
r2 = (kx.reshape(-1) ** 2 + ky.reshape(-1) ** 2) ** .5
dcf = (r2 + 1e-3).astype(np.complex128)
img = np.zeros((RO, RO), np.complex128)
for c in range(C):
    km = kd[c].reshape(-1).astype(np.complex128) * dcf
    im = finufft.nufft2d1(kxa, kya, km, (RO, RO))
    img += im * np.conj(b1[:, :, c])
rec = np.abs(img)
f = h5py.File(X.SIM, "r")
im3 = np.abs(np.array(f["results"]["images"]["GroundTruth"]["img"][:, zt])).mean(0)
tg = X._embed(im3, RO)
def corr(a, b):
    a = a.ravel() - a.mean(); b = b.ravel() - b.mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
print("recon(z15) vs truth(z15): identity %.3f  rot180 %.3f" % (corr(rec, tg), corr(rec[::-1, ::-1], tg)))
