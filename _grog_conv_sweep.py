import warnings; warnings.filterwarnings("ignore")
import sys, numpy as np
sys.path.insert(0, "/scratch/rnga/vvpshenov/grasp_pro_py")
import precompute_ref as pr
import xph_common as X
d = X.load_slice(15); kx = d["kx"]; ky = d["ky"]; kdata = d["kdata"]; C, F, nang, RO = kdata.shape
tr = [0, 1, 2, 3, 4]; NLINE = 5
kdc = kdata[:, :, tr, :].transpose(3, 1, 2, 0).reshape(RO, F * NLINE, C)
Tr = X.truth_at(15, d["times"]); tm = Tr.mean(2); body = d["labels"] > 0
def corr(a, b): a = a[body].ravel() - a[body].mean(); b = b[body].ravel() - b[body].mean(); return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
def bestcorr(im):
    cands = [im, im[::-1, ::-1], im[:, ::-1], im[::-1], im.T, im.T[::-1, ::-1], np.rot90(im), np.rot90(im, 3)]
    return max(corr(o, tm) for o in cands)
b = (kx[:, tr, :] + 1j * ky[:, tr, :]).transpose(2, 0, 1).reshape(RO, F * NLINE)
convs = {"kx+iky": b, "kx-iky": b.conj(), "-kx-iky": -b,
         "ky+ikx": (ky[:, tr, :] + 1j * kx[:, tr, :]).transpose(2, 0, 1).reshape(RO, F * NLINE),
         "-i(kx+iky)": -1j * b, "+i(kx+iky)": 1j * b}
for nm, tj in convs.items():
    Tg = (tj * RO).astype(np.complex128); kk = kdc[:, :, None, :]
    try:
        Gx, Gy = pr.get_gx_gy(kk, Tg)
        kref, _ = pr.grog_dictionary_interp(kk, Gx, Gy, Tg, 1)
        ref = np.abs(np.squeeze(pr.ifft2c_mri(kref)))
        ref = ref.sum(-1) if ref.ndim == 3 else ref
        if ref.shape[0] != RO: ref = pr.crop_img(ref[:, :, None], RO, RO)[:, :, 0] if ref.shape[0] > RO else ref
        print("%-12s sx=%d  ref-vs-truth bestcorr %.3f" % (nm, kref.shape[0], bestcorr(ref)), flush=True)
    except Exception as e:
        print("%-12s FAIL %s" % (nm, str(e)[:70]), flush=True)
