"""Reconstruction invariant harness. Call check_recon(...) after EVERY recon; failures RAISE.
Motivated by the 1px even-grid rot180 bug (PITFALLS 11): a translation is invisible to magnitude
spectra, held-out k-space NMSE, and correlation-based curve metrics, and degrades SSIM smoothly
rather than obviously. These invariants catch that class automatically.

A1 geometry   : shift < 0.1px, scale within 1e-3, rotation within 0.1deg vs a DIFFERENT-code-path ref.
A2 point-src  : a delta pushed through a render path returns at the input pixel (needs no reference).
A3 parseval   : k-space energy == image energy within tol (normalization/scale slips).
A4 orientation: best D4 transform vs ref is identity (no flip/transpose).
A5 sanity     : shape / dtype / finite on arrays crossing module boundaries.
"""
import numpy as np
from scipy.ndimage import fourier_shift, map_coordinates

class ReconAssertError(AssertionError): pass

# ---------- A5 ----------
def check_array(a, name, ndim=None, dtype_kind=None, finite=True):
    a = np.asarray(a)
    if ndim is not None and a.ndim != ndim: raise ReconAssertError(f"A5 {name}: ndim {a.ndim} != {ndim}")
    if dtype_kind is not None and a.dtype.kind not in dtype_kind: raise ReconAssertError(f"A5 {name}: dtype {a.dtype} not in {dtype_kind}")
    if finite and not np.all(np.isfinite(a)): raise ReconAssertError(f"A5 {name}: non-finite ({np.mean(~np.isfinite(a))*100:.2f}% NaN/Inf)")
    return a

# ---------- A1 ----------
def _fshift(im, sy, sx): return np.real(np.fft.ifft2(fourier_shift(np.fft.fft2(im), (sy, sx))))
def _ncc(a, b, m):
    a = a[m]-a[m].mean(); b = b[m]-b[m].mean(); return float((a*b).sum()/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
def measure_shift(mov, ref, mask=None, rng=2.0):
    if mask is None: mask = np.ones(ref.shape, bool)
    best = (-9, 0.0, 0.0)
    for sy in np.arange(-rng, rng+1e-9, 0.05):
        for sx in np.arange(-rng, rng+1e-9, 0.05):
            c = _ncc(_fshift(mov, sy, sx), ref, mask)
            if c > best[0]: best = (c, sy, sx)
    _, sy0, sx0 = best
    for sy in np.arange(sy0-0.06, sy0+0.061, 0.01):
        for sx in np.arange(sx0-0.06, sx0+0.061, 0.01):
            c = _ncc(_fshift(mov, sy, sx), ref, mask)
            if c > best[0]: best = (c, sy, sx)
    return best[1], best[2], best[0]
def measure_scale_rot(ref, mov):
    def spec(im):
        w = np.outer(np.hanning(im.shape[0]), np.hanning(im.shape[1])); return np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(im*w))))
    S1, S2 = spec(ref), spec(mov); c = np.array(S1.shape)//2; rmax = min(c); nr, nt = 256, 360
    RR, TT = np.meshgrid(np.exp(np.linspace(0, np.log(rmax), nr)), np.linspace(0, np.pi, nt), indexing="ij")
    ys = c[0]+RR*np.sin(TT); xs = c[1]+RR*np.cos(TT)
    L1 = map_coordinates(S1, [ys.ravel(), xs.ravel()], order=1).reshape(nr, nt)
    L2 = map_coordinates(S2, [ys.ravel(), xs.ravel()], order=1).reshape(nr, nt)
    R = np.fft.fft2(L1)*np.conj(np.fft.fft2(L2)); R /= np.abs(R)+1e-12
    pk = np.unravel_index(np.argmax(np.abs(np.fft.ifft2(R))), (nr, nt))
    ds = pk[0]-nr if pk[0] > nr/2 else pk[0]; dt = pk[1]-nt if pk[1] > nt/2 else pk[1]
    return float(np.exp(-ds*np.log(rmax)/nr)), float(-dt*180.0/nt)
def check_geometry(recon2d, ref2d, mask=None, tol_shift=0.1, tol_scale=1e-3, tol_rot=0.1, name="geom"):
    sy, sx, cc = measure_shift(recon2d, ref2d, mask); scale, rot = measure_scale_rot(ref2d, recon2d)
    fail = []
    if abs(sy) > tol_shift or abs(sx) > tol_shift: fail.append(f"shift ({sy:+.3f},{sx:+.3f})>{tol_shift}")
    if abs(scale-1) > tol_scale: fail.append(f"scale {scale:.4f}")
    if abs(rot) > tol_rot: fail.append(f"rot {rot:+.3f}deg")
    if fail: raise ReconAssertError(f"A1 {name}: " + "; ".join(fail))
    return dict(shift=(sy, sx), scale=scale, rot=rot, ncc=cc)

# ---------- A2 ----------
def point_source_roundtrip(render_fn, N, p=None, name="pt"):
    """render_fn: image->image (identity-expected round trip). asserts a delta at p returns at p."""
    if p is None: p = (N//2 + 7, N//2 - 5)
    delta = np.zeros((N, N)); delta[p] = 1.0
    out = np.abs(render_fn(delta)); pk = np.unravel_index(np.argmax(out), out.shape)
    off = (pk[0]-p[0], pk[1]-p[1])
    if abs(off[0]) > 0 or abs(off[1]) > 0: raise ReconAssertError(f"A2 {name}: delta at {p} -> peak {pk}, offset {off}")
    return dict(p_in=p, p_out=pk, offset=off)

# ---------- A3 ----------
def check_parseval(img, ksp, tol=1e-3, name="parseval"):
    ei = float(np.sum(np.abs(img)**2)); ek = float(np.sum(np.abs(ksp)**2))
    rel = abs(ei-ek)/(ei+1e-30)
    if rel > tol: raise ReconAssertError(f"A3 {name}: image energy {ei:.3e} vs k-space {ek:.3e} rel {rel:.2e}>{tol}")
    return dict(e_img=ei, e_ksp=ek, rel=rel)

# ---------- A4 ----------
def check_orientation(recon2d, ref2d, mask=None, name="orient"):
    if mask is None: mask = np.ones(ref2d.shape, bool)
    D4 = {"id": recon2d, "rot90": np.rot90(recon2d), "rot180": np.rot90(recon2d, 2), "rot270": np.rot90(recon2d, 3),
          "flipud": recon2d[::-1], "fliplr": recon2d[:, ::-1], "T": recon2d.T, "antiT": recon2d[::-1, ::-1].T}
    best = max(D4, key=lambda k: _ncc(D4[k], ref2d, mask) if D4[k].shape == ref2d.shape else -9)
    if best != "id": raise ReconAssertError(f"A4 {name}: best D4 transform is '{best}', not identity")
    return dict(best="id")

def check_recon(recon, ref, *, mask=None, ksp=None, render_fn=None, N=None, name="recon", geom=True):
    """run the applicable checks; RAISE on any failure. recon/ref are [H,W] or [H,W,T] (temporal-mean used)."""
    recon = check_array(recon, f"{name}.recon"); ref = check_array(ref, f"{name}.ref")
    r2 = recon.mean(-1) if recon.ndim == 3 else recon; f2 = ref.mean(-1) if ref.ndim == 3 else ref
    out = {}
    out["A5"] = "ok"
    if render_fn is not None and N is not None: out["A2"] = point_source_roundtrip(render_fn, N, name=name)
    if ksp is not None: out["A3"] = check_parseval(r2, ksp, name=name)
    out["A4"] = check_orientation(r2, f2, mask, name=name)
    if geom: out["A1"] = check_geometry(r2, f2, mask, name=name)
    return out
