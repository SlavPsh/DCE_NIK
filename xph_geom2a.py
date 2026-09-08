"""STEP 2a: rule out geometry BEFORE anything else. A sub-pixel shift / scale / rotation / linear k-space
phase ramp preserves the magnitude spectrum exactly but destroys SSIM. NIK's recon is rot180-applied on
an even grid (RO=220): rot180 centre (N-1)/2=109.5 vs FFT centre N/2=110 -> a likely ~1px shift GRASP (id)
does not have. Measure sub-pixel shift + scale + rotation of NIK-sub12 (and GRASP-K12 as contrast) vs
truth, confirm via the k-space phase ramp, then report SSIM AFTER correction. If SSIM jumps, STOP: the
gap is a registration/phase artifact and 2b-2d + STEP 3 are moot."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from scipy.ndimage import uniform_filter, shift as ndshift, rotate as ndrotate, zoom as ndzoom, map_coordinates
import xph_pipeline as P, xph_common as X
A = f"{X.OUT}/arrays"
d = P.data(); tq = d["times"]; body = d["labels"] > 0; Tr = X.truth_at(P.ZI, tq); RO = Tr.shape[0]
rv = float(Tr[body].max() - Tr[body].min())
def lsscale(v): return v * float((v[body]*Tr[body]).sum()/((v[body]**2).sum()+1e-12))
def ssim(a, b, win=7):
    C1 = (0.01*rv)**2; C2 = (0.03*rv)**2; ma = uniform_filter(a, win); mb = uniform_filter(b, win)
    va = uniform_filter(a*a, win)-ma**2; vb = uniform_filter(b*b, win)-mb**2; vab = uniform_filter(a*b, win)-ma*mb
    return float((((2*ma*mb+C1)*(2*vab+C2))/((ma**2+mb**2+C1)*(va+vb+C2)))[body].mean())
def ssim_vol(vol): return float(np.mean([ssim(vol[:, :, t], Tr[:, :, t]) for t in range(vol.shape[2])]))

# ---- upsampled DFT registration (Guizar-Sicairos) ----
def _updft(data, nor, noc, usf, roff, coff):
    kc = np.exp(-2j*np.pi/(noc*usf) * (np.fft.ifftshift(np.arange(data.shape[1]))[:, None]-data.shape[1]//2) * (np.arange(noc)-coff)[None, :])
    kr = np.exp(-2j*np.pi/(nor*usf) * (np.arange(nor)-roff)[:, None] * (np.fft.ifftshift(np.arange(data.shape[0]))[None, :]-data.shape[0]//2))
    return kr @ data @ kc
def register(ref, mov, usf=100):
    F = np.fft.fft2(ref); G = np.fft.fft2(mov); R = F*np.conj(G)
    cc = np.fft.ifft2(R); ax = np.unravel_index(np.argmax(np.abs(cc)), cc.shape)
    sh = np.array(ax, float); sh[sh > np.array(cc.shape)/2] -= np.array(cc.shape)[sh > np.array(cc.shape)/2]
    # refine
    ns = int(np.ceil(usf*1.5)); ro = ns//2 - sh[0]*usf; co = ns//2 - sh[1]*usf
    cc2 = np.conj(_updft(R, ns, ns, usf, ro, co))
    a2 = np.unravel_index(np.argmax(np.abs(cc2)), cc2.shape)
    sh = sh + (np.array(a2, float) - ns//2)/usf
    return sh  # (dy,dx): shifting mov by +sh aligns to ref

def logpolar_scale_rot(ref, mov):
    def spec(im):
        w = np.outer(np.hanning(im.shape[0]), np.hanning(im.shape[1])); F = np.abs(np.fft.fftshift(np.fft.fft2(im*w))); return np.log1p(F)
    S1, S2 = spec(ref), spec(mov); c = np.array(S1.shape)//2; rmax = min(c); nr, nt = 256, 360
    lr = np.linspace(0, np.log(rmax), nr); th = np.linspace(0, np.pi, nt)          # 0..pi (spectrum is symmetric)
    RR, TT = np.meshgrid(np.exp(lr), th, indexing="ij")
    ys = c[0]+RR*np.sin(TT); xs = c[1]+RR*np.cos(TT)
    L1 = map_coordinates(S1, [ys.ravel(), xs.ravel()], order=1).reshape(nr, nt)
    L2 = map_coordinates(S2, [ys.ravel(), xs.ravel()], order=1).reshape(nr, nt)
    F1 = np.fft.fft2(L1); F2 = np.fft.fft2(L2); Rr = F1*np.conj(F2); Rr /= np.abs(Rr)+1e-12
    cc = np.abs(np.fft.ifft2(Rr)); pk = np.unravel_index(np.argmax(cc), cc.shape)
    dscale = pk[0] if pk[0] < nr/2 else pk[0]-nr; dtheta = pk[1] if pk[1] < nt/2 else pk[1]-nt
    scale = np.exp(-dscale*np.log(rmax)/nr); rot_deg = -dtheta*180.0/nt
    return scale, rot_deg

for tag, fn, key, rotflag in [("NIK-sub12", f"{A}/img_eval_sub12_w768_s1.npz", "rec_best", True), ("GRASP-K12", f"{A}/grasp_recon.npz", "rec", False)]:
    vol = np.abs(np.load(fn)[key]).astype(np.float64); vol = lsscale(vol)
    mref = Tr.mean(2); mmov = vol.mean(2)
    sh = register(mref, mmov, usf=100); scale, rot = logpolar_scale_rot(mref, mmov)
    base = ssim_vol(vol)
    # linear k-space phase ramp check: cross-power phase slope should equal the shift
    Fr = np.fft.fft2(mref); Gm = np.fft.fft2(mmov); Rk = Fr*np.conj(Gm); ph = np.angle(np.fft.fftshift(Rk))
    kyv = (np.arange(RO)-RO//2); mag = np.abs(np.fft.fftshift(Rk)); wsel = mag > np.percentile(mag, 90)
    print(f"\n=== {tag}: rot180-applied={rotflag} ===", flush=True)
    print(f"  sub-pixel shift (dy,dx) = ({sh[0]:+.3f}, {sh[1]:+.3f}) px | scale {scale:.4f} | rotation {rot:+.3f} deg", flush=True)
    print(f"  baseline SSIM {base:.4f}", flush=True)
    # correct shift only
    volS = np.stack([ndshift(vol[:, :, t], (sh[0], sh[1]), order=3, mode="constant") for t in range(vol.shape[2])], -1); volS = lsscale(volS)
    print(f"  SSIM after SHIFT correction {ssim_vol(volS):.4f}  (delta {ssim_vol(volS)-base:+.4f})", flush=True)
    # correct shift+rot+scale if non-trivial
    if abs(rot) > 0.2 or abs(scale-1) > 0.003:
        def full_corr(im):
            im2 = ndzoom(im, scale, order=3); im2 = ndrotate(im2, rot, order=3, reshape=False)
            im2 = im2[:RO, :RO] if im2.shape[0] >= RO else np.pad(im2, ((0, RO-im2.shape[0]), (0, RO-im2.shape[1])))
            return ndshift(im2, (sh[0], sh[1]), order=3, mode="constant")
        volF = lsscale(np.stack([full_corr(vol[:, :, t]) for t in range(vol.shape[2])], -1))
        print(f"  SSIM after SHIFT+ROT+SCALE {ssim_vol(volF):.4f}  (delta {ssim_vol(volF)-base:+.4f})", flush=True)
print("\nGEOM2A_DONE")
