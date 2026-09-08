"""Step 1 verify: regenerate NIK-sub12 via the model (GPU), apply OLD rot (im[::-1,::-1]) and the FIXED
centered rot (roll(flip,1)), and register each vs truth with the trusted NCC scan. FIXED must land at
(0,0), matching GRASP-K12 (contrast, no rot180). If FIXED lands elsewhere the fix is wrong/incomplete."""
import warnings; warnings.filterwarnings("ignore")
import glob, numpy as np, torch
from scipy.ndimage import uniform_filter, fourier_shift
import xph_pipeline as P, xph_common as X
A = f"{X.OUT}/arrays"; dev = torch.device("cuda")
d = P.data(); tq = d["times"]; body = d["labels"] > 0; Tr = X.truth_at(P.ZI, tq); rv = float(Tr[body].max()-Tr[body].min()); tm = Tr.mean(2)
def lss(v): return v * float((v[body]*Tr[body]).sum()/((v[body]**2).sum()+1e-12))
def fshift(im, sy, sx): return np.real(np.fft.ifft2(fourier_shift(np.fft.fft2(im), (sy, sx))))
def ncc(a, b): a = a[body]-a[body].mean(); b = b[body]-b[body].mean(); return float((a*b).sum()/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
def ssim(a, b, win=7):
    C1 = (0.01*rv)**2; C2 = (0.03*rv)**2; ma = uniform_filter(a, win); mb = uniform_filter(b, win)
    va = uniform_filter(a*a, win)-ma**2; vb = uniform_filter(b*b, win)-mb**2; vab = uniform_filter(a*b, win)-ma*mb
    return float((((2*ma*mb+C1)*(2*vab+C2))/((ma**2+mb**2+C1)*(va+vb+C2)))[body].mean())
def offset(mm):
    best = (-9, 0, 0)
    for sy in np.arange(-2.0, 2.01, 0.05):
        for sx in np.arange(-2.0, 2.01, 0.05):
            c = ncc(fshift(mm, sy, sx), tm)
            if c > best[0]: best = (c, sy, sx)
    _, sy0, sx0 = best
    for sy in np.arange(sy0-0.06, sy0+0.061, 0.01):
        for sx in np.arange(sx0-0.06, sx0+0.061, 0.01):
            c = ncc(fshift(mm, sy, sx), tm)
            if c > best[0]: best = (c, sy, sx)
    return best[1], best[2]

# regenerate sub12 dyn from the frozen best checkpoint
_, _, _, _, nz, dims = P.build_train(dev); Cc = dims[3]
model = P.make_model_g("wire_ff_subspace", 768, P.FIX["k_sigma"], 0, Cc, dev, rank=12, warmstart=False); model.eval()
bstep = int(np.load(f"{A}/img_eval_sub12_w768_s1.npz")["best_step"])
ck = [p for p in glob.glob(f"{X.OUT}/checkpoints/sub12_w768_s1/ck_*.pt") if int(p.split('ck_')[-1].split('.')[0]) == bstep][0]
model.load_state_dict(torch.load(ck, map_location=dev, weights_only=False)["state_dict"])
dyn = P.reconstruct_g(model, nz, tq, dev)
old = lss(np.abs(np.stack([dyn[::-1, ::-1, t] for t in range(dyn.shape[2])], -1)))
new = lss(np.abs(np.stack([np.roll(dyn[::-1, ::-1, t], (1, 1), axis=(0, 1)) for t in range(dyn.shape[2])], -1)))
grasp = lss(np.abs(np.load(f"{A}/grasp_recon.npz")["rec"]).astype(np.float64))
for tag, vol in [("OLD rot im[::-1,::-1]", old), ("FIXED roll(flip,1)", new), ("GRASP-K12 (contrast)", grasp)]:
    oy, ox = offset(vol.mean(2)); ss = float(np.mean([ssim(vol[:, :, t], Tr[:, :, t]) for t in range(vol.shape[2])]))
    print(f"  {tag:28s} residual offset (sy,sx)=({oy:+.3f},{ox:+.3f}) | SSIM {ss:.4f}", flush=True)
print("STEP1_VERIFY_DONE")
