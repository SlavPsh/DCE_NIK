"""K1a: where does the low-|k| error enter? Evaluate the trained model on its OWN measured TRAIN
points, split by |k| band, in BOTH spaces:
  - NORMALIZED space (what the loss actually minimizes): MSE(model - Yn)   [the training objective]
  - DENORMALIZED space (physical k-space): relative error NMSE(denorm(model) - Y_raw)
If normalized low-|k| loss ~ high-|k| loss (balanced) but denormalized low-|k| error dominates,
the error is injected by denormalization (envelope re-multiply), not by under-fitting.
No truth used; measured train data only. sub16 both seeds (mean +/- spread)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, glob, os
import xph_pipeline as P
from kspace_normalization import KSpaceNormalizer, compute_dcf_radial
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# rebuild the exact train tensors + normalizer (same as P.build_train, but keep r01 = |k|/0.5)
X_, Y_, T_, C_, R, dims = P._dataset(P.masks()["train"], dev); Ccoils = dims[3]
dcf = compute_dcf_radial(X_, method="simple_ramp")
nz = KSpaceNormalizer(); nz.fit(X_, Y_, dcf=dcf, envelope_exponent=P.FIX["env"]); Yn = nz.normalize(X_, Y_)
R = np.asarray(R)                                                                 # |k|/0.5 in [0,1] per point
bands = {"low(<0.10)": R < 0.10, "mid(0.3-0.7)": (R >= 0.3) & (R < 0.7), "high(>0.7)": R >= 0.7}

@torch.no_grad()
def probe(tag, step):
    p = f"{P.OUT}/checkpoints/{tag}/ck_{step}.pt"
    if not os.path.exists(p): p = sorted(glob.glob(f"{P.OUT}/checkpoints/{tag}/ck_*.pt"))[-1]
    rk = int(tag.split("sub")[1].split("_")[0]); m = P.make_model_g("wire_ff_subspace", 768, P.FIX["k_sigma"], 0, Ccoils, dev, rank=rk, warmstart=False)
    m.load_state_dict(torch.load(p, map_location=dev, weights_only=False)["state_dict"]); m.eval()
    outn = m(X_, T_, C_)                                                          # NORMALIZED prediction [N,2]
    prd = nz.denormalize(X_, outn)                                                # DENORMALIZED [N,2]
    en = ((outn - Yn)**2).sum(1).cpu().numpy()                                    # normalized squared error per point
    ed = ((prd - Y_)**2).sum(1).cpu().numpy(); pw = (Y_**2).sum(1).cpu().numpy()  # denorm sq err, signal power
    yn2 = (Yn**2).sum(1).cpu().numpy()
    res = {}
    for nm, mk in bands.items():
        res[nm] = (float(en[mk].mean()),                                          # normalized MSE (loss units)
                   float(en[mk].mean()/ (yn2[mk].mean()+1e-30)),                  # normalized NMSE (relative)
                   float(ed[mk].sum()/(pw[mk].sum()+1e-30)))                      # denormalized NMSE (relative)
    return res

CFG = [("sub16_w768_s0", 24000), ("sub16_w768_s1", 40000)]
allr = [probe(t, s) for t, s in CFG]
print("K1a  sub16 (mean over 2 seeds; +/- half-spread)")
print(f"{'band':14s} {'norm-MSE(loss)':>16s} {'norm-NMSE':>12s} {'denorm-NMSE':>12s}")
for nm in bands:
    a = np.array([r[nm] for r in allr])                                          # [seeds,3]
    mu = a.mean(0); sp = (a.max(0)-a.min(0))/2
    print(f"{nm:14s} {mu[0]:10.3e}+-{sp[0]:.0e} {mu[1]:9.3e}+-{sp[1]:.0e} {mu[2]:9.3e}+-{sp[2]:.0e}")
# verdict helper: ratio of low to high normalized loss, and low denorm share
lo = np.array([r["low(<0.10)"] for r in allr]).mean(0); hi = np.array([r["high(>0.7)"] for r in allr]).mean(0)
print(f"\nnormalized loss  low/high ratio = {lo[0]/hi[0]:.2f}   (>>1 => under-fit at low-|k|; ~1 => balanced fit)")
print(f"denormalized NMSE low/high ratio = {lo[2]/hi[2]:.2f}   (>>1 => error concentrated at low-|k| in physical space)")
print("DONE_K1A")
