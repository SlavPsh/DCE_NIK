"""Render the best models for the notebook comparison figures: truth vs GRASP-K12 vs best NIK
(sub16, free), plus the single-FOV(1x)-vs-oversampled(2x) before/after on sub16. Saves a compact
npz (3 key frames + full ROI curves + masks). No training; truth eval-only."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, os, glob
import xph_pipeline as P, xph_common as X, recon_asserts as RA
import os as _os
# phantom reference-method plumbing. defaults = grasp-pro (unchanged). grasp v2:
#   GRASP_NPZ=grasp_v2_recon.npz GRASP_LABEL=GRASP-v2 TAG=_gv2
_GNPZ = _os.environ.get("GRASP_NPZ", "grasp_ksweep_K12.npz")
_GLAB = _os.environ.get("GRASP_LABEL", "GRASP-K12 (ref)")
_TAG = _os.environ.get("TAG", "")

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_, _, _, _, nz, dims = P.build_train(dev); C = dims[3]
d = P.data(); tq = d["times"]; body = d["labels"] > 0; Tr = X.truth_at(P.ZI, tq); Rz = X.rois(P.ZI, d["labels"])
RO = d["b1"].shape[0]; F = len(tq); rot = lambda im: np.roll(im[::-1, ::-1], (1, 1), axis=(0, 1))
aor = Tr[Rz["aorta"]].mean(0); pk = int(np.argmax(aor))                            # peak-enhancement frame
early = int(np.argmin(np.abs(tq-5))); late = int(np.argmin(np.abs(tq-120)))
FRM = [early, pk, late]; print(f"frames: early {tq[early]:.0f}s  peak {tq[pk]:.0f}s  late {tq[late]:.0f}s", flush=True)

def build(kind, s, st, rank=16):
    tag = (f"sub{rank}_w768_s{s}" if kind == "sub" else f"free_w768_s{s}"); p = f"{P.OUT}/checkpoints/{tag}/ck_{st}.pt"
    if not os.path.exists(p): p = sorted(glob.glob(f"{P.OUT}/checkpoints/{tag}/ck_*.pt"))[-1]
    m = P.make_model_g("wire_ff_subspace" if kind == "sub" else "wire_ff", 768, P.FIX["k_sigma"], 0, C, dev, rank=rank, warmstart=False)
    m.load_state_dict(torch.load(p, map_location=dev, weights_only=False)["state_dict"]); m.eval(); return m

@torch.no_grad()
def render(m, ov):
    P.OVERSAMPLE = ov                                                             # toggle render oversampling
    dyn = P.reconstruct_g(m, nz, tq, dev); rec = np.abs(np.stack([rot(dyn[:, :, t]) for t in range(F)], -1))
    s = np.sum(rec[body]*Tr[body])/(np.sum(rec[body]**2)+1e-12); rec = rec*s
    RA.check_recon(rec, Tr, mask=body, name=f"ov{ov}")
    return rec

msub = build("sub", 0, 24000); mfree = build("free", 0, 40000)
sub2 = render(msub, 2); sub1 = render(msub, 1); free2 = render(mfree, 2)
gk = np.abs(np.load(f"{P.OUT}/arrays/{_GNPZ}")["rec"]).astype(np.float32)
sg = np.sum(gk[body]*Tr[body])/(np.sum(gk[body]**2)+1e-12); gk = gk*sg
def curves(rec): return {nm: rec[Rz[nm]].mean(0) for nm in ("aorta", "cortex", "medulla")}
np.savez(f"{P.OUT}/arrays/compare_recons{_TAG}.npz",
         frames=np.array(FRM), ftimes=tq[FRM], times=tq, body=body,
         truth=Tr[:, :, FRM], grasp=gk[:, :, FRM], sub16_2x=sub2[:, :, FRM], sub16_1x=sub1[:, :, FRM], free_2x=free2[:, :, FRM],
         **{f"cur_truth_{k}": v for k, v in curves(Tr).items()},
         **{f"cur_grasp_{k}": v for k, v in curves(gk).items()},
         **{f"cur_sub16_{k}": v for k, v in curves(sub2).items()},
         **{f"cur_free_{k}": v for k, v in curves(free2).items()})
print("SAVED compare_recons.npz  DONE_COMPARE", flush=True)
