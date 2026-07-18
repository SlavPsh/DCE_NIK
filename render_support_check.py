"""Render the SAME NIK model at support_radius=0.5 (current) vs 1.0 (proposed fix) to prove
the render mask is the blur. out: support_check.png + HF-ratio for each."""
import sys, numpy as np, torch
sys.path.insert(0, "/scratch/rnga/vvpshenov/grasp_pro_py")
import nik_adapter as NA
from nik_output_recon import recon_nik_cart
from kspace_normalization import compute_dcf_radial, KSpaceNormalizer
import nik_model as NM
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = "/scratch/rnga/vvpshenov/grasp_pro_py/results_ref"; dev = "cuda" if torch.cuda.is_available() else "cpu"
sh = NA.load_shared(OUT); ds = NA.make_radial_dataset(OUT, 13, compute_device=dev, shared=sh)
x, t, c, y_raw, sid = ds["x_all"], ds["t_all"], ds["coil_all"], ds["y_all_raw"], ds["spoke_id_all"]
b1 = ds["b1"]; ncc = ds["meta"]["ncc"]; bas = ds["meta"]["bas"]
# same 70/30 split + normalizer (env 0.75) the model trained with
uniq = torch.unique(sid); n_tr = max(1, int(uniq.numel() * 0.7))
g = torch.Generator().manual_seed(0); perm = uniq[torch.randperm(uniq.numel(), generator=g)]
tr = torch.isin(sid, perm[:n_tr])
dcf = compute_dcf_radial(x, method="simple_ramp")
nz = KSpaceNormalizer(); nz.fit(x[tr], y_raw[tr], dcf=dcf[tr], envelope_exponent=0.75)

ck = torch.load("/tmp/ckpt_check.pt", map_location=dev)
model = NM.WIRE_FF_RES_RADIAL_KXY_COIL_T_REIM(n_coils=ncc, hidden=512, depth=12, w0=62.0, s0=15.0,
        k_freq=256, k_sigma=2.5, t_freq=32, t_sigma=1.5, radial_alpha=0.5)
model.load_state_dict(ck["best_state"]); model.to(dev); model.eval()
print(f"loaded radial a0.5 ckpt (step {ck['step']}, held {ck['best_heldout']:.4f})", flush=True)

sh1 = dict(sh); sh1["frame_time"] = np.array([0.5], np.float32); sh1["nt"] = 1   # one mid-scan frame
def hf(img):
    im = img / (img.mean() + 1e-12); F = np.abs(np.fft.fftshift(np.fft.fft2(im))) ** 2
    ny, nx = im.shape; y, xx = np.indices((ny, nx)); r = np.hypot(y - ny // 2, xx - nx // 2).astype(int)
    ps = np.bincount(r.ravel(), F.ravel()) / np.maximum(np.bincount(r.ravel()), 1); return ps[len(ps) // 2:].sum() / ps.sum()

imgs = {}
for sr in (0.5, 1.0):
    cart = NA.reconstruct_cartesian(model, nz, None, device=dev, shared=sh1, support_radius=sr, verbose=False)
    im = np.abs(recon_nik_cart(cart, b1, bas))[..., 0]
    imgs[sr] = im; print(f"support_radius={sr}: HF-ratio {hf(im):.4g}", flush=True)

cs = np.abs(np.load("/scratch/rnga/vvpshenov/presentation/assets/arm1_cs100_sl13.npy")).mean(-1)
print(f"CS-100 (reference):  HF-ratio {hf(cs):.4g}")
fig, ax = plt.subplots(1, 3, figsize=(12, 4.2))
for a, (lab, im) in zip(ax, [("NIK support=0.5 (current)", imgs[0.5]), ("NIK support=1.0 (fix)", imgs[1.0]), ("CS-100", cs)]):
    a.imshow(np.rot90(im), cmap="gray", vmin=0, vmax=np.percentile(im, 99.5)); a.set_title(lab, fontsize=11); a.axis("off")
fig.suptitle("Render-mask check: support_radius 0.5 vs 1.0 on the SAME NIK model", fontweight="bold")
fig.tight_layout(); fig.savefig("/scratch/rnga/vvpshenov/DCE_NIK/support_check.png", bbox_inches="tight", dpi=140)
print("wrote support_check.png")
