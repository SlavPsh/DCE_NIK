"""Inspect the SENSE-A recon: render the trained image INR, and put it beside truth and the adjoint
gridding of the measured data (A^H y, which conv_check showed ~truth). Also a FORWARD-CONSISTENCY test:
forward the adjoint image through A and correlate with measured -- if A is a correct forward, corr>0."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, os, glob
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import xph_pipeline as P, xph_common as X
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
d = P.data(); tq = d["times"]; F = len(tq); RO = d["b1"].shape[0]; C = d["b1"].shape[-1]; tr = np.array(P.TRAIN_ANG)
body = d["labels"] > 0; Tr = X.truth_at(P.ZI, tq); Ttime = float(tq.max())
aor = Tr[X.rois(P.ZI, d["labels"])["aorta"]].mean(0); pk = int(np.argmax(aor))
b1 = torch.tensor(d["b1"], dtype=torch.complex64, device=dev).reshape(-1, C)
ax = (np.arange(RO)-RO/2.0); xn, yn = np.meshgrid(ax, ax, indexing="ij")
xn = torch.tensor(xn.reshape(-1), dtype=torch.float32, device=dev); yn = torch.tensor(yn.reshape(-1), dtype=torch.float32, device=dev)
ic = (np.stack(np.meshgrid((np.arange(RO)-RO//2)/(RO//2), (np.arange(RO)-RO//2)/(RO//2), indexing="ij"), -1)).reshape(-1, 2)
icoords = torch.tensor(ic, dtype=torch.float32, device=dev)

ck = torch.load(f"{P.OUT}/checkpoints/senseA_r16_s0/ck_final.pt", map_location=dev, weights_only=False)
model = P.make_model_g("wire_ff_subspace", ck["width"], P.FIX["k_sigma"], ck["seed"], 1, dev, rank=ck["rank"], warmstart=False)
model.load_state_dict(ck["state_dict"]); model.eval()
with torch.no_grad():
    amp = model.amplitudes(icoords, torch.zeros(icoords.shape[0], dtype=torch.long, device=dev))
    cf = amp[:, :, 0] + 1j*amp[:, :, 1]
    tn = torch.tensor([2*tq[t]/Ttime-1 for t in range(F)], dtype=torch.float32, device=dev)
    Phi = model.basis(tn)[:, :, 0].real.float()
    rho_pk = (cf @ Phi[pk].to(cf.dtype)).reshape(RO, RO).cpu().numpy()
    rho_mean = torch.stack([(cf @ Phi[t].to(cf.dtype)) for t in range(0, F, 8)], 0).mean(0).reshape(RO, RO).cpu().numpy()

# adjoint gridding A^H(measured), peak frame, SENSE-combined (reference "good" image)
def adj_frame(t):
    KX = torch.tensor(d["kx"][t, tr].reshape(-1), dtype=torch.float32, device=dev); KY = torch.tensor(d["ky"][t, tr].reshape(-1), dtype=torch.float32, device=dev)
    ph = torch.exp(1j*2*np.pi*(KX[:, None]*xn[None, :] + KY[:, None]*yn[None, :]))              # [M,P], A^H uses conj -> +i on measured
    yc = torch.tensor(d["kdata"][:, t, tr, :].reshape(C, -1), dtype=torch.complex64, device=dev)
    imgc = (ph.conj().transpose(0, 1) @ yc.transpose(0, 1)).reshape(RO, RO, C)
    b1i = b1.reshape(RO, RO, C); return (torch.sum(torch.conj(b1i)*imgc, -1)/(torch.sum(torch.abs(b1i)**2, -1)+1e-8)).cpu().numpy()
with torch.no_grad():
    adj_pk = adj_frame(pk)

def norm(x): x = np.abs(x); return x/(x.max()+1e-12)
def corr(a, b): return float(np.corrcoef(np.abs(a)[body], np.abs(b)[body])[0, 1])
print(f"peak frame t={tq[pk]:.0f}s", flush=True)
print(f"corr(senseA rho, truth)   = {corr(rho_pk, Tr[:,:,pk]):.3f}", flush=True)
print(f"corr(adjoint A^H y, truth)= {corr(adj_pk, Tr[:,:,pk]):.3f}   (conv_check ref ~0.7)", flush=True)
print(f"corr(senseA rho, adjoint) = {corr(rho_pk, adj_pk):.3f}", flush=True)

fig, ax = plt.subplots(1, 4, figsize=(15, 4))
for a, l, im in zip(ax, ["truth", "adjoint A^H y (~truth)", "senseA rho (peak)", "senseA rho (mean)"],
                    [Tr[:, :, pk], adj_pk, rho_pk, rho_mean]):
    a.imshow(norm(im), cmap="gray"); a.set_title(l, fontsize=11); a.axis("off")
fig.suptitle(f"SENSE-A inspection, peak t={tq[pk]:.0f}s", fontsize=12); plt.tight_layout()
fig.savefig(f"{P.OUT}/figures/senseA_inspect.png", dpi=120); print("SAVED figures/senseA_inspect.png", flush=True)
print("DONE_INSPECT", flush=True)
