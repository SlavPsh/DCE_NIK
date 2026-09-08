import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, xph_pipeline as P, xph_common as X, recon_asserts as RA
dev = torch.device("cuda")
d = P.data(); tq = d["times"]; F = len(tq); RO = d["b1"].shape[0]; C = d["b1"].shape[-1]; tr = np.array(P.TRAIN_ANG)
body = d["labels"] > 0; Tr = X.truth_at(P.ZI, tq); trm = Tr.mean(-1)
b1 = torch.tensor(d["b1"], dtype=torch.complex64, device=dev); den = (torch.sum(torch.abs(b1)**2, -1)+1e-8)
ax = (np.arange(RO)-RO/2.0); xn, yn = np.meshgrid(ax, ax, indexing="ij")
xn = torch.tensor(xn.reshape(-1), dtype=torch.float32, device=dev); yn = torch.tensor(yn.reshape(-1), dtype=torch.float32, device=dev)
FR = list(range(0, F, 25))
maps = {"base": (1, 1, 0), "negx": (-1, 1, 0), "negy": (1, -1, 0), "negxy": (-1, -1, 0),
        "swap": (1, 1, 1), "swap_negx": (-1, 1, 1), "swap_negy": (1, -1, 1), "swap_negxy": (-1, -1, 1)}
for mp, (sx, sy, sw) in maps.items():
    acc = torch.zeros((RO, RO), dtype=torch.complex64, device=dev)
    for t in FR:
        KX = torch.tensor(d["kx"][t, tr].reshape(-1), dtype=torch.float32, device=dev)
        KY = torch.tensor(d["ky"][t, tr].reshape(-1), dtype=torch.float32, device=dev)
        a, b = (KY, KX) if sw else (KX, KY); a = sx*a; b = sy*b
        ph = torch.exp(1j*2*np.pi*(a[:, None]*xn[None, :] + b[:, None]*yn[None, :]))            # [M,P]
        yc = torch.tensor(d["kdata"][:, t, tr, :].reshape(C, -1), dtype=torch.complex64, device=dev)  # [C,M]
        imgc = (ph.conj().transpose(0, 1) @ yc.transpose(0, 1)).reshape(RO, RO, C)               # adjoint [P,C]
        acc = acc + torch.sum(torch.conj(b1)*imgc, -1)/den
    g = torch.abs(acc).cpu().numpy()
    sh = RA.measure_shift(g/g.max(), trm/trm.max(), mask=body)
    try: RA.check_orientation(g, trm, body, name=mp); orient = "id"
    except Exception as e: orient = str(e).split("'")[1] if "'" in str(e) else "nonid"
    cc = np.corrcoef(g[body], trm[body])[0, 1]
    print(f"{mp:12s} shift=({float(sh[0]):+.2f},{float(sh[1]):+.2f}) corr={cc:+.3f} D4best={orient}", flush=True)
print("DONE_CONV")
