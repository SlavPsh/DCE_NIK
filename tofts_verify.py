"""step 3 verification for nik_tofts_subspace. all checks print PASS/FAIL with numbers."""
import warnings; warnings.filterwarnings("ignore")
import sys, json, glob, time, numpy as np, torch
sys.path.insert(0, "/net/beegfs/users/P101440/DCE_NIK")
import dcenet_adapter as D
from scipy.integrate import solve_ivp
from nik_model import WIRE_FF_TOFTS_KXY_COIL_T_REIM, patlak_to_tofts
import nik_tofts_basis as NB
res = {}
def rep(name, val, thr, ok=None):
    ok = (val <= thr) if ok is None else ok; res[name] = dict(value=float(val), thr=thr, pass_=bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {val:.3e} (thr {thr:g})", flush=True)

# ---------- 3a forward (DCE-NET sampled convolution) vs independent numerical integration ----------
z = np.load("aif_xph.npz"); tC = np.asarray(z["tC"], float); aif = np.asarray(z["aif_frame"], float); t_min = tC / 60
cp = lambda tt: np.interp(tt, t_min, aif, left=0.0)
worst = 0.0
for ke, ve, vp, dt in [(0.7, 0.9, 0.1, 8/60), (0.7, 0.8, 0.2, 8/60), (1.5, 0.3, 0.02, 0.0), (0.1, 0.5, 0.3, 15/60), (3.0, 0.05, 0.6, 20/60)]:
    Kt = ve * ke
    # ODE: dCe/dt = Ktrans cp(t-dt) - kep Ce ; C = vp cp(t-dt) + Ce   (DCE-NET delays the whole AIF by dt)
    sol = solve_ivp(lambda tt, y: Kt * cp(tt - dt) - ke * y, (t_min[0], t_min[-1]), [0.0], t_eval=t_min, rtol=1e-9, atol=1e-12, max_step=1e-3)
    ref = vp * cp(t_min - dt) + sol.y[0]
    out = D.ext_tofts_sampled(t_min, aif, np.array([ke]), np.array([dt]), np.array([ve]), np.array([vp])).numpy()[0]
    worst = max(worst, np.linalg.norm(out - ref) / np.linalg.norm(ref))
rep("3a forward_vs_ODE relL2 (worst of 5, 0.5 s grid trapezoid)", worst, 2e-2)
# same on a 10x finer grid: convolution error must shrink (discretization, not a bug)
tf = np.linspace(t_min[0], t_min[-1], 10 * len(t_min)); aiff = np.interp(tf, t_min, aif)
ke, ve, vp, dt = 1.5, 0.3, 0.02, 0.0; Kt = ve * ke
sol = solve_ivp(lambda tt, y: Kt * cp(tt - dt) - ke * y, (tf[0], tf[-1]), [0.0], t_eval=tf, rtol=1e-9, atol=1e-12, max_step=1e-3)
out = D.ext_tofts_sampled(tf, aiff, np.array([ke]), np.array([dt]), np.array([ve]), np.array([vp])).numpy()[0]
rep("3a forward_vs_ODE relL2 fine grid (0.05 s)", np.linalg.norm(out - (vp * cp(tf) + sol.y[0])) / np.linalg.norm(vp * cp(tf) + sol.y[0]), 2e-3)

# ---------- 3b kep->0 kernel limit == Patlak ----------
P = NB.patlak_span(tC, aif)                                         # [G,3] = [aif/max, iaif/max, 1]
ia_raw = np.concatenate([[0.0], np.cumsum(0.5 * (aif[1:] + aif[:-1]) * np.diff(t_min))])   # int cp dt (min)
for ke in [1e-1, 1e-2, 1e-3]:
    ve, vp = 0.5, 0.1; Kt = ve * ke
    out = D.ext_tofts_sampled(t_min, aif, np.array([ke]), np.array([0.0]), np.array([ve]), np.array([vp])).numpy()[0]
    pat = vp * aif + Kt * ia_raw
    e = np.linalg.norm(out - pat) / np.linalg.norm(pat)
    proj = out - P @ np.linalg.lstsq(P, out, rcond=None)[0]; r = np.linalg.norm(proj) / np.linalg.norm(out)
    print(f"   kep={ke:g}: |tofts - (vp cp + Ktrans int cp)|/|.| = {e:.3e}; residual outside Patlak span {r:.3e}")
rep("3b kep->0: residual outside Patlak span at kep=1e-3", r, 1e-2)
rep("3b kep->0: |tofts-patlak| at kep=1e-3", e, 1e-2)

# ---------- 3c dictionary approximation (fresh seed, saved basis) ----------
bz = np.load("results/tofts_vs_patlak/basis_xph.npz", allow_pickle=True); B = bz["atoms"].astype(np.float64)
G = B.T @ B; rep("3c atoms orthonormality |B^T B - I|max", np.abs(G - np.eye(B.shape[1])).max(), 1e-4)
Q = bz["Q_patlak"].astype(np.float64); R = bz["R_patlak"].astype(np.float64)
rep("3c Patlak span preserved |Q R - P|max", np.abs(Q @ R - P).max(), 1e-4)
ke2, ve2, vp2, dt2 = NB.sample(3000, 777); Dt = NB.dictionary(tC, aif, ke2, ve2, vp2, dt2)
fp, wo = NB.windows(tC, aif)
rep(f"3c fresh-dict projection err total (rank {B.shape[1]})", NB.proj_err(B, Dt), 0.01)
rep("3c fresh-dict projection err first-pass", NB.proj_err(B, Dt, fp), 0.01)
rep("3c fresh-dict projection err washout", NB.proj_err(B, Dt, wo), 0.01)
# in-vivo rank (12): same check on sl21 basis
bz21 = np.load("results/tofts_vs_patlak/basis_sl21.npz", allow_pickle=True); z21 = np.load("aif_slice21.npz"); t21 = np.asarray(z21["tC"], float); a21 = np.asarray(z21["aif_frame"], float)
ke2, ve2, vp2, dt2 = NB.sample(3000, 778); Dt21 = NB.dictionary(t21, a21, ke2, ve2, vp2, dt2); B21 = bz21["atoms"].astype(np.float64)
rep(f"3c sl21 fresh-dict projection err total (rank {B21.shape[1]})", NB.proj_err(B21, Dt21), 0.01)
# phantom generator curves (pkLUT cortex/medulla/aorta) must be in the span
gen = NB.dictionary(tC, aif, np.array([0.7, 0.7, 1e-3]), np.array([0.9, 0.8, 1e-6]), np.array([0.1, 0.2, 0.6]), np.array([8/60, 8/60, 15/60]))
rep("3c phantom pkLUT curves (cortex/medulla/aorta) projection err", NB.proj_err(B, gen), 0.01)
rep("3c same curves onto Patlak span (rank 3) [expected FAIL = motivation]", NB.proj_err(Q, gen), 0.01)

# ---------- 3d Patlak checkpoint -> expanded basis, output agreement ----------
import xph_pipeline as XP
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ck = sorted(glob.glob(f"{XP.OUT}/checkpoints/w768_ks2.5_s0/ck_*.pt"))[0]; sd = torch.load(ck, map_location="cpu", weights_only=False)
C = int(sd["ncc"]); pat = XP.make_model(768, 2.5, 0, C, dev); pat.load_state_dict(sd["state_dict"]); pat.eval()
tof = XP.make_model_tofts(768, 2.5, 0, C, dev); tof = patlak_to_tofts(sd["state_dict"], tof); tof.eval()
g = torch.Generator().manual_seed(0); N = 20000
kc = (torch.rand(N, 2, generator=g) * 2 - 1).to(dev); tt = (torch.rand(N, generator=g) * 2 - 1).to(dev); ci = torch.randint(0, C, (N,), generator=g).to(dev)
with torch.no_grad(): yp = pat(kc, tt, ci); yt = tof(kc, tt, ci)
rep(f"3d converted model output agreement relL2 ({ck.split('/')[-1]}, N={N} random k,t,coil)", float(torch.norm(yp - yt) / torch.norm(yp)), 1e-4)
with torch.no_grad():
    Pp = pat.basis(tt)[:, :, 0].double(); Pt = tof.basis(tt)[:, :, 0].double()
rep("3d basis relation |Phi_patlak - Phi_tofts[:, :3] R|max", float((Pp - Pt[:, :3] @ tof.R_patlak.double()).abs().max()), 1e-4)
rep("3d extra atoms untouched (|a_head rows 3:|max == 0)", float(tof.a_head.weight[6:].abs().max()), 0.0, ok=float(tof.a_head.weight[6:].abs().max()) == 0.0)
rep("3d rank / params", tof.rank, 0, ok=tof.rank == B.shape[1]); print("   params patlak", XP.param_counts(pat), "tofts", XP.param_counts(tof))

# ---------- 3e short smoke (train 50 steps from the converted init, loss finite and non-increasing) ----------
from nik_focal_loss import composable_kspace_loss
X_, Yn, T_, C_, nz, dims = XP.build_train(dev); tof.train(); opt = torch.optim.Adam(tof.parameters(), lr=XP.LR, weight_decay=XP.WD)
losses = []; t0 = time.time(); Nn = X_.shape[0]
for s in range(60):
    idx = torch.randint(0, Nn, (XP.BATCH,), device=dev); opt.zero_grad(set_to_none=True)
    loss = composable_kspace_loss(tof(X_[idx], T_[idx], C_[idx]), Yn[idx], dcf=torch.ones(XP.BATCH, device=dev), use_dcf=False, dcf_power=0.0, use_focal=False, focal_warmup_progress=1.0)
    loss.backward(); torch.nn.utils.clip_grad_norm_(tof.parameters(), 1.0); opt.step(); losses.append(float(loss))
print(f"   smoke: loss first10 {np.mean(losses[:10]):.4e} last10 {np.mean(losses[-10:]):.4e} | {(time.time()-t0)/60:.2f} s/step | grad on extra head rows {float(tof.a_head.weight.grad[6:].abs().max()):.2e}")
rep("3e smoke finite + non-increasing (last10/first10)", np.mean(losses[-10:]) / np.mean(losses[:10]), 1.0, ok=np.isfinite(losses).all() and np.mean(losses[-10:]) <= np.mean(losses[:10]))
tof.eval(); v = XP.kspace_nmse(tof, nz, XP.masks()["val"], dev); print("   VAL nmse after 60 steps:", v)
json.dump(res, open("results/tofts_vs_patlak/verify.json", "w"), indent=1)
print("ALL", "PASS" if all(r["pass_"] for k, r in res.items() if "expected FAIL" not in k) else "FAIL"); print("VERIFY_DONE")
