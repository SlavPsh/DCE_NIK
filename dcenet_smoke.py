"""forward/backward smoke check of the DCE-NET torch extended-Tofts routines through the adapter.
plausible params from upstream hyperparams (kep 0.1-2, ve 0.01-0.7, vp 0.001-0.05, dt ~0.1-0.6 min).
deliberately probes the unstable branches. NaN gradients are REPORTED, never hidden."""
import warnings; warnings.filterwarnings("ignore")
import sys, numpy as np, torch
sys.path.insert(0, "/scratch/rnga/vvpshenov/DCE_NIK")
import dcenet_adapter as D
import aif_gt                                                # our numpy Cosine4 (same functional form)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("upstream SHA", D.upstream_sha(), "| device", dev)
T = 120; t = np.linspace(0.0, 6.0, T)                        # minutes
aif4 = D.phantom_aif4()

def run(label, fn, ke, dt, ve, vp, **kw):
    p = {k: torch.tensor(v, dtype=torch.float32, device=dev, requires_grad=True) for k, v in
         dict(ke=ke, dt=dt, ve=ve, vp=vp).items()}
    ct = fn(t, kw["aif"], p["ke"], p["dt"], p["ve"], p["vp"], device=dev)
    ok = torch.isfinite(ct).all().item()
    ct.sum().backward()
    print(f"\n[{label}] ct {tuple(ct.shape)} finite={ok} range [{ct.min().item():.4g}, {ct.max().item():.4g}]")
    for k, v in p.items():
        g = v.grad; bad = (~torch.isfinite(g)).reshape(-1).nonzero().reshape(-1).tolist()
        print(f"   d/d{k:2s}: finite={len(bad)==0}" + (f"  NaN/inf at samples {bad}" if bad else f"  |g| max {g.abs().max().item():.3g}"))
    return ct

# --- 1. plausible parameters, 4-param phantom AIF ---
ke = [0.5, 1.0, 0.2, 1.5]; dt = [0.10, 0.20, 0.15, 0.30]; ve = [0.3, 0.2, 0.5, 0.1]; vp = [0.02, 0.05, 0.01, 0.03]
ct4 = run("cosine4 phantom AIF, plausible", D.ext_tofts_cosine4, ke, dt, ve, vp, aif=aif4)

# --- 2. mapping check: vp=1, ve=0, dt=0 must reproduce our numpy plasma curve cp_curve ---
with torch.no_grad():
    ctp = D.ext_tofts_cosine4(t, aif4, [1e-3], [0.0], [0.0], [1.0], device=dev)[0].cpu().numpy()
cp_np = aif_gt.cp_curve(t, aif4[1], aif4[2], aif4[3], aif4[4], aif4[0])
rel = np.linalg.norm(ctp - cp_np) / (np.linalg.norm(cp_np) + 1e-12)
print(f"\n[mapping] torch Cosine4 (vp=1,ve=0) vs our aif_gt.cp_curve: rel L2 diff {rel:.2e}  ->", "MATCH" if rel < 1e-4 else "MISMATCH")

# --- 3. unstable branches, probed on purpose ---
run("ke -> 0  (SpecialCosineExp (1-e^-x)/x at x=k*t)", D.ext_tofts_cosine4, [1e-8, 1e-4, 0.0, 0.5], [0.1]*4, [0.3]*4, [0.02]*4, aif=aif4)
run("ke == me (ConvBolusExpExp (y1-y2)/(k2-k1))", D.ext_tofts_cosine4, [0.171, 0.171 + 1e-6, 0.171 + 1e-3, 0.5], [0.1]*4, [0.3]*4, [0.02]*4, aif=aif4)
run("dt == t0 exactly (t_eff = 0 sample on grid)", D.ext_tofts_cosine4, [0.5]*4, [aif4[0]]*4, [0.3]*4, [0.02]*4, aif=aif4)

# --- 4. 8-param population AIF, plausible ---
aif8 = D.population_aif8_plasma()
run("cosine8 population AIF, plausible", D.ext_tofts_cosine8, ke, dt, ve, vp, aif=aif8)
run("cosine8, ke == mm and ke == me", D.ext_tofts_cosine8, [aif8["mm"], aif8["me"], 0.5, 1.0], [0.1]*4, [0.3]*4, [0.02]*4, aif=aif8)

# --- 5. sampled-AIF adapter on the SAME plasma curve: should agree with the analytic model ---
with torch.no_grad():
    cs = D.ext_tofts_sampled(t, cp_np, ke, dt, ve, vp, device=dev).cpu().numpy()
rel2 = np.linalg.norm(cs - ct4.detach().cpu().numpy()) / (np.linalg.norm(ct4.detach().cpu().numpy()) + 1e-12)
print(f"\n[sampled adapter] vs analytic cosine4 on identical cp: rel L2 diff {rel2:.2e} (trapezoid on {T} pts)")
p = {k: torch.tensor(v, dtype=torch.float32, device=dev, requires_grad=True) for k, v in dict(ke=ke, dt=dt, ve=ve, vp=vp).items()}
D.ext_tofts_sampled(t, cp_np, p["ke"], p["dt"], p["ve"], p["vp"], device=dev).sum().backward()
print("   sampled adapter grads finite:", all(torch.isfinite(v.grad).all().item() for v in p.values()))
print("\nSMOKE_DONE")
