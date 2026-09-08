"""namespaced adapter for the DCE-NET extended-Tofts forward model (Ottens et al., MedIA 80, 102512, 2022).

upstream: third_party/DCENET @ the SHA in third_party/DCENET.lock, GPL-3.0, source UNCHANGED. all
compatibility lives here. loaded by FILE PATH under private module names, so nothing ever does
`import functions` (that bare name collides with anything else called functions.py on sys.path).

conventions of the upstream torch routines, verified from functions.py / model.py / hyperparams.py:
  time      t and dt in MINUTES. rates ke, mb, me, mm, mr in min^-1.
  domain    output is tissue CONCENTRATION ct = vp*cp + ve*ce (plasma-based, mM). NOT MR signal.
            zero for t <= 0 after the delay shift.
  parameter ke is k_ep (efflux rate). Ktrans = ve * ke is NOT a parameter of the model.
  delay     t_eff = t - t0 - dt, so a POSITIVE dt moves arrival LATER.
  shapes    t [N,T]; ke, dt, ve, vp each [N,1]; aif [n_param, N] (some rows per-sample, some read at [k,0]).
  Hct       upstream applies hematocrit ONCE, in model.py, as ab_blood/(1-Hct) before calling the forward.
            the forward itself never touches Hct. THIS ADAPTER TAKES ab ALREADY IN PLASMA UNITS AND DOES
            NOT DIVIDE AGAIN. our phantom aif_gt.py already has ab = 2.84/(1-0.4).
  torch vs numpy: functions.py = torch, differentiable (used here). DCE_matt.py = numpy/scipy curve_fit
            FITTING routines (fit_aif, fit_tofts_model) -- a separate, non-differentiable code path.
"""
import importlib.util as _ilu
import os as _os
import numpy as np
import torch

_ROOT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "third_party", "DCENET")

def _load(fname, modname):
    """exec upstream file under a private name. registered in sys.modules so pickling / torch.save of
    objects referencing it works. bytecode writing is suppressed so NOTHING is written into the
    upstream tree (a __pycache__ there would show as a local modification)."""
    import sys as _sys
    if modname in _sys.modules: return _sys.modules[modname]
    spec = _ilu.spec_from_file_location(modname, _os.path.join(_ROOT, fname))
    mod = _ilu.module_from_spec(spec)
    _prev = _sys.dont_write_bytecode; _sys.dont_write_bytecode = True
    try: spec.loader.exec_module(mod)
    finally: _sys.dont_write_bytecode = _prev
    _sys.modules[modname] = mod; return mod

F = _load("functions.py", "dcenet_functions")          # torch forward model (what we use)
_fit = None
def fitting():
    """lazy: DCE_matt.py, the numpy/scipy FITTING side. only if you actually want curve_fit."""
    global _fit
    if _fit is None: _fit = _load("DCE_matt.py", "dcenet_fit")
    return _fit

def upstream_sha():
    with open(_os.path.join(_ROOT, "..", "DCENET.lock")) as f:
        return next(l.split("=", 1)[1].strip() for l in f if l.startswith("sha="))

# ---------------- helpers ----------------
def _n(x):
    """element count for torch tensors, numpy arrays, lists or scalars. np.size() on a torch tensor
    returns the bound .size METHOD, not an int, which is why this exists."""
    return int(x.numel()) if torch.is_tensor(x) else int(np.size(x))

def _col(x, N, device):
    """-> [N,1] float tensor"""
    x = torch.as_tensor(x, dtype=torch.float32, device=device).reshape(-1)
    if x.numel() == 1: x = x.expand(N)
    return x.reshape(N, 1)

def ktrans(ve, ke):
    """Ktrans = ve * k_ep. the model parameter is ke; this is derived."""
    return ve * ke

# ---------------- analytic AIF forward models ----------------
def ext_tofts_cosine4(t_min, aif4, ke, dt, ve, vp, device="cpu"):
    """4-parameter Cosine AIF (our phantom's form). aif4 = [t0_min, ab_plasma, mb, ae, me].
    ab_plasma must ALREADY include 1/(1-Hct). returns concentration ct [N,T]."""
    t = torch.as_tensor(t_min, dtype=torch.float32, device=device)
    if t.ndim == 1: t = t[None, :]
    N = max(t.shape[0], _n(ke), _n(ve), _n(vp), _n(dt))
    if t.shape[0] == 1 and N > 1: t = t.expand(N, -1)
    t = t.contiguous()
    a = torch.as_tensor(np.asarray(aif4, float).reshape(5), dtype=torch.float32, device=device)
    aif = torch.zeros(5, N, device=device); aif[:] = a[:, None]     # rows 1,3 read per-sample, 0,2,4 at [k,0]
    return F.Cosine4AIF_ExtKety_deep_aif(t, aif, _col(ke, N, device), _col(dt, N, device),
                                         _col(ve, N, device), _col(vp, N, device), device=device)

COSINE8_ROWS = ("t0", "tr", "ab", "mb", "ae", "me", "ar", "mm", "mr")   # upstream aif tensor row order

def ext_tofts_cosine8(t_min, aif8, ke, dt, ve, vp, device="cpu"):
    """8-parameter Cosine AIF (upstream's population form). aif8 = dict with keys COSINE8_ROWS.
    aif8['ab'] must be PLASMA (already /(1-Hct)); this does NOT divide. returns ct [N,T]."""
    t = torch.as_tensor(t_min, dtype=torch.float32, device=device)
    if t.ndim == 1: t = t[None, :]
    N = max(t.shape[0], _n(ke), _n(ve), _n(vp), _n(dt))
    if t.shape[0] == 1 and N > 1: t = t.expand(N, -1)
    t = t.contiguous()
    aif = torch.zeros(9, N, device=device)
    for i, k in enumerate(COSINE8_ROWS): aif[i] = float(aif8[k])
    return F.Cosine8AIF_ExtKety_deep_aif(t, aif, _col(ke, N, device), _col(dt, N, device),
                                         _col(ve, N, device), _col(vp, N, device), device=device)

def phantom_aif4(hct=0.40, t0_min=12.0/60.0):
    """our XCAT phantom AIF, the exact constants of aif_gt.py, as upstream's 4-param vector.
    ab is returned in PLASMA units (2.84/(1-hct)); pass straight into ext_tofts_cosine4.
    t0 = injection onset only. the 7 s artery transit that aif_gt folds into t0_art belongs in the
    free delay dt when fitting tissue, so it is NOT included here."""
    return np.array([t0_min, 2.84 / (1.0 - hct), 22.8, 1.36, 0.171], float)

def population_aif8_plasma(hct=0.40):
    """upstream's population AIF (hyperparams.AIF_parameters), H&N patients, converted to plasma
    the way model.py does it: ab_blood/(1-Hct). for reference / smoke only -- NOT a substitute for
    our own AIF."""
    d = dict(ab=7.9785, ae=0.5216, ar=0.0482, mb=32.8855, me=0.1811, mm=9.1868, mr=15.8167, t0=0.0, tr=0.2533)
    d["ab"] = d["ab"] / (1.0 - hct); return d

# ---------------- sampled AIF (what our in-vivo aif_slice21.npz actually is) ----------------
def ext_tofts_sampled(t_min, cp_min, ke, dt, ve, vp, device="cpu"):
    """extended Tofts with a SAMPLED plasma AIF cp(t) on grid t_min (minutes), differentiable in
    ke/dt/ve/vp.  ct = vp*cp(t-dt) + ve*ke * int_0^t cp(tau-dt) exp(-ke (t-tau)) dtau  (trapezoid).
    cp_min MUST be plasma CONCENTRATION. our aif_slice21.npz['aif_frame'] is a SIGNAL-domain ROI
    mean, not concentration: it needs SPGR inversion (T1, relaxivity) before it can be used here.
    this adapter exists so the analytic-AIF routines are never silently replaced by upstream's
    population AIF."""
    t = torch.as_tensor(t_min, dtype=torch.float32, device=device).reshape(-1)
    cp = torch.as_tensor(cp_min, dtype=torch.float32, device=device).reshape(-1)
    N = max(_n(ke), _n(ve), _n(vp), _n(dt))
    ke, dt, ve, vp = (_col(x, N, device) for x in (ke, dt, ve, vp))
    ts = t[None, :] - dt                                         # [N,T] shifted time
    # cp(t - dt) by linear interpolation on the sampled grid (zero before onset)
    idx = torch.clamp(torch.searchsorted(t, ts.contiguous(), right=True) - 1, 0, t.numel() - 2)
    t0 = t[idx]; t1 = t[idx + 1]; w = (ts - t0) / (t1 - t0)
    cps = torch.where(ts >= t[0], cp[idx] + w * (cp[idx + 1] - cp[idx]), torch.zeros_like(ts))
    # convolution with ke*exp(-ke t): cumulative trapezoid of cps*exp(ke*tau), times exp(-ke*t)
    dtau = torch.diff(t)
    g = cps * torch.exp(ke * t[None, :])
    cum = torch.cat([torch.zeros(N, 1, device=device), torch.cumsum(0.5 * (g[:, 1:] + g[:, :-1]) * dtau[None, :], 1)], 1)
    ce = ke * torch.exp(-ke * t[None, :]) * cum
    return vp * cps + ve * ce
