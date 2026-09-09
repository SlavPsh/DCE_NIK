# DCE-NET extended-Tofts as a NIK dependency

**checkout** `third_party/DCENET` @ `6c320ff99e2f5f990a299db8170ff36a06636ca1`, GPL-3.0, upstream source unchanged.
Reproduce: `bash third_party/fetch_dcenet.sh` (reads `DCENET.lock`). No extra packages needed in `torch29`.
Cite: Ottens et al., *Medical Image Analysis* 80, 102512 (2022). https://doi.org/10.1016/j.media.2022.102512

**import** (never `import functions`; the bare name collides)
```python
import sys; sys.path.insert(0, "/net/beegfs/users/P101440/DCE_NIK")
import dcenet_adapter as D
ct = D.ext_tofts_cosine4(t_min, D.phantom_aif4(), ke, dt, ve, vp, device="cuda")   # [N,T] concentration
```

## reusable functions (torch, differentiable) -- `third_party/DCENET/functions.py`
| function | role |
|---|---|
| `Cosine4AIF_ExtKety_deep_aif(t, aif, ke, dt, ve, vp)` | ext-Tofts with 4-param cosine AIF. **our phantom AIF form.** |
| `Cosine8AIF_ExtKety_deep_aif(t, aif, ke, dt, ve, vp)` | ext-Tofts with 8-param cosine AIF (upstream population form) |
| `CosineBolus_deep`, `ConvBolusExp_deep`, `ConvBolusExpExp_deep`, `ConvBolusGamma_deep`, `SpecialCosineExp_deep`, `SpecialCosineGamma_deep` | analytic convolution kernels |

`DCE_matt.py` is the **numpy/scipy fitting** side (`fit_aif`, `fit_tofts_model` via `curve_fit`): separate, non-differentiable. Exposed lazily as `D.fitting()`.

## conventions (verified from code, not README)
- `t`, `dt` in **minutes**; rates `ke, mb, me, mm, mr` in min⁻¹.
- output `ct = vp·cp + ve·ce` is tissue **concentration**, not MR signal; zero for t ≤ 0.
- **`ke` is k_ep.** `Ktrans = ve·ke` is derived (`D.ktrans`), not a model parameter.
- delay `t_eff = t − t0 − dt`: **positive dt → later arrival**.
- shapes: `t [N,T]`; `ke, dt, ve, vp` each `[N,1]`; `aif` `[5,N]` rows `t0,ab,mb,ae,me` (Cosine4) or `[9,N]` rows `t0,tr,ab,mb,ae,me,ar,mm,mr` (Cosine8).
- **hematocrit is applied once, by the caller** (`model.py:152`, `ab_blood/(1−Hct)`); the forward never sees Hct. The adapter takes `ab` **already in plasma units and does not divide**. Our `aif_gt.py` already has `ab = 2.84/(1−0.4)`. `DCE_matt.aif()` bakes Hct into a *different* dict -- do not mix the two.

## AIF status
- **phantom** (`aif_gt.py` Cosine4: ab 2.84/(1−0.4), mb 22.8, ae 1.36, me 0.171, t0 12 s): identical functional form to upstream Cosine4; `D.phantom_aif4()` maps it directly (torch vs numpy rel L2 5.8e-7). Note `aif_gt` folds the 7 s artery transit into `t0_art`; for tissue that belongs in the free `dt`, so `phantom_aif4` uses injection-only t0.
- **in-vivo** (`aif_slice21.npz['aif_frame']`, 342 pts, 0–375 s): an ROI mean of `|image|` from `aif_gate.py` -- **signal domain, arbitrary units, 0…1.9e-3**. It is a *sampled curve*, so it **cannot be passed to the analytic model** (which takes AIF parameters) and it is **not concentration**. Use `D.ext_tofts_sampled` (numerical convolution, validated to 2.4e-3 vs analytic) **after** SPGR inversion (needs T1, relaxivity). Do **not** substitute `population_aif8_plasma()` for it.

## numerical issues (smoke, `dcenet_smoke.py`)
- **ke = 0 exactly → NaN** in `ct` and in `∂/∂ke, ∂/∂dt, ∂/∂ve` (`SpecialCosineExp_deep`: `(1−e^{−x})/x`, `/(x²+y²)` at `x = ke·t = 0`). ke = 1e-8 is finite. numpy version has `errstate`+`where(isfinite)` guards; torch has none. **Bound ke ≥ 1e-6 in any basis; do not `nan_to_num`.**
- equal rates (ke == me, ke == mm): **finite**, forward and backward -- upstream's `tT = tol/|k2−k1|` routes exact equality to the Gamma branch. `torch.abs` at equality gives a 0 subgradient there, acceptable.
- dt == t0 exactly (t_eff = 0 on the grid): finite.
- gradients near ke→0⁺ are finite but large (∝ 1/ke); prefer a log-ke parameterization.

## outstanding
1. in-vivo AIF signal→concentration conversion (SPGR inversion) before any in-vivo Tofts basis.
2. `ext_tofts_sampled` uses trapezoid on the acquisition grid (2.4e-3 error at 120 pts); refine grid or use exponential integrator if higher fidelity is needed.
3. no upstream README statement of units; conventions above come from `hyperparams.py` (`dt = Tonset/60`) and code.
