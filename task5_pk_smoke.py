"""TASK 5 Parts 10+14: PK truth-recovery check + end-to-end integration smoke test (read-only).
(1) Confirm whether a PHYSICAL PK truth is constructible from the XCAT .mat (pkLUT x labelGT) and
whether any S->concentration->PK fitter exists to recover it. (2) Smoke-test truth+NIK paths on the
Task-4 basis-matched arm (image metrics, one ROI curve, shapes/times); report GRASP-Pro XCAT-domain
output as a missing dependency. No training, no reconstruction, no data generation."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, csv, os, h5py
from scipy.ndimage import uniform_filter
OUT = "/net/beegfs/users/P101440/DCE_NIK/results/task5_evaluation_code_audit"
MAT = "/net/beegfs/users/P101440/XCAT-ERIC/results/simulation_results_20260527T175428.mat"
T4 = "/net/beegfs/users/P101440/DCE_NIK/results/task4_xcat_nomotion_pilot/arrays"
rows_pk, rows_sm = [], []
def R(a, b, m):
    return float(np.sqrt(np.mean((np.abs(a)[m]-np.abs(b)[m])**2))/(np.abs(b)[m].max()-np.abs(b)[m].min()+1e-12))

# ---------- (1) physical PK truth constructibility ----------
try:
    with h5py.File(MAT, "r") as f:
        def deref(name):
            g = f["results"];
            for k in name.split("/"): g = g[k]
            return g
        lab = np.array(f["results"]["labelGT"]); pk = f["results"]["pkLUT"]
        pk_fields = list(pk.keys()) if isinstance(pk, h5py.Group) else "struct-array"
        # pkLUT is a struct array {ve,vp,ke,dt}; read ve,vp,ke
        def col(field):
            arr = pk[field] if isinstance(pk, h5py.Group) else None
            return np.array(arr).ravel() if arr is not None else None
        ve = col("ve"); vp = col("vp"); ke = col("ke")
        gt_shape = f["results"]["images"]["GroundTruth"]["img"].shape
        nlab = int(lab.max()) + 1
        rows_pk.append(dict(check="physical_XCAT_signal_truth", status="PRESENT",
                            detail=f"GroundTruth.img shape {gt_shape} (concentration->R1->SPGR, extended-Tofts)"))
        rows_pk.append(dict(check="true_PK_maps", status="PER-TISSUE-CLASS ONLY",
                            detail=f"pkLUT fields {pk_fields}, {0 if ve is None else ve.size} entries; label map max {int(lab.max())}. Ktrans=ke*ve derivable; PIECEWISE-CONSTANT, no voxelwise heterogeneity"))
        if ve is not None and ke is not None:
            ktrans = ke * ve
            rows_pk.append(dict(check="derived_Ktrans_range_permin", status="CONSTRUCTIBLE",
                                detail=f"Ktrans=ke*ve range [{np.nanmin(ktrans):.4g},{np.nanmax(ktrans):.4g}] /min; vp range [{np.nanmin(vp):.3g},{np.nanmax(vp):.3g}]"))
        print("physical PK truth: GroundTruth.img", gt_shape, "| pkLUT", pk_fields, "| labels", nlab)
except Exception as e:
    rows_pk.append(dict(check="physical_XCAT_read", status="ERROR", detail=str(e)[:120]))
    print("mat read error:", e)
# fitter presence (established by code audit)
rows_pk.append(dict(check="signal_to_concentration_step", status="MISSING", detail="no relaxivity/T1/FA/TR/SPGR-inverse anywhere in fitting path (code audit)"))
rows_pk.append(dict(check="nonlinear_PK_fitter", status="MISSING", detail="no Tofts/extended-Tofts/Patlak optimizer; only linear pinv(Phi) onto F0 SIGNAL basis (task4_csfit.py, task2_*)"))
rows_pk.append(dict(check="truth_recovery_test_runnable", status="BLOCKED", detail="cannot fit physical GroundTruth->PK: no S->C->fit pipeline exists. Only basis-matched linear recovery is validated (Task-4, NRMSE ~1e-3 to truth-in-basis)"))
# achievable context: linear F0-basis recovery on the basis-matched arm (already validated in Task-4)
try:
    S = np.load(f"{T4}/sim.npz"); th = S["theta_true"]; Phi = S["Phi"]; body = S["labels"] > 0
    I_true = S["I_true"]; th_lin = np.einsum("rt,xyt->xyr", np.linalg.pinv(Phi), I_true)  # linear fit of truth-in-basis
    rows_pk.append(dict(check="linear_basis_recovery_of_basis_matched_truth", status="OK(context)",
                        detail=f"pinv(Phi)@I_true recovers theta_true intAIF NRMSE {R(th_lin[...,1],th[...,1],body):.2e} (validates linear inversion, NOT physical PK)"))
except Exception as e:
    rows_pk.append(dict(check="linear_basis_recovery", status="ERROR", detail=str(e)[:100]))
with open(f"{OUT}/pk_truth_recovery.csv", "w", newline="") as fp:
    w = csv.DictWriter(fp, fieldnames=["check", "status", "detail"]); w.writeheader(); [w.writerow(r) for r in rows_pk]

# ---------- (2) smoke test: truth + NIK on the basis-matched Task-4 arm ----------
def ssim(a, b, rv, win=7):
    a = np.abs(a).astype(float); b = np.abs(b).astype(float); C1 = (0.01*rv)**2; C2 = (0.03*rv)**2
    ma = uniform_filter(a, win); mb = uniform_filter(b, win)
    va = uniform_filter(a*a, win)-ma**2; vb = uniform_filter(b*b, win)-mb**2; vab = uniform_filter(a*b, win)-ma*mb
    return float((((2*ma*mb+C1)*(2*vab+C2))/((ma**2+mb**2+C1)*(va+vb+C2))).mean())
S = np.load(f"{T4}/sim.npz"); R2 = np.load(f"{T4}/rois.npz"); times = S["times"]; Phi = S["Phi"]; body = S["labels"] > 0
I_true = S["I_true"]  # [N,N,F] truth dynamic (basis-matched arm)
def add(k, v): rows_sm.append(dict(item=k, value=str(v)))
add("xcat_truth_dynamic_shape", I_true.shape); add("truth_times_s", f"{times.min():.2f}..{times.max():.2f} ({len(times)} frames)")
# NIK output (Task-4 thetaC -> dynamic on the SAME frame times = common grid)
nik = np.load(f"{T4}/nik_F0_f25_seed0.npz")["thetaC"]
s = np.vdot(nik[body], S["theta_true"][body])/(np.vdot(nik[body], nik[body])+1e-12)  # one global complex scale
I_nik = np.einsum("xyr,tr->xyt", nik*s, Phi)  # NIK queried at the SAME frame centres
add("nik_dynamic_shape", I_nik.shape); add("nik_output_domain", "complex coil-combined signal (scale-matched)")
add("common_time_grid", "identical frame_time (NIK queried at truth/GRASP frame centres)")
rv = np.abs(I_true[body]).max() - np.abs(I_true[body]).min()
# frame-level image metrics (report a few frames + mean), body-masked, ONE truth-derived range
fr = [5, 30, 90, 180]
for t in fr:
    add(f"NRMSE_frame{t}_bodymask", round(R(I_nik[..., t], I_true[..., t], body), 4))
    add(f"SSIM_frame{t}", round(ssim(I_nik[..., t], I_true[..., t], rv), 4))
add("NRMSE_allframes_mean_bodymask", round(np.mean([R(I_nik[..., t], I_true[..., t], body) for t in range(len(times))]), 4))
# one ROI curve (aorta), physical time, signal domain, mean over ROI
aorta = R2["aorta"]; ca_true = np.abs(I_true[aorta].mean(0)); ca_nik = np.abs(I_nik[aorta].mean(0))
add("aorta_curve_NRMSE", round(float(np.linalg.norm(ca_nik-ca_true)/(np.linalg.norm(ca_true)+1e-12)), 4))
add("aorta_ROI_voxels", int(aorta.sum()))
# PK fit on truth: not runnable
add("PK_fit_on_truth", "NOT RUN - no S->concentration->PK fitter exists (see pk_truth_recovery.csv)")
# GRASP-Pro XCAT-domain output dependency
gp = "/net/beegfs/users/P101440/grasp_pro_py/results_spoke_cs"
gp_exists = os.path.isdir(gp)
add("grasppro_output_for_XCAT", "MISSING - on-disk GRASP-Pro outputs (%s) are IN-VIVO (real twix), not the XCAT sim" % ("present but in-vivo" if gp_exists else "absent"))
with open(f"{OUT}/smoke_test_results.csv", "w", newline="") as fp:
    w = csv.DictWriter(fp, fieldnames=["item", "value"]); w.writeheader(); [w.writerow(r) for r in rows_sm]
print("SMOKE: NIK vs truth mean bodymask NRMSE", [r["value"] for r in rows_sm if r["item"]=="NRMSE_allframes_mean_bodymask"])
print("PK_SMOKE_DONE")
