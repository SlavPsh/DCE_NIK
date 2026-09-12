"""fitted extended-kety parameter maps on the phantom, truth known per label (results/pkLUT), for every method on the same spokes.
signal -> concentration: spgr inversion with the label's T10 (oracle, from XCAT_to_MR_DCE.m), TR / flip / relaxivity from the sim, per-voxel baseline S0.
fit: DCE-NET fit_tofts_model (curve_fit, Cosine4 aif = the sim's aif, ke = kep, ve, vp, dt), enhancing voxels only.
out: <OUT>/pk_maps/pk_maps.npz (maps), pk_maps.json + pk_maps.md (per-tissue medians vs truth), figures/pk_maps_phantom.png
usage: XPH_SIM=nomotion python pk_maps_phantom.py [--jobs 8] [--methods truth,tofts,free,patlak,pro,grasp]"""
import warnings; warnings.filterwarnings("ignore")
import os, re, sys, json, argparse, time, h5py, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/net/beegfs/users/P101440/DCE_NIK"); sys.path.insert(0, "/net/beegfs/users/P101440/DCE_NIK/third_party/DCENET")
import xph_pipeline as P, xph_common as X
import DCE_matt as M
MFILE = "/net/beegfs/users/P101440/XCAT-ERIC/utilities/newContrastCalculation/XCAT_to_MR_DCE.m"
HCT = 0.4; AIF = dict(ab=2.84 / (1 - HCT), mb=22.8, ae=1.36, me=0.171, t0=12.0 / 60.0)          # aif_gt.py, sim convention, minutes
NAMES = {"truth": "truth signal", "tofts": "NIK-tofts", "free": "NIK-free", "patlak": "NIK-patlak", "sub16": "NIK-sub16", "pro": "GRASP-Pro K5 25spf", "grasp": "GRASP 25spf"}

def t1_table():
    """label -> T1 at 3T (ms) from the simulator's Tissue(i,:) = [T1_1.5T; T2_1.5T; T1_3T; T2_3T]"""
    t = {}
    for m in re.finditer(r"^\s*Tissue\((\d+),:\)\s*=\s*\[([^\]]*)\]", open(MFILE).read(), re.M):
        v = [float(x) for x in re.split(r"[;,]", m.group(2)) if x.strip()]
        if len(v) == 4: t[int(m.group(1))] = v[2]
    return t

def spgr_inverse(S, S0, T10_ms, TR_ms, fa_deg, r1):
    """concentration (mM) from the signal ratio S/S0 given the baseline T10; E1 solved from the spgr ratio"""
    a = np.deg2rad(fa_deg); E10 = np.exp(-TR_ms / T10_ms); g = (1 - E10) / (1 - np.cos(a) * E10)
    r = S / (S0 + 1e-12); rg = np.clip(r * g, 1e-6, 0.999999)
    E1 = np.clip((1 - rg) / (1 - rg * np.cos(a)), 1e-9, 0.999999); R1 = -np.log(E1) / TR_ms * 1000.0; R10 = 1000.0 / T10_ms
    return (R1 - R10) / r1

def fit(Ct, t_min, jobs):
    out = np.asarray(M.fit_tofts_model(Ct, t_min, AIF, jobs=jobs, model="Cosine4"))
    return out if out.shape[0] == 4 else out.T                                                                # (4, n): ke, dt, ve, vp (verified on synthetic curves, not the X0 order)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=8); ap.add_argument("--methods", default="truth,tofts,free,patlak,pro,grasp"); ap.add_argument("--max-vox", type=int, default=0)
    a = ap.parse_args(); t0 = time.time()
    d = P.data(); tq = np.asarray(d["times"], float); lab = np.asarray(d["labels"]).astype(int); body = lab > 0; Tr = X.truth_at(P.ZI, tq).astype(np.float32); A = f"{P.OUT}/arrays"
    f = h5py.File(X.SIM, "r"); r = f["results"]; lut = {k: np.asarray(r["pkLUT"][k]).ravel() for k in ("ke", "ve", "vp", "dt")}
    TR = float(np.asarray(r["sim"]["TR"]).ravel()[0]); FA = float(np.asarray(r["sim"]["alpha"]).ravel()[0]); R1X = float(np.asarray(r["sim"]["relaxivity"]).ravel()[0]); TINJ = float(np.asarray(r["sim"]["injectionTime"]).ravel()[0])
    TR_ms = TR * 1000 if TR < 1 else TR; tinj_s = TINJ * 60 if TINJ < 5 else TINJ
    T1 = t1_table(); print(f"sim: TR {TR_ms:.2f} ms, FA {FA:g}, r1 {R1X:g}, injection {tinj_s:g} s; T1 table {len(T1)} tissues; labels in slice {np.unique(lab).tolist()}", flush=True)
    ke_t, ve_t, vp_t, dt_t = (np.where(body, lut[k][np.clip(lab - 1, 0, len(lut[k]) - 1)], np.nan) for k in ("ke", "ve", "vp", "dt"))
    dt_unit = "min" if np.nanmax(dt_t) < 3 else "s"; print("pkLUT dt unit guess:", dt_unit, "| label 5 (artery?) ke/ve/vp/dt:", [float(lut[k][4]) for k in ("ke", "ve", "vp", "dt")], flush=True)
    T10 = np.full(lab.shape, np.nan); [np.putmask(T10, lab == l, T1.get(l, np.nan)) for l in np.unique(lab) if l > 0]
    enh = body & ((ke_t > 0) | (vp_t > 0)) & np.isfinite(T10) & (T10 > 0); idx = np.flatnonzero(enh.ravel())
    if a.max_vox and idx.size > a.max_vox: idx = np.random.default_rng(0).choice(idx, a.max_vox, replace=False)
    print(f"enhancing voxels to fit: {idx.size}", flush=True)
    G = 5; tw = np.array([tq[g*G:(g+1)*G].mean() for g in range(len(tq) // G)])
    def load(key):
        if key == "truth": return Tr, tq
        if key == "tofts": return np.abs(np.load(f"{A}/nik_eval_w768_ks2.5_s0_tofts16.npz", allow_pickle=True)["rec_best"]).astype(np.float32), tq
        if key == "patlak": return np.abs(np.load(f"{A}/nik_eval_w768_ks2.5_s0.npz", allow_pickle=True)["rec_best"]).astype(np.float32), tq
        if key in ("free", "sub16"): return np.abs(np.load(f"{A}/nik_fine_{key}.npy")).astype(np.float32), tq
        if key == "pro": return np.abs(np.load(f"{A}/grasp_pro_K5_G5.npz")["rec"]).astype(np.float32), tw
        if key == "grasp": return np.abs(np.load(f"{P.OUT}/v2_sweep/v2_G05.npy")).astype(np.float32), tw
    maps = {"truth_lut": dict(ke=ke_t, ve=ve_t, vp=vp_t, ktrans=ke_t * ve_t, dt=dt_t)}; rows = {}
    for key in a.methods.split(","):
        v, t = load(key); v = v * (np.sum(v[body] * (Tr if len(t) == len(tq) else np.stack([Tr[:, :, g*G:(g+1)*G].mean(2) for g in range(len(t))], -1))[body]) / (np.sum(v[body]**2) + 1e-12))
        S = v.reshape(-1, v.shape[-1])[idx]; pre = t < tinj_s - 2; S0 = S[:, pre].mean(1, keepdims=True) if pre.sum() >= 2 else S[:, :3].mean(1, keepdims=True)
        C = spgr_inverse(S, S0, T10.ravel()[idx][:, None], TR_ms, FA, R1X); C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
        ts = time.time(); par = fit(C, t / 60.0, a.jobs); print(f"{key}: fitted {idx.size} voxels on {len(t)} frames in {time.time() - ts:.0f} s", flush=True)
        mp = {}
        for j, nm in enumerate(("ke", "dt", "ve", "vp")):
            m = np.full(lab.size, np.nan); m[idx] = par[j]; mp[nm] = m.reshape(lab.shape)
        mp["ktrans"] = mp["ke"] * mp["ve"]; maps[key] = mp
    labs = [int(l) for l in np.unique(lab) if l > 0 and enh[lab == l].sum() > 30]
    for key in maps:
        rows[key] = {}
        for l in labs:
            m = (lab == l) & enh; rows[key][l] = {nm: float(np.nanmedian(maps[key][nm][m])) for nm in ("ke", "ve", "vp", "ktrans", "dt")}
    os.makedirs(f"{P.OUT}/pk_maps", exist_ok=True); np.savez(f"{P.OUT}/pk_maps/pk_maps.npz", labels=lab, enh=enh, **{f"{k}_{nm}": maps[k][nm] for k in maps for nm in maps[k]})
    json.dump(dict(rows=rows, labels=labs, dt_unit=dt_unit, sim=dict(TR_ms=TR_ms, FA=FA, r1=R1X, inj_s=tinj_s), n_fit=int(idx.size)), open(f"{P.OUT}/pk_maps/pk_maps.json", "w"), indent=1)
    lines = [f"# fitted extended-kety parameters on the phantom (per-tissue medians over enhancing voxels; truth from results/pkLUT; {idx.size} voxels fitted per method)", "",
             "signal -> concentration by spgr inversion with the label's T10 (oracle); fit = DCE-NET curve_fit with the sim's Cosine4 aif; ke = kep (1/min), ve, vp, ktrans = ke*ve (1/min), dt (" + dt_unit + ")", ""]
    for nm in ("ktrans", "ve", "vp", "ke"):
        lines += [f"## {nm}", "| label | truth | " + " | ".join(NAMES.get(k, k) for k in maps if k != "truth_lut") + " |", "|---|---|" + "---|" * (len(maps) - 1)]
        for l in labs: lines.append(f"| {l} | {rows['truth_lut'][l][nm]:.3f} | " + " | ".join(f"{rows[k][l][nm]:.3f}" for k in maps if k != "truth_lut") + " |")
        lines.append("")
    open(f"{P.OUT}/pk_maps/pk_maps.md", "w").write("\n".join(lines)); print("\n".join(lines))
    keys = [k for k in maps]; fig, ax = plt.subplots(3, len(keys), figsize=(2.9 * len(keys), 8.6), squeeze=False)
    for j, k in enumerate(keys):
        for i, (nm, vmax) in enumerate((("ktrans", np.nanpercentile(maps["truth_lut"]["ktrans"], 99)), ("ve", 1.0), ("vp", np.nanpercentile(maps["truth_lut"]["vp"], 99)))):
            im = np.where(enh, maps[k][nm], np.nan); ax[i, j].imshow(np.nan_to_num(im), cmap="inferno" if nm == "ktrans" else ("viridis" if nm == "vp" else "magma"), vmin=0, vmax=max(vmax, 1e-3)); ax[i, j].axis("off")
            if i == 0: ax[i, j].set_title("truth (pkLUT)" if k == "truth_lut" else NAMES.get(k, k), fontsize=11, fontweight="bold")
            if j == 0: ax[i, j].text(-0.06, 0.5, {"ktrans": "Ktrans (1/min)", "ve": "ve", "vp": "vp"}[nm], transform=ax[i, j].transAxes, rotation=90, va="center", fontsize=12)
    fig.suptitle("fitted extended-kety maps, phantom, same 5 of 7 spokes per frame; spgr inversion with the true T10, DCE-NET fit with the sim aif", fontsize=12)
    fig.tight_layout(); fig.savefig(f"{P.OUT}/figures/pk_maps_phantom.png", dpi=150, facecolor="white"); print(f"saved {P.OUT}/figures/pk_maps_phantom.png ({time.time() - t0:.0f} s)"); print("PK_MAPS_DONE")

if __name__ == "__main__": main()
