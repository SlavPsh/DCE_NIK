"""quick in vivo extended-kety maps at the k80 standard, no truth: literature T10 per roi (cortex, medulla, blood; generic elsewhere), spgr inversion
per voxel with its own pre-contrast baseline, aif = model-free aorta curve inverted with blood T10, plasma by (1 - hct), fitted to a Cosine4 aif
(one aif for every method), DCE-NET curve_fit per body voxel. rulers: per-roi medians and iqr per method, consistency across slices.
TR and flip angle are read from the twix header of the dce scan.
out: results/realdata_nik_vs_cs_figures/pk_maps/pk_invivo_k80_sl<Z>.{npz,json}, pk_invivo_k80.md, figures/pk_maps_invivo_k80_sl<Z>.png
usage: python pk_maps_invivo.py --slices 21,18,19 [--jobs 16] [--hct 0.4] [--r1 3.5] [--t10 cortex=1142,medulla=1545,blood=1650,other=1000]"""
import warnings; warnings.filterwarnings("ignore")
import os, re, sys, json, glob, argparse, time, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/net/beegfs/users/P101440/DCE_NIK"); sys.path.insert(0, "/net/beegfs/users/P101440/DCE_NIK/third_party/DCENET")
import consolidated as C, DCE_matt as M
B = "/net/beegfs/users/P101440/DCE_NIK"; GV = "/net/beegfs/users/P101440/grasp_v2/results_grasp_v2"; GP = "/net/beegfs/users/P101440/grasp_pro_py/results_spoke_cs"
DAT = "/net/beegfs/users/P101440/dce_data/orig/meas_p3_dce.dat"; TA = 375.0; OUT = f"{B}/results/realdata_nik_vs_cs_figures"
NAMES = {"free": "NIK-free", "sub16": "NIK-sub16", "patlak": "NIK-patlak", "tofts": "NIK-tofts", "tofts8": "NIK-tofts8", "pro": "GRASP-Pro f80match", "grasp": "GRASP n12 k80"}

def header_params():
    """TR (ms) and flip (deg) of the LAST measurement in the twix file (the dce scan; adjustment scans come first)"""
    import struct
    with open(DAT, "rb") as f:
        head = f.read(10240); n = struct.unpack("<I", head[4:8])[0]
        off = [struct.unpack("<Q", head[8 + 152 * k + 8: 8 + 152 * k + 16])[0] for k in range(n)]
        f.seek(off[-1]); txt = f.read(8_000_000).decode("latin-1")
    tr = re.findall(r"alTR\[0\]\s*=\s*([0-9.]+)", txt); fa = re.findall(r"adFlipAngleDegree\[0\]\s*=\s*([0-9.]+)", txt); seq = re.findall(r"tSequenceFileName\s*=\s*\"([^\"]*)\"", txt)
    print(f"twix: {n} measurements, last at offset {off[-1]}, sequence {seq[-1] if seq else '?'}, TR matches {tr[:3]}, flip matches {fa[:3]}", flush=True)
    return (float(tr[-1]) / 1000.0 if tr else None), (float(fa[-1]) if fa else None)                   # TR us -> ms

def spgr_inverse(S, S0, T10_ms, TR_ms, fa_deg, r1):
    a = np.deg2rad(fa_deg); E10 = np.exp(-TR_ms / T10_ms); g = (1 - E10) / (1 - np.cos(a) * E10)
    rg = np.clip(S / (S0 + 1e-12) * g, 1e-6, 0.999999); E1 = np.clip((1 - rg) / (1 - rg * np.cos(a)), 1e-9, 0.999999)
    return (-np.log(E1) / TR_ms * 1000.0 - 1000.0 / T10_ms) / r1

def ft(nt): e = np.linspace(0, TA, nt + 1); return 0.5 * (e[:-1] + e[1:])

def sources(Z):
    IV = f"{B}/results/tofts_vs_patlak/invivo_k80"; s = {}
    if Z == 21:
        s["free"] = f"{B}/results_sl21_k80/nik_slice_21.npy"; s["sub16"] = f"{OUT}/outcoil_subspace_output_slice21.npy"
    for a in ("patlak", "tofts", "tofts8"): s[a] = f"{IV}/{a}_sl{Z}_s0/nik_slice_{Z:02d}_cplx.npy"
    s["pro"] = f"{GP}/cs_slice{Z}_f80match.npy"; s["grasp"] = f"{GV}/gv2_slice{Z}_n12_k80.npy"
    return {k: p for k, p in s.items() if os.path.exists(p)}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--slices", default="21,18,19"); ap.add_argument("--jobs", type=int, default=16); ap.add_argument("--hct", type=float, default=0.4); ap.add_argument("--r1", type=float, default=3.5)
    ap.add_argument("--t10", default="cortex=1142,medulla=1545,blood=1650,other=1000"); ap.add_argument("--TR", type=float, default=None); ap.add_argument("--FA", type=float, default=None); ap.add_argument("--pre-s", type=float, default=45.0)
    a = ap.parse_args(); t0 = time.time(); T10 = {k: float(v) for k, v in (kv.split("=") for kv in a.t10.split(","))}
    TR, FA = header_params(); TR = a.TR or TR; FA = a.FA or FA; print(f"header: TR {TR} ms, flip {FA} deg; hct {a.hct}, r1 {a.r1}, T10 {T10}", flush=True)
    os.makedirs(f"{OUT}/pk_maps", exist_ok=True); md = ["# in vivo extended-kety maps at k80 (no truth; literature T10 per roi, model-free aorta aif, one aif for every method)", "",
                                                        f"TR {TR:.2f} ms, flip {FA:g}, hct {a.hct}, r1 {a.r1} /mM/s, T10 {T10} ms; DCE-NET curve_fit; ke = kep (1/min), ktrans = ke * ve (1/min); per-roi median [iqr]", ""]
    for Z in [int(z) for z in a.slices.split(",")]:
        ctx = C.slice_ctx(Z); rois = ctx["rois"]; body = ctx["BODY"]; z = np.load(f"{B}/step2_slice{Z}.npz"); mf = np.abs(z["mf"]).transpose(1, 2, 0).astype(np.float32); tmf = np.asarray(z["tmf"]).astype(np.float64)
        t10map = np.full(body.shape, T10["other"]); t10map[rois["cortex"]] = T10["cortex"]; t10map[rois["medulla"]] = T10["medulla"]; t10map[rois["aorta"]] = T10["blood"]
        # aif from the model-free aorta curve
        sa = np.array([mf[..., i][rois["aorta"]].mean() for i in range(mf.shape[-1])]); pre = tmf < a.pre_s; cb = spgr_inverse(sa, sa[pre].mean(), T10["blood"], TR, FA, a.r1); cp = np.clip(cb / (1 - a.hct), 0, None)
        aif = M.fit_aif(cp, tmf / 60.0, model="Cosine4"); fitc = M.Cosine4AIF(tmf / 60.0, aif["ab"], aif["ae"], aif["mb"], aif["me"], aif["t0"])
        print(f"slice {Z}: aif peak {cp.max():.2f} mM at {tmf[cp.argmax()]:.0f} s; cosine4 fit {dict((k, round(float(v), 3)) for k, v in aif.items())}, fit nrmse {np.linalg.norm(fitc - cp) / np.linalg.norm(cp):.3f}", flush=True)
        idx = np.flatnonzero(body.ravel()); maps = {}; rows = {}
        for key, p in sources(Z).items():
            v = np.abs(np.load(p)).astype(np.float32); t = ft(v.shape[-1]); S = v.reshape(-1, v.shape[-1])[idx]; pre = t < a.pre_s
            S0 = S[:, pre].mean(1, keepdims=True); Cc = np.nan_to_num(spgr_inverse(S, S0, t10map.ravel()[idx][:, None], TR, FA, a.r1))
            ts = time.time(); par = np.asarray(M.fit_tofts_model(Cc, t / 60.0, aif, jobs=a.jobs, model="Cosine4")); par = par if par.shape[0] == 4 else par.T     # (ke, dt, ve, vp)
            mp = {}
            for j, nm in enumerate(("ke", "dt", "ve", "vp")):
                m = np.full(body.size, np.nan); m[idx] = par[j]; mp[nm] = m.reshape(body.shape)
            mp["ktrans"] = mp["ke"] * mp["ve"]; maps[key] = mp
            rows[key] = {r: {nm: [float(np.nanpercentile(mp[nm][rois[r]], q)) for q in (50, 25, 75)] for nm in ("ktrans", "ve", "vp", "ke")} for r in ("cortex", "medulla") if rois[r].sum() > 0}
            print(f"  {NAMES[key]:20s} {len(t):3d} fr, {idx.size} voxels, {time.time() - ts:.0f} s | cortex ktrans {rows[key]['cortex']['ktrans'][0]:.3f} ve {rows[key]['cortex']['ve'][0]:.3f} vp {rows[key]['cortex']['vp'][0]:.3f} | medulla ktrans {rows[key]['medulla']['ktrans'][0]:.3f} ve {rows[key]['medulla']['ve'][0]:.3f} vp {rows[key]['medulla']['vp'][0]:.3f}", flush=True)
        np.savez(f"{OUT}/pk_maps/pk_invivo_k80_sl{Z}.npz", body=body, **{f"{k}_{nm}": maps[k][nm] for k in maps for nm in maps[k]})
        json.dump(dict(rows=rows, aif=aif, TR=TR, FA=FA, hct=a.hct, r1=a.r1, T10=T10), open(f"{OUT}/pk_maps/pk_invivo_k80_sl{Z}.json", "w"), indent=1)
        md += [f"## slice {Z}", "| method | cortex ktrans | cortex ve | cortex vp | medulla ktrans | medulla ve | medulla vp |", "|---|---|---|---|---|---|---|"]
        for key in maps: md.append(f"| {NAMES[key]} | " + " | ".join(f"{rows[key][r][nm][0]:.3f} [{rows[key][r][nm][1]:.2f}, {rows[key][r][nm][2]:.2f}]" for r in ("cortex", "medulla") for nm in ("ktrans", "ve", "vp")) + " |")
        md.append("")
        keys = list(maps); fig, ax = plt.subplots(3, len(keys), figsize=(2.9 * len(keys), 8.8), squeeze=False); anat = mf.mean(-1)
        for j, k in enumerate(keys):
            for i, (nm, vmax) in enumerate((("ktrans", 1.0), ("ve", 1.0), ("vp", 0.3))):
                im = np.where(body, maps[k][nm], np.nan); ax[i, j].imshow(anat, cmap="gray", vmin=0, vmax=np.percentile(anat[body], 99.5)); ax[i, j].imshow(np.ma.masked_invalid(im), cmap="inferno" if nm == "ktrans" else ("viridis" if nm == "vp" else "magma"), vmin=0, vmax=vmax, alpha=0.85); ax[i, j].axis("off")
                if i == 0: ax[i, j].set_title(NAMES[k], fontsize=11, fontweight="bold")
                if j == 0: ax[i, j].text(-0.06, 0.5, {"ktrans": "Ktrans (1/min)", "ve": "ve", "vp": "vp"}[nm], transform=ax[i, j].transAxes, rotation=90, va="center", fontsize=12)
        fig.suptitle(f"in vivo slice {Z}, k80 (1368 views) for every method: fitted extended-kety maps, no truth; literature T10, model-free aorta aif", fontsize=12)
        fig.tight_layout(); fig.savefig(f"{OUT}/figures/pk_maps_invivo_k80_sl{Z}.png", dpi=150, facecolor="white"); plt.close(fig); print(f"saved {OUT}/figures/pk_maps_invivo_k80_sl{Z}.png", flush=True)
    open(f"{OUT}/pk_maps/pk_invivo_k80.md", "w").write("\n".join(md)); print("\n".join(md)); print(f"PK_INVIVO_DONE ({time.time() - t0:.0f} s)")

if __name__ == "__main__": main()
