"""in vivo extended-kety maps on the k100 standard recons, dataset-aware (p3 kidney, p14 liver), no truth: literature T10 per roi (generic
elsewhere), spgr inversion per voxel with its own pre-contrast baseline, aif = model-free aorta curve inverted with blood T10, plasma by (1 - hct),
fitted to a Cosine4 aif (one aif for every method), DCE-NET curve_fit per body voxel. TR and flip from the twix header of the dataset.
generalized copy of pk_maps_invivo.py (that file stays as the k80 / p3 version). caveat p14: the aorta is inflow-bright before contrast, so the
spgr-inverted aif underestimates the arterial concentration there; the maps are then relative between methods, not absolute.
out: results/realdata_nik_vs_cs_figures/pk_maps/pk_<tag>_sl<Z>.{npz,json}, pk_<tag>.md, figures/pk_maps_<tag>_sl<Z>.png
usage: DCE_DS=p14 python pk_maps_k100.py --slices 21,24,27 --tag p14_k100std --items "label:path-with-{Z},..." --t10 liver=810,spleen=1330,blood=1650,other=1000"""
import warnings; warnings.filterwarnings("ignore")
import os, re, sys, json, argparse, time, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/net/beegfs/users/P101440/DCE_NIK"); sys.path.insert(0, "/net/beegfs/users/P101440/DCE_NIK/third_party/DCENET")
import dsp, consolidated as C, DCE_matt as M
B = "/net/beegfs/users/P101440/DCE_NIK"; OUT = f"{B}/results/realdata_nik_vs_cs_figures"; TA = dsp.TA

def header_params():
    import struct
    with open(dsp.RAW, "rb") as f:
        head = f.read(10240); n = struct.unpack("<I", head[4:8])[0]
        off = [struct.unpack("<Q", head[8 + 152 * k + 8: 8 + 152 * k + 16])[0] for k in range(n)]
        f.seek(off[-1]); txt = f.read(8_000_000).decode("latin-1")
    tr = re.findall(r"alTR\[0\]\s*=\s*([0-9.]+)", txt); fa = re.findall(r"adFlipAngleDegree\[0\]\s*=\s*([0-9.]+)", txt)
    return (float(tr[-1]) / 1000.0 if tr else None), (float(fa[-1]) if fa else None)

def spgr_inverse(S, S0, T10_ms, TR_ms, fa_deg, r1):
    a = np.deg2rad(fa_deg); E10 = np.exp(-TR_ms / T10_ms); g = (1 - E10) / (1 - np.cos(a) * E10)
    rg = np.clip(S / (S0 + 1e-12) * g, 1e-6, 0.999999); E1 = np.clip((1 - rg) / (1 - rg * np.cos(a)), 1e-9, 0.999999)
    return (-np.log(E1) / TR_ms * 1000.0 - 1000.0 / T10_ms) / r1

def ft(nt): e = np.linspace(0, TA, nt + 1); return 0.5 * (e[:-1] + e[1:])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--slices", required=True); ap.add_argument("--items", required=True, help="label:path with {Z} for the slice, comma separated")
    ap.add_argument("--tag", required=True); ap.add_argument("--jobs", type=int, default=16); ap.add_argument("--hct", type=float, default=0.4); ap.add_argument("--r1", type=float, default=3.5)
    ap.add_argument("--t10", default="cortex=1142,medulla=1545,liver=810,spleen=1330,blood=1650,other=1000"); ap.add_argument("--TR", type=float, default=None); ap.add_argument("--FA", type=float, default=None); ap.add_argument("--pre-s", type=float, default=45.0)
    a = ap.parse_args(); t0 = time.time(); T10 = {k: float(v) for k, v in (kv.split("=") for kv in a.t10.split(","))}
    TR, FA = header_params(); TR = a.TR or TR; FA = a.FA or FA; print(f"{dsp.DS}: TR {TR} ms, flip {FA} deg; hct {a.hct}, r1 {a.r1}, T10 {T10}", flush=True)
    items = [(sp.split(":", 1)[0], sp.split(":", 1)[1]) for sp in a.items.split(",") if sp]; R1, R2 = dsp.T1, dsp.T2
    os.makedirs(f"{OUT}/pk_maps", exist_ok=True); md = [f"# in vivo extended-kety maps, {dsp.DS}, {a.tag} (no truth; literature T10 per roi, model-free aorta aif, one aif for every method)", "",
                                                        f"TR {TR:.2f} ms, flip {FA:g}, hct {a.hct}, r1 {a.r1} /mM/s, T10 {T10} ms; DCE-NET curve_fit; ke = kep (1/min), ktrans = ke * ve (1/min); per-roi median [iqr]" + ("; p14 aorta inflow-bright before contrast: aif underestimated, maps relative between methods" if dsp.DS != "p3" else ""), ""]
    for Z in [int(z) for z in a.slices.split(",")]:
        ctx = C.slice_ctx(Z); rois = ctx["rois"]; body = ctx["BODY"]; z = np.load(dsp.STEP2(Z)); mf = np.abs(z["mf"]).transpose(1, 2, 0).astype(np.float32); tmf = np.asarray(z["tmf"]).astype(np.float64)
        t10map = np.full(body.shape, T10["other"])
        for r in (R1, R2):
            if r in rois and r in T10: t10map[rois[r]] = T10[r]
        t10map[rois["aorta"]] = T10["blood"]
        sa = np.array([mf[..., i][rois["aorta"]].mean() for i in range(mf.shape[-1])]); pre = tmf < a.pre_s; cb = spgr_inverse(sa, sa[pre].mean(), T10["blood"], TR, FA, a.r1); cp = np.clip(cb / (1 - a.hct), 0, None)
        aif = M.fit_aif(cp, tmf / 60.0, model="Cosine4"); fitc = M.Cosine4AIF(tmf / 60.0, aif["ab"], aif["ae"], aif["mb"], aif["me"], aif["t0"])
        print(f"slice {Z}: aif peak {cp.max():.2f} mM at {tmf[cp.argmax()]:.0f} s; cosine4 fit nrmse {np.linalg.norm(fitc - cp) / np.linalg.norm(cp):.3f}", flush=True)
        idx = np.flatnonzero(body.ravel()); maps = {}; rows = {}
        late = mf[..., tmf > 150].mean(-1); base = mf[..., tmf < a.pre_s].mean(-1); enh = body & (late > 1.35 * base); enh60 = body & (late > 1.6 * base)
        for key, pth in items:
            p = pth.replace("{Z}", str(Z)); p2 = pth.replace("{Z}", f"{Z:02d}")
            p = p if os.path.exists(p) else p2
            if not os.path.exists(p): print("missing", p); continue
            v = np.abs(np.load(p)).astype(np.float32); t = ft(v.shape[-1]); S = v.reshape(-1, v.shape[-1])[idx]; pre = t < a.pre_s
            S0 = S[:, pre].mean(1, keepdims=True); Cc = np.nan_to_num(spgr_inverse(S, S0, t10map.ravel()[idx][:, None], TR, FA, a.r1))
            ts = time.time(); par = np.asarray(M.fit_tofts_model(Cc, t / 60.0, aif, jobs=a.jobs, model="Cosine4")); par = par if par.shape[0] == 4 else par.T
            mp = {}
            for j, nm in enumerate(("ke", "dt", "ve", "vp")):
                m = np.full(body.size, np.nan); m[idx] = par[j]; mp[nm] = m.reshape(body.shape)
            mp["ktrans"] = mp["ke"] * mp["ve"]; maps[key] = mp
            rows[key] = {r: {nm: [float(np.nanpercentile(mp[nm][rois[r]], q)) for q in (50, 25, 75)] for nm in ("ktrans", "ve", "vp", "ke")} for r in (R1, R2) if r in rois and rois[r].sum() > 0}
            print(f"  {key:28s} {len(t):3d} fr, {idx.size} voxels, {time.time() - ts:.0f} s | {R1} ktrans {rows[key][R1]['ktrans'][0]:.3f} ve {rows[key][R1]['ve'][0]:.3f} vp {rows[key][R1]['vp'][0]:.3f} | {R2} ktrans {rows[key][R2]['ktrans'][0]:.3f} ve {rows[key][R2]['ve'][0]:.3f} vp {rows[key][R2]['vp'][0]:.3f}", flush=True)
        np.savez(f"{OUT}/pk_maps/pk_{a.tag}_sl{Z}.npz", body=body, **{f"{k}_{nm}": maps[k][nm] for k in maps for nm in maps[k]})
        json.dump(dict(rows=rows, aif=aif, TR=TR, FA=FA, hct=a.hct, r1=a.r1, T10=T10), open(f"{OUT}/pk_maps/pk_{a.tag}_sl{Z}.json", "w"), indent=1)
        md += [f"## slice {Z}", f"| method | {R1} ktrans | {R1} ve | {R1} vp | {R2} ktrans | {R2} ve | {R2} vp |", "|---|---|---|---|---|---|---|"]
        for key in maps: md.append(f"| {key} | " + " | ".join(f"{rows[key][r][nm][0]:.3f} [{rows[key][r][nm][1]:.2f}, {rows[key][r][nm][2]:.2f}]" for r in (R1, R2) for nm in ("ktrans", "ve", "vp")) + " |")
        md.append("")
        keys = list(maps); fig, ax = plt.subplots(3, len(keys), figsize=(2.9 * len(keys), 8.8), squeeze=False); anat = mf.mean(-1)
        for j, k in enumerate(keys):
            for i, (nm, vmax) in enumerate((("ktrans", 0.5), ("ve", 0.8), ("vp", 0.15))):
                im = np.where(enh60 if nm == "ve" else enh, maps[k][nm], np.nan); ax[i, j].imshow(anat, cmap="gray", vmin=0, vmax=np.percentile(anat[body], 99.5))
                ax[i, j].imshow(np.ma.masked_invalid(im), cmap="inferno" if nm == "ktrans" else ("viridis" if nm == "ve" else "magma"), vmin=0, vmax=vmax, alpha=0.9); ax[i, j].axis("off")
                if i == 0: ax[i, j].set_title(k, fontsize=10, fontweight="bold")
                if j == 0: ax[i, j].text(-0.06, 0.5, {"ktrans": "Ktrans (1/min)", "ve": "ve", "vp": "vp"}[nm], transform=ax[i, j].transAxes, rotation=90, va="center", fontsize=12)
        fig.suptitle(f"{dsp.DS} slice {Z}, {a.tag}: fitted extended-kety maps, no truth; literature T10, model-free aorta aif", fontsize=12)
        fig.tight_layout(); fig.savefig(f"{OUT}/figures/pk_maps_{a.tag}_sl{Z}.png", dpi=150, facecolor="white"); plt.close(fig); print(f"saved {OUT}/figures/pk_maps_{a.tag}_sl{Z}.png", flush=True)
    open(f"{OUT}/pk_maps/pk_{a.tag}.md", "w").write("\n".join(md)); print("\n".join(md)); print(f"PK_DONE ({time.time() - t0:.0f} s)")

if __name__ == "__main__": main()
