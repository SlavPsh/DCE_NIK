"""image quality vs curve fidelity for the pk arms: the trade-off the 10k / unit-rms protocol moved along. scores every amp_track snapshot
(per training step) and the final recons (old and new pk arms, nik-free, nik-sub16, grasp, grasp-pro) against the grasp-pro all-spoke
anatomy (cs100, window-matched frames) with body-masked haarpsi / ssim / psnr, air energy (streaks + noise) and a temporal noise proxy in the
static roi; pairs them with the cortex / medulla amplitude ratios and curve nrmse vs model-free. no truth in vivo: cs100 is a sharpness
reference, model-free the dynamics reference.
out: results/tofts_vs_patlak/iq_track_sl<Z>.{md,json}, figures/iq_track_sl<Z>.png (metrics vs step, trade-off scatter), figures/iq_zoom_sl<Z>.png
usage: python tofts_iq_track.py --slice 21"""
import warnings; warnings.filterwarnings("ignore")
import os, re, sys, json, glob, argparse, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B = "/net/beegfs/users/P101440/DCE_NIK"; GV = "/net/beegfs/users/P101440/grasp_v2/results_grasp_v2"; GP = "/net/beegfs/users/P101440/grasp_pro_py/results_spoke_cs"; sys.path.insert(0, B)
import consolidated as C
from story_figs import ls_scale
TA = 375.0

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--slice", type=int, default=21); a = ap.parse_args(); Z = a.slice
    ctx = C.slice_ctx(Z); rois = ctx["rois"]; body = ctx["BODY"]; air = ctx["AIR"]; cs = ctx["cs100"]; tcs = np.linspace(0, TA, cs.shape[-1])
    z = np.load(f"{B}/step2_slice{Z}.npz"); mf = np.abs(z["mf"]).transpose(1, 2, 0).astype(np.float32); tmf = np.asarray(z["tmf"], float)
    def ft(nt): e = np.linspace(0, TA, nt + 1); return 0.5 * (e[:-1] + e[1:])
    def refwin(nt):
        e = np.linspace(0, TA, nt + 1); o = np.zeros(mf.shape[:2] + (nt,), np.float32)
        for g in range(nt):
            m = (tmf >= e[g]) & (tmf < e[g + 1])
            if not m.any(): m = np.zeros_like(tmf, bool); m[np.argmin(np.abs(tmf - 0.5 * (e[g] + e[g + 1])))] = True
            o[:, :, g] = mf[:, :, m].mean(2)
        return o
    RW = {}
    def frame(v, t, ts, w=10): m = (t > ts - w) & (t < ts + w); return v[..., m].mean(2) if m.any() else v[..., int(np.argmin(np.abs(t - ts)))]
    static = rois.get("liver", body)
    def enh(c, t): return c - np.median(c[t < 40])
    def amp(c, t): e = enh(c, t); return float(e[(t > 20) & (t < 210)].max()), float(e[t > 200].mean())
    mfc = {r: np.array([mf[..., i][rois[r]].mean() for i in range(mf.shape[-1])]) for r in ("cortex", "medulla")}; ref_amp = {r: amp(mfc[r], tmf) for r in mfc}
    def iq(v, t, label):
        nt = v.shape[-1]
        if nt not in RW: RW[nt] = refwin(nt)
        v = ls_scale(v, RW[nt], body); out = dict(label=label)
        for ts in (90.0, 300.0):
            p = frame(v, t, ts); r = frame(cs, tcs, ts); s = C.score(p, r, body); out.update({f"haarpsi_{int(ts)}": s["haarpsi"], f"ssim_{int(ts)}": s["ssim"], f"psnr_{int(ts)}": s["psnr"], f"airE_{int(ts)}": C.bgE(p, r, body, air)})
        cst = np.array([v[..., i][static].mean() for i in range(nt)]); hp = cst - np.convolve(cst, np.ones(9) / 9, mode="same"); out["static_noise"] = float(np.std(hp[5:-5]) / (np.abs(cst).mean() + 1e-12))
        for r in ("cortex", "medulla"):
            c = np.array([v[..., i][rois[r]].mean() for i in range(nt)]); pk, wo = amp(c, t); out[f"{r}_peak"] = pk / (ref_amp[r][0] + 1e-12); out[f"{r}_washout"] = wo / (ref_amp[r][1] + 1e-12)
            ci = np.interp(tmf, t, c); out[f"{r}_nrmse"] = float(np.linalg.norm(enh(ci, tmf) - enh(mfc[r], tmf)) / (np.linalg.norm(enh(mfc[r], tmf)) + 1e-12))
        return out
    rows = {"final": [], "steps": {}}
    fin = [("model-free", mf, tmf), ("cs100 (GRASP-Pro all spokes)", cs, tcs), ("NIK-free", f"{B}/results_sl21_k80/nik_slice_{Z}.npy" if Z == 21 else f"{B}/results_sl{Z}_k80/nik_slice_{Z}.npy", None),
           ("NIK-sub16", f"{B}/results/realdata_nik_vs_cs_figures/outcoil_subspace_output_slice{Z}.npy", None),
           ("NIK-tofts8 old (3k, restore)", f"{B}/results/tofts_vs_patlak/invivo_k80/tofts8_sl{Z}_s0/nik_slice_{Z}_cplx.npy", None), ("NIK-tofts8 new (10k, rms1)", f"{B}/results/tofts_vs_patlak/invivo_k80_rms1/tofts8_sl{Z}_s0/nik_slice_{Z}_cplx.npy", None),
           ("NIK-tofts new (10k, rms1)", f"{B}/results/tofts_vs_patlak/invivo_k80_rms1/tofts_sl{Z}_s0/nik_slice_{Z}_cplx.npy", None), ("NIK-patlak new (10k)", f"{B}/results/tofts_vs_patlak/invivo_k80_rms1/patlak_sl{Z}_s0/nik_slice_{Z}_cplx.npy", None),
           ("GRASP-Pro", f"{GP}/cs_slice{Z}_f80match.npy", None), ("GRASP", f"{GV}/gv2_slice{Z}_n12_k80.npy", None)]
    for spec in [x for x in os.environ.get("IQ_EXTRA", "").split(",") if x]: fin.append((spec.split(":")[0], spec.split(":")[1], None))   # IQ_EXTRA name:path,...
    ims = {}
    for nm, v, t in fin:
        if isinstance(v, str):
            if not os.path.exists(v): print("missing", v); continue
            v = np.abs(np.load(v)).astype(np.float32)
        t = t if t is not None else ft(v.shape[-1]); r = iq(v, t, nm); rows["final"].append(r); ims[nm] = (frame(ls_scale(v, RW.get(v.shape[-1], refwin(v.shape[-1])), body), t, 90.0), frame(ls_scale(v, RW.get(v.shape[-1], refwin(v.shape[-1])), body), t, 300.0)); print(nm, {k: round(x, 3) for k, x in r.items() if k != "label"}, flush=True)
    for var in os.environ.get("IQ_VARIANTS", "base,wd0,atomscale,env0,dcf1,lr1e-4").split(","):                 # IQ_VARIANTS: amp_track dirs to score per step; IQ_TAG: output suffix
        d = f"{B}/results/tofts_vs_patlak/amp_track/{var}_sl{Z}"; rows["steps"][var] = {}
        for p in sorted(glob.glob(f"{d}/snap_slice_{Z:02d}_step*.npy")):
            step = int(re.search(r"step(\d+)", p).group(1)); v = np.load(p).astype(np.float32); rows["steps"][var][step] = iq(v, ft(v.shape[-1]), f"{var} {step}")
        if rows["steps"][var]: print(var, "steps", sorted(rows["steps"][var]), flush=True)
    R = f"{B}/results/tofts_vs_patlak"; TG = os.environ.get("IQ_TAG", ""); json.dump(rows, open(f"{R}/iq_track_sl{Z}{TG}.json", "w"), indent=1)
    K = ["haarpsi_90", "ssim_90", "psnr_90", "airE_90", "haarpsi_300", "airE_300", "static_noise", "cortex_peak", "cortex_washout", "cortex_nrmse", "medulla_nrmse"]
    L = [f"# image quality vs curve fidelity, slice {Z}, k80 (image metrics vs cs100 = grasp-pro all spokes, body-masked, window-matched frames at 90 and 300 s; airE = rms in air / rms in body; static_noise = high-pass temporal std in the static roi; curves vs model-free)", "",
         "| recon | " + " | ".join(K) + " |", "|---|" + "---|" * len(K)]
    for r in rows["final"]: L.append(f"| {r['label']} | " + " | ".join(f"{r[k]:.3f}" for k in K) + " |")
    for var, st in rows["steps"].items():
        if not st: continue
        L += ["", f"## {var}, per training step", "| step | " + " | ".join(K) + " |", "|---|" + "---|" * len(K)]
        for s in sorted(st): L.append(f"| {s} | " + " | ".join(f"{st[s][k]:.3f}" for k in K) + " |")
    open(f"{R}/iq_track_sl{Z}{TG}.md", "w").write("\n".join(L)); print("\n".join(L[:16]))
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.4))
    for var, st in rows["steps"].items():
        if not st: continue
        s = sorted(st); ax[0].plot(s, [st[k]["haarpsi_90"] for k in s], "-o", ms=3, label=var); ax[1].plot(s, [st[k]["airE_90"] for k in s], "-o", ms=3, label=var); ax[2].plot(s, [st[k]["static_noise"] for k in s], "-o", ms=3, label=var)
        ax[3].plot([st[k]["cortex_nrmse"] for k in s], [st[k]["haarpsi_90"] for k in s], "-", lw=0.8, alpha=0.6); ax[3].scatter([st[k]["cortex_nrmse"] for k in s], [st[k]["haarpsi_90"] for k in s], c=s, cmap="viridis", s=14, label=var)
    for r in rows["final"]:
        if r["label"] in ("model-free", "cs100 (GRASP-Pro all spokes)"): continue
        ax[3].scatter(r["cortex_nrmse"], r["haarpsi_90"], marker="*", s=120, edgecolor="k", label=r["label"], zorder=5)
    for k, ttl in ((0, "HaarPSI at 90 s vs cs100"), (1, "air energy at 90 s (streaks + noise)"), (2, "temporal noise in the static roi")): ax[k].set_title(ttl, fontsize=10); ax[k].set_xlabel("step"); ax[k].legend(fontsize=7)
    ax[3].set_xlabel("cortex curve NRMSE vs model-free (lower = better dynamics)"); ax[3].set_ylabel("HaarPSI at 90 s (higher = sharper)"); ax[3].set_title("trade-off: snapshots coloured by step, stars = final recons", fontsize=10); ax[3].legend(fontsize=6, loc="lower left")
    fig.suptitle(f"slice {Z}: image quality vs curve fidelity along training and across methods", fontsize=11); fig.tight_layout(); fig.savefig(f"{R}/figures/iq_track_sl{Z}{TG}.png", dpi=130, facecolor="white")
    names = [n for n in ("cs100 (GRASP-Pro all spokes)", "GRASP", "GRASP-Pro", "NIK-tofts8 old (3k, restore)", "NIK-tofts8 new (10k, rms1)", "NIK-free", "NIK-sub16") if n in ims]
    c = np.argwhere(rois["cortex"]).mean(0).astype(int); h = 46
    fig, ax = plt.subplots(2, len(names), figsize=(3.1 * len(names), 6.4))
    for j, n in enumerate(names):
        for i in range(2):
            im = ims[n][i]; sl = (slice(max(c[0] - h, 0), c[0] + h), slice(max(c[1] - 2 * h, 0), c[1] + 2 * h)); ax[i, j].imshow(im[sl], cmap="gray", vmin=0, vmax=np.percentile(im[body], 99.5)); ax[i, j].axis("off")
            if i == 0: ax[i, j].set_title(n, fontsize=8)
    ax[0, 0].set_ylabel("t = 90 s"); ax[1, 0].set_ylabel("t = 300 s"); fig.suptitle(f"slice {Z}: kidney zoom at 90 s (top) and 300 s (bottom), one global scale, same k80 input", fontsize=10); fig.tight_layout()
    fig.savefig(f"{R}/figures/iq_zoom_sl{Z}{TG}.png", dpi=140, facecolor="white"); print("IQ_TRACK_DONE")

if __name__ == "__main__": main()
