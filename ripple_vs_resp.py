"""is the sub16 curve ripple respiration? for every recon: roi curves on the recon's own frame grid; (a) high-pass (9-frame running mean removed)
correlated with the k-centre respiratory navigator (build_rulers.py, per spoke, averaged into the recon's frames) and its coherence at the
breathing peak; (b) curve nrmse vs model-free before and after smoothing the recon curve with the reference's own window (31 spokes = 6.8 s on
p3); (c) a gated model-free reference curve: 31-spoke sliding-window nufft restricted to end-expiration spokes is too sparse, so instead the
model-free series itself is high-passed and correlated with the navigator as the control (it should NOT correlate: 6.8 s windows average breathing).
usage: DCE_DS=p3 python ripple_vs_resp.py --slice 21 --items "label:path,..." --tag x"""
import warnings; warnings.filterwarnings("ignore")
import sys, os, json, argparse, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B = "/net/beegfs/users/P101440/DCE_NIK"; sys.path.insert(0, B)
import dsp, consolidated as C

def hp(c, k=9): return c - np.convolve(c, np.ones(k) / k, mode="same")
def corr(a, b): a = a - a.mean(); b = b - b.mean(); return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--slice", type=int, required=True); ap.add_argument("--items", required=True); ap.add_argument("--tag", default=""); a = ap.parse_args(); Z = a.slice; TA = dsp.TA
    ctx = C.slice_ctx(Z); rois = ctx["rois"]; names = [r for r in dsp.ROI_NAMES if r in rois]
    z = np.load(dsp.STEP2(Z)); mf = np.abs(z["mf"]).transpose(1, 2, 0).astype(np.float32); tmf = np.asarray(z["tmf"], float); nspk = 31; win_s = nspk * TA / dsp.NTV
    nav = np.load(f"{dsp.NUF(Z)}/resp_nav.npy").astype(np.float64); ts = np.asarray(np.load(f"{dsp.REF}/shared.npz")["view_time"]).ravel() * TA; meta = json.load(open(f"{dsp.NUF(Z)}/meta.json"))
    def enh(c, t): return c - np.median(c[t < 40])
    mfc = {r: np.array([mf[..., i][rois[r]].mean() for i in range(mf.shape[-1])]) for r in names}
    def nav_on(t):                                                                  # navigator averaged into the frames of a grid with spacing dt
        dt = t[1] - t[0]; return np.array([nav[(ts >= tt - dt / 2) & (ts < tt + dt / 2)].mean() if ((ts >= tt - dt / 2) & (ts < tt + dt / 2)).any() else 0.0 for tt in t])
    def smooth_to_ref(c, t):                                                        # same window as the 31-spoke reference, on the recon grid
        dt = t[1] - t[0]; k = max(1, int(round(win_s / dt)) | 1); return np.convolve(np.pad(c, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")
    rows = []; navm = nav_on(tmf)
    rows.append(dict(label="model-free 31-spoke (control)", **{f"{r}_corr_nav": corr(hp(mfc[r])[5:-5], hp(navm)[5:-5]) for r in names}, **{f"{r}_ripple": float(np.std(hp(mfc[r])[5:-5]) / (np.abs(mfc[r]).mean() + 1e-12)) for r in names}))
    fig, ax = plt.subplots(len(names), 1, figsize=(14, 2.6 * len(names)), sharex=True)
    for spec in [x for x in a.items.split(",") if x]:
        lab, path = spec.split(":", 1)
        if not os.path.exists(path): print("missing", path); continue
        v = np.abs(np.load(path)).astype(np.float32); t = (np.arange(v.shape[-1]) + 0.5) * TA / v.shape[-1]; nv = nav_on(t); row = dict(label=lab, frames=int(v.shape[-1]))
        for r in names:
            c = np.array([v[..., i][rois[r]].mean() for i in range(v.shape[-1])]); h = hp(c); row[f"{r}_ripple"] = float(np.std(h[5:-5]) / (np.abs(c).mean() + 1e-12))
            row[f"{r}_corr_nav"] = corr(h[5:-5], hp(nv)[5:-5])
            ci = np.interp(tmf, t, c); cs = np.interp(tmf, t, smooth_to_ref(c, t)); ref = enh(mfc[r], tmf)
            def nrmse(x):                                                           # affine ruler as in tofts_eval_invivo: recon units differ from the reference
                X = np.column_stack([x, np.ones_like(x)]); b, *_ = np.linalg.lstsq(X, ref, rcond=None); return float(np.linalg.norm(X @ b - ref) / (np.linalg.norm(ref) + 1e-12))
            row[f"{r}_nrmse"] = nrmse(enh(ci, tmf)); row[f"{r}_nrmse_smoothed"] = nrmse(enh(cs, tmf)); row[f"{r}_ripple_smoothed"] = float(np.std(hp(np.interp(t, tmf, cs))[5:-5]) / (np.abs(c).mean() + 1e-12))
            if r == names[1] if len(names) > 1 else r == names[0]:
                k = names.index(r); ax[k].plot(t, h / (np.abs(c).mean() + 1e-12), lw=0.7, label=f"{lab} ({r} high-pass)")
        rows.append(row); print(row, flush=True)
    for k, r in enumerate(names):
        nvv = hp(navm); ax[k].plot(tmf, 0.05 * nvv / (np.abs(nvv).max() + 1e-12), color="k", lw=0.6, alpha=0.6, label="navigator (scaled)"); ax[k].set_xlim(100, 160); ax[k].set_ylabel(r); ax[k].legend(fontsize=6, ncol=3)
    ax[-1].set_xlabel("t [s] (60 s excerpt)"); fig.suptitle(f"{dsp.DS} slice {Z}: high-passed roi curves vs the respiratory navigator (resp {meta.get('resp_freq_hz', 0):.2f} Hz)", fontsize=10); fig.tight_layout()
    K = ["frames"] + [f"{r}_ripple" for r in names] + [f"{r}_ripple_smoothed" for r in names] + [f"{r}_corr_nav" for r in names] + [f"{r}_nrmse" for r in names] + [f"{r}_nrmse_smoothed" for r in names]
    L = [f"# ripple vs respiration, {dsp.DS} slice {Z}: ripple = high-pass temporal std / mean; corr_nav = correlation of the high-passed curve with the k-centre respiratory navigator; nrmse vs model-free before / after smoothing the recon curve with the reference window ({win_s:.1f} s)", "",
         "| recon | " + " | ".join(K) + " |", "|---|" + "---|" * len(K)]
    for r in rows: L.append(f"| {r['label']} | " + " | ".join((f"{r[k]:.3f}" if isinstance(r.get(k), float) else str(r[k])) if k in r else "-" for k in K) + " |")
    out = f"{B}/results/tofts_vs_patlak/ripple_vs_resp{dsp.SFX}_sl{Z}{a.tag}"; open(out + ".md", "w").write("\n".join(L) + "\n"); json.dump(rows, open(out + ".json", "w"), indent=1); fig.savefig(out + ".png", dpi=120, facecolor="white")
    print("\n".join(L)); print("RIPPLE_DONE")

if __name__ == "__main__": main()
