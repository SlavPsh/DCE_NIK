"""amplitude vs training step for the pk-basis arms: roi enhancement of the rendered snapshots (train_grasp_nik --snapshot-every) relative to
the model-free 31-spoke reference, per variant, plus the held-out mse parsed from train.log. tells whether the deficit is early stopping
(ratio keeps rising with steps), regularization (ratio saturates below 1, wd 0 differs) or neither.
out: results/tofts_vs_patlak/amp_track_sl<Z>.{md,json}, figures/amp_track_sl<Z>.png
usage: python tofts_amp_track.py --slice 21 --runs base:<dir>,wd0:<dir>,lr1e-4:<dir>"""
import warnings; warnings.filterwarnings("ignore")
import os, re, sys, json, glob, argparse, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B = "/net/beegfs/users/P101440/DCE_NIK"; sys.path.insert(0, B)
import dsp                                                                                   # dataset paths (DCE_DS=p3 default / p8 / p14)
import consolidated as C
from story_figs import ls_scale

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--slice", type=int, default=21); ap.add_argument("--runs", required=True); a = ap.parse_args(); Z = a.slice
    ctx = C.slice_ctx(Z); rois = ctx["rois"]; body = ctx["BODY"]; z = np.load(f"{B}/step2_slice{Z}.npz"); mf = np.abs(z["mf"]).transpose(1, 2, 0).astype(np.float32); tmf = np.asarray(z["tmf"], float); TA = dsp.TA
    names = [r for r in ("aorta", "cortex", "medulla", "liver") if r in rois]
    def ft(nt): e = np.linspace(0, TA, nt + 1); return 0.5 * (e[:-1] + e[1:])
    def refwin(nt):
        e = np.linspace(0, TA, nt + 1); o = np.zeros(mf.shape[:2] + (nt,), np.float32)
        for g in range(nt):
            m = (tmf >= e[g]) & (tmf < e[g + 1])
            if not m.any(): m = np.zeros_like(tmf, bool); m[np.argmin(np.abs(tmf - 0.5 * (e[g] + e[g + 1])))] = True
            o[:, :, g] = mf[:, :, m].mean(2)
        return o
    def roi_curve(v): return {r: np.array([v[..., i][rois[r]].mean() for i in range(v.shape[-1])]) for r in names}
    def amp(c, t): e = c - np.median(c[t < 40]); return dict(peak=float(e[(t > 20) & (t < 210)].max()), washout=float(e[t > 200].mean()))
    ref = {r: amp(c, tmf) for r, c in roi_curve(mf).items()}
    out = {}; RW = {}
    for spec in a.runs.split(","):
        nm, d = spec.split(":", 1); rows = {}
        for p in sorted(glob.glob(f"{d}/snap_slice_{Z:02d}_step*.npy")):
            step = int(re.search(r"step(\d+)", p).group(1)); v = np.load(p).astype(np.float32); nt = v.shape[-1]; t = ft(nt)
            if nt not in RW: RW[nt] = refwin(nt)
            v = ls_scale(v, RW[nt], body); c = roi_curve(v)
            rows[step] = {r: dict(peak=amp(c[r], t)["peak"] / (ref[r]["peak"] + 1e-9), washout=amp(c[r], t)["washout"] / (ref[r]["washout"] + 1e-9)) for r in names}
        hl = {}
        if os.path.exists(f"{d}/train.log"):
            for m in re.finditer(r"step\s+(\d+)\s+train\s+([\d.e+-]+)\s+heldout\s+([\d.e+-]+)", open(f"{d}/train.log").read()): hl[int(m.group(1))] = dict(train=float(m.group(2)), heldout=float(m.group(3)))
        out[nm] = dict(dir=d, steps={str(s): dict(rows[s], **hl.get(s, {})) for s in sorted(rows)}, heldout={str(s): v for s, v in sorted(hl.items())})
        print(nm, "snapshots", sorted(rows), flush=True)
    L = [f"# amplitude vs step, slice {Z}, k80, tofts8 (roi enhancement relative to the model-free reference; peak 20 to 210 s / washout mean t > 200 s)", ""]
    for nm, o in out.items():
        L += [f"## {nm} ({o['dir']})", "| step | " + " | ".join(f"{r} peak / washout" for r in names) + " | heldout mse |", "|---|" + "---|" * (len(names) + 1)]
        for s, row in o["steps"].items(): L.append(f"| {s} | " + " | ".join(f"{row[r]['peak']:.2f} / {row[r]['washout']:.2f}" for r in names) + f" | {row.get('heldout', float('nan')):.3e} |")
        L.append("")
    R = f"{B}/results/tofts_vs_patlak"; open(f"{R}/amp_track_sl{Z}.md", "w").write("\n".join(L)); json.dump(out, open(f"{R}/amp_track_sl{Z}.json", "w"), indent=1); print("\n".join(L))
    fig, ax = plt.subplots(1, len(names) + 1, figsize=(4.4 * (len(names) + 1), 3.8))
    for j, r in enumerate(names):
        for nm, o in out.items():
            st = [int(s) for s in o["steps"]]; ax[j].plot(st, [o["steps"][str(s)][r]["peak"] for s in st], "-o", ms=3, label=f"{nm} peak"); ax[j].plot(st, [o["steps"][str(s)][r]["washout"] for s in st], "--s", ms=3, label=f"{nm} washout")
        ax[j].axhline(1, color="0.6", lw=0.8); ax[j].set_title(r); ax[j].set_xlabel("step"); ax[j].set_ylim(0, 2)
        if j == 0: ax[j].set_ylabel("amplitude / model-free"); ax[j].legend(fontsize=6)
    for nm, o in out.items():
        st = [int(s) for s in o["heldout"]]
        if st: ax[-1].semilogy(st, [o["heldout"][str(s)]["heldout"] for s in st], "-", label=nm)
    ax[-1].set_title("held-out mse (val k80 m8)"); ax[-1].set_xlabel("step"); ax[-1].legend(fontsize=7)
    fig.suptitle(f"slice {Z}: pk-arm enhancement amplitude vs training step, no restore", fontsize=11); fig.tight_layout()
    fig.savefig(f"{R}/figures/amp_track_sl{Z}.png", dpi=140, facecolor="white"); print("AMP_TRACK_DONE")

if __name__ == "__main__": main()
