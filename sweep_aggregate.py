"""Aggregate the factorized R-sweep: held-out (from logs) + swing/nav-corr/CS-corr (from
recons) for each rank, vs the full-rank baseline. Table + plots. out: sweep_R.png"""
import re, glob, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = "/net/beegfs/users/P101440/DCE_NIK"
GP = "/net/beegfs/users/P101440/grasp_pro_py/results_ref"
BASE = dict(held=0.3245, swing=45.5, nav=0.985)     # full-rank wire_ff_res

# held-out per rank from the sweep logs
held = {}
for lg in glob.glob(f"{D}/logs/slurm-nik-subR*-*.out"):
    txt = open(lg).read()
    mr = re.search(r"model=wire_ff_subspace rank(\d+)", txt)
    mh = re.search(r"restored best \(heldout ([\d.eE+-]+)\)", txt)
    if mr and mh:
        held[int(mr.group(1))] = float(mh.group(1))

# navigator + ROI for swing/nav-corr
import numpy as np
from figpath import fig as fpath
sl = np.load(f"{GP}/slice_13.npz"); sh = np.load(f"{GP}/shared.npz")
krad = np.asarray(sl["kdata_radial"]); vt = np.asarray(sh["view_time"]).ravel()
c0 = krad.shape[0] // 2
dc = np.sqrt((np.abs(krad[c0, :, :]) ** 2).sum(-1))
o = np.argsort(vt); ts = vt[o]; dcs = np.clip(dc[o], np.percentile(dc, 1), np.percentile(dc, 99))
nav = np.convolve(np.pad(dcs, (30, 30), mode="reflect"), np.ones(61)/61, mode="valid")[:len(dcs)]
nav = nav / np.median(nav[int(.35*len(nav)):])
cs = np.abs(np.asarray(sl["cs_img"]))
def n01(a): a = np.asarray(a, float); return (a-a.min())/(a.max()-a.min()+1e-12)

rows = []
for R in sorted(set(list(held) + [16])):
    f = f"{D}/results_nik_subspace_r{R}/nik_slice_13.npy"
    try:
        rec = np.abs(np.load(f)); nt = rec.shape[-1]
        roi = rec.mean(-1) > np.quantile(rec.mean(-1), 0.6)
        c = np.array([rec[..., i][roi].mean() for i in range(nt)])
        sw = (c.max()-c.min())/c.mean()*100
        tt = np.linspace(0, 1, nt)
        nc = float(np.corrcoef(n01(c), np.interp(tt, ts, n01(nav)))[0, 1])
        cscorr = float(np.corrcoef(rec.mean(-1).ravel(), cs.mean(-1).ravel())[0, 1]) if cs.ndim == 3 else np.nan
        rows.append((R, held.get(R, np.nan), sw, nc, cscorr))
    except FileNotFoundError:
        rows.append((R, held.get(R, np.nan), np.nan, np.nan, np.nan))

print(f"{'R':>4} {'held-out':>9} {'swing%':>7} {'nav-corr':>8} {'vsCS':>6}")
print(f"{'full':>4} {BASE['held']:9.4f} {BASE['swing']:7.1f} {BASE['nav']:+8.3f} {'0.993':>6}  (baseline)")
for R, h, sw, nc, cs2 in rows:
    print(f"{R:>4} {h:9.4f} {sw:7.1f} {nc:+8.3f} {cs2:6.3f}")

done = [(R, h, sw, nc) for R, h, sw, nc, _ in rows if np.isfinite(h) and np.isfinite(sw)]
if done:
    Rs = [d[0] for d in done]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for a, idx, lab, base in [(ax[0], 1, "held-out MSE", BASE['held']),
                              (ax[1], 2, "swing %", BASE['swing']),
                              (ax[2], 3, "nav-corr", BASE['nav'])]:
        a.plot(Rs, [d[idx] for d in done], "o-", lw=2, ms=7)
        a.axhline(base, color="tab:red", ls="--", lw=1.5, label="full-rank baseline")
        a.set_xlabel("rank R"); a.set_title(lab); a.grid(alpha=.3); a.legend(fontsize=9)
    fig.suptitle("Factorized low-rank NIK: R-sweep vs full-rank baseline (slice 13)", fontweight="bold")
    fig.tight_layout(); fig.savefig(fpath(f"sweep_R.png"), bbox_inches="tight", dpi=140)
    print(f"\nwrote {D}/sweep_R.png  ({len(done)} ranks)")
