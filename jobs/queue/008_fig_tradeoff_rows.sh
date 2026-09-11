#!/bin/bash
#SBATCH -J tradeoff
#SBATCH -p defq
#SBATCH -c 4
#SBATCH --mem 16G
#SBATCH -t 0:20:00
# slide figure: grasp v2 at 100 spokes/frame (sharp image, bolus lost) vs 10 spokes/frame (noisy image, bolus kept); peak-frame image + aorta curve vs truth per row
cd /net/beegfs/users/P101440/DCE_NIK
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH XPH_SIM=nomotion
micromamba run -n torch29 python -u - <<'PY'
import os, json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import xph_pipeline as P, xph_common as X
d = P.data(); tq = np.asarray(d["times"]); body = d["labels"] > 0; Tr = X.truth_at(P.ZI, tq); Rz = X.rois(P.ZI, d["labels"]); ao = Rz["aorta"]
tru = np.array([Tr[..., i][ao].mean() for i in range(Tr.shape[-1])]); tru = tru - np.median(tru[:8])
rows = [(20, 100, "sharp image, bolus averaged away"), (2, 10, "bolus timing kept, noisy image")]
fig, axs = plt.subplots(2, 2, figsize=(13, 9.2), gridspec_kw=dict(width_ratios=[1, 1.35], hspace=0.28, wspace=0.12))
OR = "#e67e22"; vmax = float(np.percentile(Tr[body], 99.5)); ipk = int(np.argmin(np.abs(tq - 27.4)))
for r, (G, spf, msg) in enumerate(rows):
    v = np.abs(np.load(f"{P.OUT}/v2_sweep/v2_G{G:02d}.npy")).astype(np.float32); T = v.shape[-1]; tw = np.array([tq[g*G*5:(g+1)*G*5].mean() for g in range(T)]) if False else np.array([tq[i*len(tq)//T:(i+1)*len(tq)//T].mean() for i in range(T)])
    v = v * (np.sum(v[body] * np.stack([Tr[..., i*len(tq)//T:(i+1)*len(tq)//T].mean(-1) for i in range(T)], -1)[body]) / (np.sum(v[body]**2) + 1e-12))
    m = json.load(open(f"{P.OUT}/v2_sweep/v2_G{G:02d}.json")); c = np.array([v[..., i][ao].mean() for i in range(T)]); c = c - np.median(c[:max(1, T//40)])
    i = int(np.argmin(np.abs(tw - 27.4)))
    ax = axs[r, 0]; ax.imshow(v[:, :, i], cmap="gray", vmin=0, vmax=vmax); ax.axis("off")
    ax.set_title(f"{spf} spokes per frame, {m['dt_s']:.1f} s per frame", fontsize=14, fontweight="bold", color=OR, loc="left")
    ax.text(0.02, 0.03, f"HaarPSI {m['haarpsi']:.2f}", transform=ax.transAxes, color="w", fontsize=12)
    ax = axs[r, 1]; ax.plot(tq, tru, "k", lw=2.5, label="truth"); ax.plot(tw, c, "-o", color=OR, lw=2.2, ms=5, label="GRASP"); ax.set_xlim(0, 120)
    ax.set_title(msg, fontsize=14, color="#455a64", loc="left"); ax.set_xlabel("time (s)", fontsize=12); ax.set_ylabel("aorta enhancement", fontsize=12); ax.grid(alpha=.3)
    ax.text(0.98, 0.9, f"bolus peak {100*m['aorta_pk']/m['aorta_pk_truth']:.0f}% of truth", transform=ax.transAxes, ha="right", fontsize=12, color=OR)
    if r == 0: ax.legend(fontsize=11, loc="center right")
fig.suptitle("GRASP: spatial vs temporal resolution (XCAT phantom, truth known)", fontsize=16, fontweight="bold")
out = f"{P.OUT}/figures/tradeoff_rows_grasp_v2.png"; fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight"); print("saved", out)
PY
echo "FIG exit $?"
