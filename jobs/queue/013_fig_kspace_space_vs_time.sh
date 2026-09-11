#!/bin/bash
#SBATCH -J kspt3
#SBATCH -p defq
#SBATCH -c 4
#SBATCH --mem 24G
#SBATCH -t 0:20:00
# slide figure v3: along k = measured spoke; along t = measured k-centre (running median), phantom truth k-points, singular values (phantom truth vs in vivo model-free)
cd /net/beegfs/users/P101440/DCE_NIK
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH XPH_SIM=nomotion
micromamba run -n torch29 python -u - <<'PY'
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import xph_pipeline as P, xph_common as X
R = "/net/beegfs/users/P101440/grasp_pro_py/results_ref"; TA = 375.0
sl = np.load(f"{R}/slice_21.npz"); kr = np.asarray(sl["kdata_radial"]); sh = np.load(f"{R}/shared.npz"); vt = np.asarray(sh["view_time"]).ravel() * TA
nx, nv, nc = kr.shape; c = int(np.argmax(np.abs(kr).sum((0, 1)))); k = kr[:, :, c]; kx = np.linspace(-0.5, 0.5, nx)
md = np.load("step2_slice21.npz"); mf = md["mf"].astype(np.float32); Kv = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(mf, axes=(1, 2)), axes=(1, 2)), axes=(1, 2)).reshape(mf.shape[0], -1)
d = P.data(); tq = np.asarray(d["times"]); Tr = X.truth_at(P.ZI, tq).astype(np.float32); H, W, T = Tr.shape
Kp = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(Tr, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))
OR, GR, BL = "#e67e22", "#455a64", "#0369a1"
fig, ax = plt.subplots(2, 2, figsize=(15, 10), gridspec_kw=dict(hspace=0.4, wspace=0.25))
i = int(np.argmin(np.abs(vt - 65.0))); s = k[:, i]; sn = s.real / np.abs(s).max()
ax[0, 0].plot(kx, sn, color=OR, lw=1.2); ax[0, 0].set_xlim(-0.5, 0.5); ax[0, 0].set_xlabel("k along one spoke  (kx / kmax)", fontsize=12); ax[0, 0].set_ylabel("Re k-space (normalized)", fontsize=12); ax[0, 0].grid(alpha=.3)
ax[0, 0].set_title("along k: one measured spoke, in vivo", fontsize=14, color=GR, loc="left")
ins = ax[0, 0].inset_axes([0.58, 0.55, 0.4, 0.33]); m = (kx > 0.1) & (kx < 0.2); ins.plot(kx[m], sn[m], color=OR, lw=1.2); ins.text(0.02, 0.85, "0.1 to 0.2 kmax", transform=ins.transAxes, fontsize=9); ins.tick_params(labelsize=7)
dc = np.abs(k[nx // 2 - 2: nx // 2 + 3, :]).mean(0); dc = dc / dc.max(); med = np.array([np.median(dc[max(0, j - 15): j + 16]) for j in range(nv)])
ax[0, 1].plot(vt, dc, ".", color="#f5cba7", ms=3, label="every spoke"); ax[0, 1].plot(vt, med, color=OR, lw=2.5, label="running median, 31 spokes"); ax[0, 1].set_ylim(0, 1.05); ax[0, 1].grid(alpha=.3); ax[0, 1].legend(fontsize=10, loc="lower right")
ax[0, 1].set_xlabel("time (s)", fontsize=12); ax[0, 1].set_ylabel("|k-space| at the k-centre (normalized)", fontsize=12); ax[0, 1].set_title("along t: the measured k-centre, all 1710 spokes, in vivo", fontsize=14, color=GR, loc="left")
cy, cx = H // 2, W // 2
for (dy, dx), col in zip([(0, 0), (0, 6), (8, 8), (0, 24)], [OR, BL, "#1e8449", "#8e44ad"]):
    q = np.abs(Kp[cy + dy, cx + dx, :]); ax[1, 0].plot(tq, q / q.max(), "-", color=col, lw=2, label=f"|k| = {np.hypot(dy, dx) / (W / 2):.2f} kmax")
ax[1, 0].set_xlabel("time (s)", fontsize=12); ax[1, 0].set_ylabel("|k-space| at a fixed k-point (normalized)", fontsize=12); ax[1, 0].legend(fontsize=10, loc="lower right"); ax[1, 0].grid(alpha=.3)
ax[1, 0].set_title("along t: fixed k-points, phantom truth (noise-free)", fontsize=14, color=GR, loc="left")
svp = np.linalg.svd(Kp.reshape(-1, T), compute_uv=False); ep = np.cumsum(svp**2) / np.sum(svp**2); svv = np.linalg.svd(Kv, compute_uv=False); ev = np.cumsum(svv**2) / np.sum(svv**2)
ax[1, 1].semilogy(np.arange(1, 41), svp[:40] / svp[0], "o-", color=OR, ms=4, label="phantom truth"); ax[1, 1].semilogy(np.arange(1, 41), svv[:40] / svv[0], "s-", color="#90a4ae", ms=3, label="in vivo, model-free (streak noise floor)")
ax[1, 1].set_xlabel("temporal component", fontsize=12); ax[1, 1].set_ylabel("singular value (normalized)", fontsize=12); ax[1, 1].grid(alpha=.3, which="both"); ax[1, 1].legend(fontsize=10)
ax[1, 1].text(0.97, 0.62, "phantom truth, energy captured:" + "".join(f"\n{r_} components  {100*ep[r_-1]:.3f}%" for r_ in (3, 8, 16)), transform=ax[1, 1].transAxes, ha="right", va="top", fontsize=11, color=GR, bbox=dict(fc="white", ec="#cfd8dc"))
ax[1, 1].set_title("along t: singular values of the k-space x time matrix", fontsize=14, color=GR, loc="left")
fig.suptitle("k-space varies fast along k and slowly along t", fontsize=16, fontweight="bold")
out = "results/realdata_nik_vs_cs_figures/figures/kspace_space_vs_time_sl21.png"; fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight"); print("saved", out, "phantom energy 3/8/16:", [round(float(ep[j-1]), 5) for j in (3, 8, 16)], "invivo:", [round(float(ev[j-1]), 4) for j in (3, 8, 16)])
PY
echo "FIG exit $?"
