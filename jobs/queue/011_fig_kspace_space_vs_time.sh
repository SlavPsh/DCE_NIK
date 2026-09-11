#!/bin/bash
#SBATCH -J kspt
#SBATCH -p defq
#SBATCH -c 4
#SBATCH --mem 24G
#SBATCH -t 0:20:00
# slide figure: k-space varies fast along k (measured spoke) and slowly along t (k-centre sample vs time, cartesian k-points of the grasp pro series, singular values)
cd /net/beegfs/users/P101440/DCE_NIK
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
micromamba run -n torch29 python -u - <<'PY'
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = "/net/beegfs/users/P101440/grasp_pro_py/results_ref"; TA = 375.0
sl = np.load(f"{R}/slice_21.npz"); kr = np.asarray(sl["kdata_radial"]); sh = np.load(f"{R}/shared.npz"); vt = np.asarray(sh["view_time"]).ravel() * TA
nx, nv, nc = kr.shape; c = int(np.argmax(np.abs(kr).sum((0, 1)))); k = kr[:, :, c]; kx = np.linspace(-0.5, 0.5, nx)
img = np.abs(np.load("/net/beegfs/users/P101440/grasp_pro_py/results_spoke_cs/cs_slice21_f100.npy")).astype(np.float32); H, W, T = img.shape; tf = (np.arange(T) + 0.5) * TA / T
K = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))
OR, GR, BL = "#e67e22", "#455a64", "#0369a1"
fig, ax = plt.subplots(2, 2, figsize=(15, 10), gridspec_kw=dict(hspace=0.35, wspace=0.25))
# a: one measured spoke near the bolus peak, real part vs kx
i = int(np.argmin(np.abs(vt - 65.0))); s = k[:, i]
ax[0, 0].plot(kx, s.real / np.abs(s).max(), color=OR, lw=1.2); ax[0, 0].set_xlim(-0.5, 0.5); ax[0, 0].set_xlabel("k along one spoke  (kx / kmax)", fontsize=12); ax[0, 0].set_ylabel("Re k-space (normalized)", fontsize=12)
ax[0, 0].set_title("along space: one measured spoke, t = 65 s", fontsize=14, color=GR, loc="left"); ax[0, 0].grid(alpha=.3)
ins = ax[0, 0].inset_axes([0.6, 0.6, 0.38, 0.36]); m = (kx > 0.1) & (kx < 0.2); ins.plot(kx[m], s.real[m] / np.abs(s).max(), color=OR, lw=1.2); ins.set_title("zoom 0.1 to 0.2 kmax", fontsize=9); ins.tick_params(labelsize=7)
# b: the k-centre sample of every spoke vs time (measured, 1710 points)
dc = np.abs(k[nx // 2, :]); ax[0, 1].plot(vt, dc / dc.max(), ".", color=OR, ms=3); ax[0, 1].set_xlabel("time (s)", fontsize=12); ax[0, 1].set_ylabel("|k-space| at the k-centre (normalized)", fontsize=12)
ax[0, 1].set_title("along time: the k-centre sample of all 1710 spokes, measured", fontsize=14, color=GR, loc="left"); ax[0, 1].grid(alpha=.3); ax[0, 1].set_ylim(0, 1.05)
# c: off-centre cartesian k-points vs time (grasp pro series fft), normalized magnitude
pts = [(0, 0), (0, 12), (10, 30), (40, 0)]; cy, cx = H // 2, W // 2
for (dy, dx), col in zip(pts, [OR, BL, "#1e8449", "#8e44ad"]):
    q = np.abs(K[cy + dy, cx + dx, :]); r = np.hypot(dy, dx) / (W / 2); ax[1, 0].plot(tf, q / q.max(), "-", color=col, lw=2, label=f"|k| = {r:.2f} kmax")
ax[1, 0].set_xlabel("time (s)", fontsize=12); ax[1, 0].set_ylabel("|k-space| at a fixed k-point (normalized)", fontsize=12); ax[1, 0].legend(fontsize=10); ax[1, 0].grid(alpha=.3)
ax[1, 0].set_title("along time: fixed k-points of the reconstructed series (GRASP-Pro, 122 frames)", fontsize=13, color=GR, loc="left")
# d: singular values of the k-space x time matrix
M = K.reshape(-1, T); sv = np.linalg.svd(M, compute_uv=False); e = np.cumsum(sv**2) / np.sum(sv**2)
ax[1, 1].semilogy(np.arange(1, 41), sv[:40] / sv[0], "o-", color=OR, ms=4); ax[1, 1].set_xlabel("temporal component", fontsize=12); ax[1, 1].set_ylabel("singular value (normalized)", fontsize=12); ax[1, 1].grid(alpha=.3, which="both")
for r_ in (3, 8, 16): ax[1, 1].annotate(f"{r_} components: {100*e[r_-1]:.1f}% of the energy", xy=(r_, sv[r_-1] / sv[0]), xytext=(r_ + 3, sv[r_-1] / sv[0] * 2.5), fontsize=10, color=GR, arrowprops=dict(arrowstyle="->", color=GR))
ax[1, 1].set_title("time is low-rank: k-space x time matrix, singular values", fontsize=14, color=GR, loc="left")
fig.suptitle("in vivo slice 21: k-space varies fast along k and slowly along t", fontsize=16, fontweight="bold")
out = "results/realdata_nik_vs_cs_figures/figures/kspace_space_vs_time_sl21.png"; fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight"); print("saved", out, "coil", c, "energy 3/8/16:", [round(float(e[j-1]), 4) for j in (3, 8, 16)])
PY
echo "FIG exit $?"
