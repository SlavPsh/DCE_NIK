#!/bin/bash
#SBATCH -J kspt2
#SBATCH -p defq
#SBATCH -c 4
#SBATCH --mem 24G
#SBATCH -t 0:20:00
# slide figure: k-space fast along k (measured spoke), slow along t (measured k-centre, model-free k-points, singular values of the model-free k x t matrix; no temporal model in any source)
cd /net/beegfs/users/P101440/DCE_NIK
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
micromamba run -n torch29 python -u - <<'PY'
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = "/net/beegfs/users/P101440/grasp_pro_py/results_ref"; TA = 375.0
sl = np.load(f"{R}/slice_21.npz"); kr = np.asarray(sl["kdata_radial"]); sh = np.load(f"{R}/shared.npz"); vt = np.asarray(sh["view_time"]).ravel() * TA
nx, nv, nc = kr.shape; c = int(np.argmax(np.abs(kr).sum((0, 1)))); k = kr[:, :, c]; kx = np.linspace(-0.5, 0.5, nx)
md = np.load("step2_slice21.npz"); mf = md["mf"].astype(np.float32); tmf = np.asarray(md["tmf"]).ravel(); T, H, W = mf.shape       # model-free, 31-spoke window, no temporal model
K = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(mf, axes=(1, 2)), axes=(1, 2)), axes=(1, 2))                                     # [T,H,W]
OR, GR, BL = "#e67e22", "#455a64", "#0369a1"
fig, ax = plt.subplots(2, 2, figsize=(15, 10), gridspec_kw=dict(hspace=0.4, wspace=0.25))
i = int(np.argmin(np.abs(vt - 65.0))); s = k[:, i]; sn = s.real / np.abs(s).max()
ax[0, 0].plot(kx, sn, color=OR, lw=1.2); ax[0, 0].set_xlim(-0.5, 0.5); ax[0, 0].set_xlabel("k along one spoke  (kx / kmax)", fontsize=12); ax[0, 0].set_ylabel("Re k-space (normalized)", fontsize=12); ax[0, 0].grid(alpha=.3)
ax[0, 0].set_title("along k: one measured spoke, t = 65 s", fontsize=14, color=GR, loc="left")
ins = ax[0, 0].inset_axes([0.58, 0.55, 0.4, 0.33]); m = (kx > 0.1) & (kx < 0.2); ins.plot(kx[m], sn[m], color=OR, lw=1.2); ins.text(0.02, 0.85, "0.1 to 0.2 kmax", transform=ins.transAxes, fontsize=9); ins.tick_params(labelsize=7)
dc = np.abs(k[nx // 2 - 2: nx // 2 + 3, :]).mean(0); ax[0, 1].plot(vt, dc / dc.max(), ".", color=OR, ms=3); ax[0, 1].set_ylim(0, 1.05); ax[0, 1].grid(alpha=.3)
ax[0, 1].set_xlabel("time (s)", fontsize=12); ax[0, 1].set_ylabel("|k-space| at the k-centre (normalized)", fontsize=12); ax[0, 1].set_title("along t: the k-centre of all 1710 spokes, measured", fontsize=14, color=GR, loc="left")
cy, cx = H // 2, W // 2
for (dy, dx), col in zip([(0, 0), (0, 6), (8, 8), (0, 24)], [OR, BL, "#1e8449", "#8e44ad"]):
    q = np.abs(K[:, cy + dy, cx + dx]); ax[1, 0].plot(tmf, q / q.max(), "-", color=col, lw=1.8, label=f"|k| = {np.hypot(dy, dx) / (W / 2):.2f} kmax")
ax[1, 0].set_xlabel("time (s)", fontsize=12); ax[1, 0].set_ylabel("|k-space| at a fixed k-point (normalized)", fontsize=12); ax[1, 0].legend(fontsize=10, loc="lower right"); ax[1, 0].grid(alpha=.3)
ax[1, 0].set_title("along t: fixed k-points, model-free series (no temporal model)", fontsize=13, color=GR, loc="left")
sv = np.linalg.svd(K.reshape(T, -1), compute_uv=False); e = np.cumsum(sv**2) / np.sum(sv**2)
ax[1, 1].semilogy(np.arange(1, 41), sv[:40] / sv[0], "o-", color=OR, ms=4); ax[1, 1].set_xlabel("temporal component", fontsize=12); ax[1, 1].set_ylabel("singular value (normalized)", fontsize=12); ax[1, 1].grid(alpha=.3, which="both")
for r_ in (3, 8, 16): ax[1, 1].annotate(f"{r_} components: {100*e[r_-1]:.2f}% of the energy", xy=(r_, sv[r_-1] / sv[0]), xytext=(r_ + 4, sv[r_-1] / sv[0] * 3), fontsize=10, color=GR, arrowprops=dict(arrowstyle="->", color=GR))
ax[1, 1].set_title("along t: singular values of the k-space x time matrix (model-free)", fontsize=13, color=GR, loc="left")
fig.suptitle("in vivo slice 21: k-space varies fast along k and slowly along t", fontsize=16, fontweight="bold")
out = "results/realdata_nik_vs_cs_figures/figures/kspace_space_vs_time_sl21.png"; fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight"); print("saved", out, "coil", c, "energy 3/8/16:", [round(float(e[j-1]), 4) for j in (3, 8, 16)])
PY
echo "FIG exit $?"
