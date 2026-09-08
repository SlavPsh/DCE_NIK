"""CV evidence figure: held-out k-space NMSE vs K picks K*=12 from the data alone (no truth), and that
is exactly where the aorta bolus appears. Contrast with variance selection (K=5, misses the bolus)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import xph_common as X
OUT = X.OUT; FIG = f"{OUT}/figures"
c = pd.read_csv(f"{OUT}/grasp_kcv.csv"); K = c["K"].values
Kstar = int(K[np.argmin(c["val_heldout_NMSE"].values)])
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
ax[0].semilogy(K, c["val_heldout_NMSE"], "-o", label="val (angle 5)")
ax[0].semilogy(K, c["test_heldout_NMSE"], "-s", label="test (angle 6)")
ax[0].axvline(5, color="grey", ls="--", lw=1); ax[0].axvline(Kstar, color="C2", ls="--", lw=1.5)
ax[0].text(5, ax[0].get_ylim()[1], " variance K=5", color="grey", fontsize=8, va="top", rotation=90)
ax[0].text(Kstar, ax[0].get_ylim()[1], f" CV K*={Kstar}", color="C2", fontsize=8, va="top", rotation=90)
ax[0].set_xlabel("subspace rank K"); ax[0].set_ylabel("held-out k-space NMSE (lower better)")
ax[0].set_title("data-driven K: held-out prediction, no truth"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.2)
ax[1].plot(K, c["aorta_peak_truth"], "-o", color="C3", label="grasp aorta peak (truth)")
ax[1].axhline(0.772, color="k", ls=":", lw=1); ax[1].text(K.max(), 0.772, " truth 0.77", fontsize=8, va="bottom", ha="right")
ax[1].axvline(5, color="grey", ls="--", lw=1); ax[1].axvline(Kstar, color="C2", ls="--", lw=1.5)
ax[1].set_xlabel("subspace rank K"); ax[1].set_ylabel("aorta first-pass peak")
ax[1].set_title(f"CV-chosen K*={Kstar} is where the bolus is captured"); ax[1].grid(alpha=0.2); ax[1].legend(fontsize=8)
fig.suptitle("held-out cross-validation selects the bolus-capturing K from the data (not the truth). z15 phantom")
fig.tight_layout(); fig.savefig(f"{FIG}/fig_kcv.png", dpi=130); plt.close(fig)
print(f"CV-selected K*={Kstar}; SAVED {FIG}/fig_kcv.png"); print("CVFIG_DONE")
