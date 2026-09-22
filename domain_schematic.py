"""minimal schematic: where the three nik models are real and where complex (re, im pair). no colors, no hyperparameters.
usage: python domain_schematic.py --out figures/domain_schematic.png"""
import argparse
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
W, H, GAP = 4.0, 0.72, 0.42

def box(ax, x, y, text):
    ax.add_patch(Rectangle((x, y), W, H, fc="white", ec="black", lw=1.0)); ax.text(x + W / 2, y + H / 2, text, ha="center", va="center", fontsize=9.5)

def column(ax, x, title, rows, top):
    ax.text(x + W / 2, top + 0.25, title, ha="center", va="bottom", fontsize=12, fontweight="bold"); y = top - H
    for i, text in enumerate(rows):
        box(ax, x, y, text)
        if i < len(rows) - 1: ax.annotate("", xy=(x + W / 2, y - GAP + 0.04), xytext=(x + W / 2, y - 0.02), arrowprops=dict(arrowstyle="-|>", lw=1.0, color="black"))
        y -= H + GAP

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="figures/domain_schematic.png"); a = ap.parse_args()
    fig, ax = plt.subplots(figsize=(14.5, 6.6)); ax.set_xlim(0, 14.5); ax.set_ylim(0.6, 6.8); ax.axis("off")
    top = 6.0
    column(ax, 0.5, "NIK-free", ["kx, ky, t, coil:  real", "network:  real", "prediction p:  (re, im)", "loss  |p - y|^2:  real"], top)
    column(ax, 5.25, "NIK-sub16", ["kx, ky, t, coil:  real", "network:  real", "coefficients a_r:  (re, im)\nbasis functions Phi_r(t):  (re, im), learned", "p = sum_r a_r Phi_r:  (re, im)", "loss  |p - y|^2:  real"], top)
    column(ax, 10.0, "NIK-tofts / patlak", ["kx, ky, t, coil:  real", "network:  real", "coefficients a_r:  (re, im)\nbasis functions Phi_r(t):  real, fixed", "p = sum_r a_r Phi_r:  (re, im)", "loss  |p - y|^2:  real"], top)
    fig.savefig(a.out, dpi=150, facecolor="white", bbox_inches="tight"); print("saved", a.out)

if __name__ == "__main__": main()
