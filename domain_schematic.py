"""minimal schematic: where the three nik models are real, complex (re, im pair) and magnitude. no colors, no hyperparameters.
usage: python domain_schematic.py --out figures/domain_schematic.png"""
import argparse
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
W, H, GAP = 4.0, 0.72, 0.42

def box(ax, x, y, text, dashed=False):
    ax.add_patch(Rectangle((x, y), W, H, fc="white", ec="black", lw=1.0, ls="--" if dashed else "-")); ax.text(x + W / 2, y + H / 2, text, ha="center", va="center", fontsize=9.5)

def column(ax, x, title, rows, top):
    ax.text(x + W / 2, top + 0.25, title, ha="center", va="bottom", fontsize=12, fontweight="bold"); y = top - H
    for i, (text, dashed) in enumerate(rows):
        box(ax, x, y, text, dashed)
        if i < len(rows) - 1: ax.annotate("", xy=(x + W / 2, y - GAP + 0.04), xytext=(x + W / 2, y - 0.02), arrowprops=dict(arrowstyle="-|>", lw=1.0, color="black"))
        y -= H + GAP
    return y + H + GAP

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="figures/domain_schematic.png"); a = ap.parse_args()
    fig, ax = plt.subplots(figsize=(14.5, 9.2)); ax.set_xlim(0, 14.5); ax.set_ylim(0, 9.6); ax.axis("off")
    top = 8.6
    column(ax, 0.5, "NIK-free", [("kx, ky, t, coil:  real", False), ("network:  real", False), ("prediction p:  (re, im)", False), ("loss  |p - y|^2:  real", False)], top)
    column(ax, 5.25, "NIK-sub16", [("kx, ky, t, coil:  real", False), ("network:  real", False), ("coefficients a_r:  (re, im)\natoms Phi_r(t):  (re, im), learned", False), ("p = sum_r a_r Phi_r:  (re, im)", False), ("loss  |p - y|^2:  real", False)], top)
    yl = column(ax, 10.0, "NIK-tofts / patlak", [("kx, ky, t, coil:  real", False), ("network:  real", False), ("coefficients a_r:  (re, im)\natoms Phi_r(t):  real, fixed", False), ("p = sum_r a_r Phi_r:  (re, im)", False), ("loss  |p - y|^2:  real", False)], top)
    # priors and rendering, shared
    y = yl - 1.55; ax.add_patch(Rectangle((5.25, y), 8.75, H, fc="white", ec="black", lw=1.0, ls="--")); ax.text(5.25 + 8.75 / 2, y + H / 2, "priors (sub16, tofts, patlak):  a_r -> ifft2 (complex) -> |.|^2 -> real penalty", ha="center", va="center", fontsize=9.5)
    for xc in (5.25 + W / 2, 10.0 + W / 2): ax.annotate("", xy=(xc, y + H), xytext=(xc, yl - 0.02), arrowprops=dict(arrowstyle="-|>", lw=1.0, color="black", linestyle="--"))
    y2 = y - 1.15; ax.add_patch(Rectangle((0.5, y2), 13.5, H, fc="white", ec="black", lw=1.0)); ax.text(0.5 + 13.5 / 2, y2 + H / 2, "after training:  k-space (re, im) -> ifft2 -> coil combine (complex) -> |.| -> magnitude frames, curves, metrics.  no gradient here.", ha="center", va="center", fontsize=9.5)
    ax.text(0.5, y2 - 0.5, "complex values are stored as (re, im) pairs; every loss is a real scalar of real tensors, so autograd never needs a complex derivative.\nthe priors are the only native complex tensors with gradient; their penalty is real (|.|^2), differentiated by the Wirtinger rule.", fontsize=9.5, va="top")
    fig.savefig(a.out, dpi=150, facecolor="white", bbox_inches="tight"); print("saved", a.out)

if __name__ == "__main__": main()
