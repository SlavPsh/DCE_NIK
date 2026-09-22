"""schematic of the real / complex domains along the three nik models (free, sub16, tofts): what is real, what is complex-as-two-real-channels,
where the one real scalar (the loss) sits, and why every parameter gradient is the gradient of a real scalar (no holomorphic derivative anywhere).
usage: python domain_schematic.py --out figures/domain_schematic.png (runs on the laptop, no data)"""
import argparse
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
REAL, PAIR, SCALAR, FIXED = "#dbeafe", "#fde68a", "#fecaca", "#e5e7eb"                                 # real tensor, complex as (re, im) pair, real scalar, fixed table
W, H, GAP, NOTE = 4.3, 0.85, 0.5, 0.6                                                                  # box width / height, gap for the arrow, note height

def box(ax, x, y, w, h, text, fc, fs=8.4, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec="#374151", lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", linespacing=1.25)

def column(ax, x, title, rows, top):
    ax.text(x + W / 2, top + 0.15, title, ha="center", va="bottom", fontsize=11.5, fontweight="bold"); y = top - H
    for i, (text, fc, note) in enumerate(rows):
        box(ax, x, y, W, H, text, fc)
        if note: ax.text(x + W / 2, y - 0.08, note, fontsize=7.4, color="#4b5563", ha="center", va="top", linespacing=1.2)
        if i < len(rows) - 1: ax.annotate("", xy=(x + W / 2, y - NOTE - GAP + 0.05), xytext=(x + W / 2, y - NOTE - 0.02), arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#374151"))
        y -= H + NOTE + GAP
    return y + H + NOTE + GAP                                                                            # y of the last box

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="figures/domain_schematic.png"); a = ap.parse_args()
    fig, ax = plt.subplots(figsize=(16, 13.5)); ax.set_xlim(0, 16); ax.set_ylim(-0.2, 15.2); ax.axis("off")
    fig.suptitle("real and complex domains in the three NIK models: every gradient that reaches a parameter is d(real scalar) / d(real tensor)", fontsize=12.5, fontweight="bold", y=0.985)
    common_in = ("inputs: kx, ky, t (real), coil index (integer)", REAL, "Fourier features of (kx, ky) and of t;\ncoil -> 8-dim learned embedding (input-coil mode)")
    backbone = ("Gabor / WIRE backbone, 12 layers\nreal arithmetic, width 2 x hidden", REAL, "each layer: g = exp(-(s0 h)^2), out = [g cos(w0 h), g sin(w0 h)]\nconcatenated: two REAL branches, never a complex tensor")
    loss = ("loss = sum over the batch of\n(p_re - y_re)^2 + (p_im - y_im)^2", SCALAR, "= |p - y|^2 in real parts: a real scalar, ordinary real gradient\n(identical to 2 dL/dz-bar, the Wirtinger descent direction)")
    top = 14.3
    y1 = column(ax, 0.5, "NIK-free (wire_ff_res)", [common_in, backbone,
        ("head: Linear -> 2 numbers\n(p_re, p_im) at (k, t, coil)", PAIR, "the complex prediction as a pair; t enters the backbone,\nso the time dependence is nonlinear: no coefficient maps, no support prior"),
        loss], top)
    y2 = column(ax, 5.85, "NIK-sub16 (wire_ff_subspace)", [common_in, backbone,
        ("head: Linear -> 2R numbers\nR = 16 complex coefficients a_r(k, coil)", PAIR, "R pairs (re, im); output-coil mode: 2RC numbers, coil picked at the head"),
        ("temporal net: FF(t) -> SIREN -> 2R\nR COMPLEX atoms Phi_r(t)", PAIR, "learned, warm-started from the navigator PCA;\ncomplex so a time-varying phase can be absorbed"),
        ("prediction: complex dot over r\np_re = sum(a_re P_re - a_im P_im)\np_im = sum(a_re P_im + a_im P_re)", PAIR, "the complex product written out in real parts: no complex multiply op"),
        loss], top)
    y3 = column(ax, 11.2, "NIK-tofts / patlak (wire_ff_tofts)", [common_in, backbone,
        ("head: Linear -> 2R numbers\nR complex coefficients a_r(k, coil)", PAIR, "R = 8 (tofts8) or 3 (patlak); output-coil mode: 2RC numbers"),
        ("fixed atom table [342, R], REAL\ninterpolated at t -> Phi_r(t), imag = 0", FIXED, "unit-rms orthonormal atoms from the basis file; no parameters, no gradient"),
        ("prediction: p = sum_r a_r Phi_r(t)\np_re = sum a_re Phi,  p_im = sum a_im Phi", PAIR, "real atoms: re and im channels never mix; the phase lives in a_r only"),
        loss], top)
    # priors (coefficient-map arms)
    py = y3 - NOTE - 1.55; box(ax, 5.85, py, 9.65, 1.25, "support / TV priors, coefficient-map arms only (sub16, tofts, patlak):\na_r on the full k grid -> torch.complex -> ifft2 -> |image|^2 -> real penalty\nthe only native complex tensors with gradient; PyTorch complex autograd returns the conjugate-Wirtinger gradient of the real penalty;\n|.|^2 is smooth, the TV term is Huber (no kink at zero)", REAL, fs=7.8)
    for xc in (5.85 + W / 2, 11.2 + W / 2): ax.annotate("", xy=(xc, py + 1.25), xytext=(xc, y3 - NOTE - 0.05), arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#374151", linestyle="--"))
    # legend + statement
    for k, (fc, lab) in enumerate(((REAL, "real tensor"), (PAIR, "complex value stored as two real channels (re, im)"), (FIXED, "fixed table, no parameters"), (SCALAR, "the one real scalar that is differentiated"))):
        ax.add_patch(FancyBboxPatch((0.5, py + 1.05 - 0.34 * k), 0.32, 0.22, boxstyle="round,pad=0.01", fc=fc, ec="#374151", lw=0.8)); ax.text(0.9, py + 1.16 - 0.34 * k, lab, fontsize=8, va="center")
    ax.text(0.5, py - 1.05, "why no holomorphic derivative is ever needed: each model is a map from REAL parameters to a REAL scalar L; autograd differentiates that map.\n"
            "complex numbers appear only as pairs of reals (main path) or, in the priors, as native complex tensors feeding |.|^2, where PyTorch applies the Wirtinger rule.\n"
            "after training the rendered frames are magnitudes; no derivative is ever taken through them.", fontsize=8.2, va="bottom")
    fig.savefig(a.out, dpi=150, facecolor="white", bbox_inches="tight"); print("saved", a.out)

if __name__ == "__main__": main()
