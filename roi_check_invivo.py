"""roi placement check, in vivo k80: the slice's roi masks (consolidated.slice_ctx, one set per slice, shared by every method) drawn on the
model-free reference and on each method's image at t_show, plus mask sizes and where the masks sit relative to the kidney/aorta bright blobs.
out: results/realdata_nik_vs_cs_figures/figures/roi_check_k80_sl<Z>.png, roi_check_k80.md
usage: python roi_check_invivo.py --slices 21,18,19 --t-show 90"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, argparse, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B = "/net/beegfs/users/P101440/DCE_NIK"; sys.path.insert(0, B)
import dsp                                                                                   # dataset paths (DCE_DS=p3 default / p8)
GV = dsp.GV; GP = dsp.GP
import consolidated as C
COL = dict(aorta="cyan", cortex="lime", medulla="orange", liver="magenta")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--slices", default="21,18,19"); ap.add_argument("--t-show", type=float, default=90.0); a = ap.parse_args(); TA = 375.0
    L = ["# roi check, in vivo k80 (masks from consolidated.slice_ctx: grasp-pro 100% anatomy, identical for every method)", "", "| slice | " + " | ".join(f"{r} px" for r in COL) + " | cortex/medulla centroid (row, col) | aorta centroid |", "|---|" + "---|" * (len(COL) + 2)]
    for Z in [int(s) for s in a.slices.split(",")]:
        ctx = C.slice_ctx(Z); rois = ctx["rois"]; z = np.load(dsp.STEP2(Z)); mf = np.abs(z["mf"]).transpose(1, 2, 0); tmf = np.asarray(z["tmf"], float)
        IV = f"{B}/results/tofts_vs_patlak/invivo_k80"
        src = [("model-free", mf, tmf), ("NIK-free", f"{B}/results_sl{Z}_k80/nik_slice_{Z}.npy", None), ("NIK-sub16", f"{B}/results/realdata_nik_vs_cs_figures/outcoil_subspace_output_slice{Z}.npy", None),
               ("NIK-patlak", f"{IV}/patlak_sl{Z}_s0/nik_slice_{Z}_cplx.npy", None), ("NIK-tofts", f"{IV}/tofts_sl{Z}_s0/nik_slice_{Z}_cplx.npy", None), ("NIK-tofts8", f"{IV}/tofts8_sl{Z}_s0/nik_slice_{Z}_cplx.npy", None),
               ("GRASP-Pro", f"{GP}/cs_slice{Z}_f80match.npy", None), ("GRASP", f"{GV}/gv2_slice{Z}_n12_k80.npy", None)]
        ims = []
        for nm, v, t in src:
            if isinstance(v, str):
                if not os.path.exists(v): print("missing", v); continue
                v = np.abs(np.load(v)).astype(np.float32)
            nt = v.shape[-1]; t = t if t is not None else (np.arange(nt) + 0.5) * TA / nt
            if nm == "model-free": m = (tmf > a.t_show - 8) & (tmf < a.t_show + 8); im = v[..., m].mean(2)
            else: im = v[..., int(np.argmin(np.abs(t - a.t_show)))]
            ims.append((nm, im))
        cen = lambda m: tuple(int(x) for x in np.argwhere(m).mean(0)) if m.any() else None
        L.append(f"| {Z} | " + " | ".join(str(int(rois[r].sum())) if r in rois else "-" for r in COL) + f" | {cen(rois['cortex'])} / {cen(rois['medulla'])} | {cen(rois['aorta'])} |")
        n = len(ims); fig, ax = plt.subplots(2, (n + 1) // 2, figsize=(3.6 * ((n + 1) // 2), 7.4)); ax = ax.ravel()
        for k, (nm, im) in enumerate(ims):
            ax[k].imshow(im, cmap="gray", vmin=0, vmax=np.percentile(im[ctx["BODY"]], 99.5)); ax[k].set_title(nm, fontsize=11); ax[k].axis("off")
            for r, c in COL.items():
                if r in rois and rois[r].any(): ax[k].contour(rois[r].astype(float), levels=[0.5], colors=[c], linewidths=1.0)
        for k in range(n, len(ax)): ax[k].axis("off")
        ax[0].plot([], [], color="cyan", label="aorta"); ax[0].plot([], [], color="lime", label="cortex"); ax[0].plot([], [], color="orange", label="medulla"); ax[0].plot([], [], color="magenta", label="liver (static)"); ax[0].legend(fontsize=7, loc="lower left")
        fig.suptitle(f"slice {Z}, k80: the shared roi masks on every method at t = {a.t_show:.0f} s", fontsize=12); fig.tight_layout()
        out = f"{B}/results/realdata_nik_vs_cs_figures/figures/roi_check_k80{dsp.SFX}_sl{Z}.png"; fig.savefig(out, dpi=130, facecolor="white"); plt.close(fig); print("saved", out, flush=True)
    open(f"{B}/results/realdata_nik_vs_cs_figures/roi_check_k80{dsp.SFX}.md", "w").write("\n".join(L)); print("\n".join(L)); print("ROI_CHECK_DONE")

if __name__ == "__main__": main()
