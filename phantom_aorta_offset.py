"""phantom: why the nik washout sits above truth in the aorta. partial volume (offset shrinks when the roi is eroded, error bipolar at the
vessel edge) vs a regional low-|k| offset (offset flat under erosion, smooth error bump around the vessel) vs a temporal-model effect (differs
between nik-free and nik-tofts). same input for every method (5 of 7 spokes), one global scale vs truth on the body, late window 100 to 175 s.
out: results/xcat_physical_nomotion_nik_vs_grasp/aorta_offset.{md,json}, figures/aorta_offset.png
usage: XPH_SIM=nomotion python phantom_aorta_offset.py"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, json, argparse, numpy as np, scipy.ndimage as ndi
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/net/beegfs/users/P101440/DCE_NIK")
import xph_pipeline as P, xph_common as X
from story_figs import ls_scale, winavg

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--extra", default="", help="name:path[:key],... additional recons on the frame grid tq"); ap.add_argument("--suffix", default=""); a = ap.parse_args()
    d = P.data(); tq = np.asarray(d["times"], float); lab = np.asarray(d["labels"]); body = lab > 0; Tr = X.truth_at(P.ZI, tq).astype(np.float32); Rz = X.rois(P.ZI, lab); A = f"{P.OUT}/arrays"; G = 5
    tw = np.array([tq[g*G:(g+1)*G].mean() for g in range(len(tq) // G)]); Tw = winavg(Tr, G)
    L = lambda p, key=None: (lambda z: np.abs(z[key] if key else z).astype(np.float32))(np.load(p, allow_pickle=True))
    M = {"NIK-free": (ls_scale(L(f"{A}/nik_fine_free.npy"), Tr, body), tq, Tr), "NIK-tofts": (ls_scale(L(f"{A}/nik_eval_w768_ks2.5_s0_tofts16.npz", "rec_best"), Tr, body), tq, Tr),
         "NIK-sub16": (ls_scale(L(f"{A}/nik_fine_sub16.npy"), Tr, body), tq, Tr), "GRASP-Pro K5": (ls_scale(L(f"{A}/grasp_pro_K5_G5.npz", "rec"), Tw, body), tw, Tw), "GRASP": (ls_scale(L(f"{P.OUT}/v2_sweep/v2_G05.npy"), Tw, body), tw, Tw)}
    for spec in [x for x in a.extra.split(",") if x]:
        nm, pth = spec.split(":")[0], spec.split(":")[1]; key = spec.split(":")[2] if spec.count(":") > 1 else None
        if os.path.exists(pth): M[nm] = (ls_scale(L(pth, key), Tr, body), tq, Tr)
        else: print("missing", pth)
    late = lambda t: (t > 100) & (t < 175); pre = lambda t: t < 15
    ao = Rz["aorta"]; ring = ndi.binary_dilation(ao, iterations=6) & ~ndi.binary_dilation(ao, iterations=2) & body
    out = {}; rows = []
    for nm, (v, t, T) in M.items():
        e_late = v[..., late(t)].mean(2) - T[..., late(t)].mean(2); e_pre = v[..., pre(t)].mean(2) - T[..., pre(t)].mean(2)
        r = dict(ring_late=float(e_late[ring].mean()), ring_pre=float(e_pre[ring].mean()), aorta_px=int(ao.sum()))
        for k in (0, 1, 2, 3):
            m = ndi.binary_erosion(ao, iterations=k) if k else ao
            if m.sum() < 4: break
            r[f"erode{k}_late"] = float(e_late[m].mean()); r[f"erode{k}_pre"] = float(e_pre[m].mean()); r[f"erode{k}_px"] = int(m.sum())
        r["truth_aorta_late"] = float(T[..., late(t)].mean(2)[ao].mean()); r["truth_ring_late"] = float(T[..., late(t)].mean(2)[ring].mean())
        for kr in ("cortex", "medulla"):
            if kr in Rz: r[f"{kr}_late"] = float(e_late[Rz[kr]].mean()); r[f"{kr}_truth_late"] = float(T[..., late(t)].mean(2)[Rz[kr]].mean())
        out[nm] = r; rows.append((nm, r))
    md = ["# phantom aorta washout offset (method minus truth, late window 100 to 175 s, one global scale on the body)", "",
          f"aorta roi {int(ao.sum())} px, truth late aorta {rows[0][1]['truth_aorta_late']:.3f}, truth late ring (2 to 6 px around) {rows[0][1]['truth_ring_late']:.3f}", "",
          "| method | aorta offset, erode 0 / 1 / 2 / 3 px | pre-contrast offset (erode 0) | ring offset late / pre | cortex / medulla late offset |", "|---|---|---|---|---|"]
    for nm, r in rows:
        er = " / ".join(f"{r[f'erode{k}_late']:+.4f} ({r[f'erode{k}_px']})" for k in range(4) if f"erode{k}_late" in r)
        md.append(f"| {nm} | {er} | {r['erode0_pre']:+.4f} | {r['ring_late']:+.4f} / {r['ring_pre']:+.4f} | {r.get('cortex_late', float('nan')):+.4f} / {r.get('medulla_late', float('nan')):+.4f} (truth {r.get('cortex_truth_late', float('nan')):.3f} / {r.get('medulla_truth_late', float('nan')):.3f}) |")
    md += ["", "reading: offset falling with erosion and a ring offset of opposite sign = partial volume / blur; flat offset with the same sign in the ring = regional low-|k| error; nik-free vs nik-tofts equal = not the temporal model"]
    open(f"{P.OUT}/aorta_offset{a.suffix}.md", "w").write("\n".join(md)); json.dump(out, open(f"{P.OUT}/aorta_offset{a.suffix}.json", "w"), indent=1); print("\n".join(md))
    c = np.argwhere(ao).mean(0).astype(int); s = 28; sl = (slice(c[0]-s, c[0]+s), slice(c[1]-s, c[1]+s))
    fig, ax = plt.subplots(2, len(M), figsize=(3.4 * len(M), 6.6))
    for j, (nm, (v, t, T)) in enumerate(M.items()):
        e = (v[..., late(t)].mean(2) - T[..., late(t)].mean(2)); vm = 0.06
        ax[0, j].imshow(e[sl], cmap="RdBu_r", vmin=-vm, vmax=vm); ax[0, j].contour(ao[sl].astype(float), levels=[0.5], colors=["k"], linewidths=0.7); ax[0, j].set_title(f"{nm}\nlate error, zoom on the aorta", fontsize=9); ax[0, j].axis("off")
        ax[1, j].plot(t, v[..., :][ao].mean(0), label=nm); ax[1, j].plot(t, T[ao].mean(0), "k", lw=1.8, label="truth"); ax[1, j].plot(t, v[ndi.binary_erosion(ao, iterations=2)].mean(0) if ndi.binary_erosion(ao, iterations=2).sum() >= 4 else np.nan * t, "--", label="eroded 2 px"); ax[1, j].set_xlabel("t [s]"); ax[1, j].legend(fontsize=7)
    fig.suptitle("phantom aorta: late-frame error map and roi curve, full roi vs eroded roi", fontsize=11); fig.tight_layout(); fig.savefig(f"{P.OUT}/figures/aorta_offset{a.suffix}.png", dpi=130, facecolor="white"); print("AORTA_DONE")

if __name__ == "__main__": main()
