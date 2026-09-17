"""how much of the first-pass peak does the 31-spoke model-free reference lose to its own window? regrid the model-free series with
narrower sliding windows (same nufft, ramp dcf, sense combine as step2_kidney.py) and read the roi enhancement peaks with the approved
rois. a peak that keeps rising as the window narrows means the 31-spoke reference under-reads the true peak, and a nik peak ratio
below 1 against it is not necessarily an under-read. out: results/tofts_vs_patlak/mf_peak_check_sl<Z>.md
usage: python mf_peak_check.py --slice 21 --windows 31,21,15,11,7"""
import warnings; warnings.filterwarnings("ignore")
import sys, json, argparse, numpy as np, finufft
D = "/net/beegfs/users/P101440/DCE_NIK"; REF = "/net/beegfs/users/P101440/grasp_pro_py/results_ref"; sys.path.insert(0, D); sys.path.insert(0, "/net/beegfs/users/P101440/grasp_pro_py")
import consolidated as C
TA = 375.0

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--slice", type=int, default=21); ap.add_argument("--windows", default="31,21,15,11,7"); a = ap.parse_args(); Z = a.slice
    sh = np.load(f"{REF}/shared.npz"); traj = np.asarray(sh["traj_norm"]).astype(np.complex64); vt = np.asarray(sh["view_time"]).ravel().astype(np.float64)
    nx = int(sh["nx"]); bas = int(sh["bas"]); sl = np.load(f"{REF}/slice_{Z:02d}.npz"); kdata = np.asarray(sl["kdata_radial"]).astype(np.complex64)
    b1 = np.asarray(sl["b1"]).astype(np.complex64); ncc = kdata.shape[2]; den = np.sum(np.abs(b1) ** 2, 2) + 1e-12
    SIGN = json.load(open(f"{D}/results_nufft/meta.json"))["sign"]; order = np.argsort(vt)
    ctx = C.slice_ctx(Z); rois = ctx["rois"]; names = [r for r in ("aorta", "cortex", "medulla") if r in rois]
    def win_img(idx):
        tr = traj[:, idx]; w = np.maximum(np.abs(tr), 1 / nx / 4)
        x = (SIGN * 2 * np.pi * tr.real).ravel().astype(np.float64); y = (SIGN * 2 * np.pi * tr.imag).ravel().astype(np.float64)
        acc = sum(finufft.nufft2d1(x, y, (kdata[:, idx, c] * w).astype(np.complex128).ravel(), (nx, nx), isign=1, eps=1e-4) * np.conj(b1[:, :, c]) for c in range(ncc))
        s = (nx - bas) // 2; return np.abs(acc / den)[s:s + bas, s:s + bas] / len(idx)                                   # per-spoke normalization so windows of different width are on one scale
    out = {}; ref31 = None
    for W in [int(w) for w in a.windows.split(",")]:
        step = max(W // 4, 2); wins = [order[i:i + W] for i in range(0, len(order) - W + 1, step)]; t = np.array([vt[idx].mean() * TA for idx in wins])
        cur = {r: [] for r in names}
        for idx in wins:
            im = win_img(idx)
            for r in names: cur[r].append(float(im[rois[r]].mean()))
        row = {}
        for r in names:
            c = np.array(cur[r]); e = c - np.median(c[t < 40]); pk = (t > 20) & (t < 210)
            row[r] = dict(peak=float(e[pk].max()), ttp=float(t[pk][np.argmax(e[pk])]), washout=float(e[t > 200].mean()), noise=float(np.std(np.diff(e[t > 200])) / np.sqrt(2)))
        out[W] = dict(frames=len(wins), window_s=float(W * np.median(np.diff(np.sort(vt))) * TA), **row); ref31 = ref31 or (row if W == 31 else None)
        print(W, {r: round(row[r]["peak"], 6) for r in names}, flush=True)
    r31 = out[31] if 31 in out else out[max(out)]
    L = [f"# model-free reference: first-pass peak vs sliding-window width, slice {Z} (nufft, ramp dcf, sense combine, approved rois; enhancement = baseline-subtracted roi mean)", "",
         "| window (spokes) | window (s) | frames | " + " | ".join(f"{r} peak (ratio to 31) / ttp s / late noise" for r in names) + " |", "|---|---|---|" + "---|" * len(names)]
    for W in sorted(out, reverse=True):
        o = out[W]; L.append(f"| {W} | {o['window_s']:.1f} | {o['frames']} | " + " | ".join(f"{o[r]['peak']:.5f} ({o[r]['peak'] / (r31[r]['peak'] + 1e-12):.2f}) / {o[r]['ttp']:.0f} / {o[r]['noise'] / (o[r]['peak'] + 1e-12):.3f}" for r in names) + " |")
    L += ["", "reading: ratio > 1 at narrow windows = the 31-spoke reference under-reads the peak by that factor (temporal smoothing of the window); the late noise column shows what the narrower window costs"]
    open(f"{D}/results/tofts_vs_patlak/mf_peak_check_sl{Z}.md", "w").write("\n".join(L)); json.dump(out, open(f"{D}/results/tofts_vs_patlak/mf_peak_check_sl{Z}.json", "w"), indent=1); print("\n".join(L)); print("MF_PEAK_DONE")

if __name__ == "__main__": main()
