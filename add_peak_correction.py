"""append the reference-peak correction to the in vivo curve tables: the 31-spoke model-free reference clips the cortex / medulla / aorta first pass
(mf_peak_check.py: 11-spoke window over 31-spoke, per-spoke normalized); every peak ratio vs the reference is divided by that factor so a nik peak
above the 31-spoke reference is not read as overshoot. runs on the laptop from the synced json files. usage: python add_peak_correction.py invivo_k80_rms1 invivo_k80_oc"""
import sys, json, os, glob, re, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import dsp
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "tofts_vs_patlak"); WIN = "11"; SFX = "" if os.environ.get("DCE_DS", "p3") == "p3" else "_" + os.environ["DCE_DS"]
corr = {}
for p in sorted(glob.glob(f"{R}/mf_peak_check{SFX}_sl*.json")):                                          # every slice with a peak check for this dataset
    Z = int(re.search(r"_sl(\d+)\.json$", p).group(1))
    o = json.load(open(p)); raw = {r: o[WIN][r]["peak"] / o["31"][r]["peak"] for r in dsp.ROI_NAMES}
    legacy = raw[dsp.T1] < 0.6                                                              # json written before the per-spoke normalization: peaks scale with the window width
    corr[Z] = {r: (raw[r] * 31.0 / float(WIN) if legacy else raw[r]) for r in raw}
for suf in sys.argv[1:]:
    rows = json.load(open(f"{R}/{suf}.json")); md = f"{R}/{suf}.md"; s = open(md, encoding="utf-8").read()
    if "## reference peak correction" in s: s = s.split("## reference peak correction")[0].rstrip("\n") + "\n"
    L = ["", "## reference peak correction", "", f"the 31-spoke model-free reference (6.8 s window) clips the first-pass peak; factor = peak at an {WIN}-spoke window / peak at 31 spokes (`mf_peak_check.py`, per-spoke normalized, approved rois). corrected ratio = ratio vs the 31-spoke reference / factor. state this with every peak comparison.", "",
         "| slice | roi | clip factor | " + " | ".join(f"{a} corrected peak ratio" for a in sorted({r["arm"] for r in rows if r.get("status") == "complete" and r.get("seed", -1) >= 0})) + " | GRASP-v2 corrected | GRASP-Pro corrected |", "|---|---|---|" + "---|" * (len({r["arm"] for r in rows if r.get("status") == "complete" and r.get("seed", -1) >= 0}) + 2)]
    arms = sorted({r["arm"] for r in rows if r.get("status") == "complete" and r.get("seed", -1) >= 0})
    for Z in sorted(corr):
        for roi in (dsp.T1, dsp.T2, "aorta"):
            f = corr[Z][roi]; k = f"{roi}_peak_ratio"
            def m(arm): v = [r[k] for r in rows if r.get("slice") == Z and r.get("arm") == arm and k in r]; return f"{np.mean(v) / f:.2f} (raw {np.mean(v):.2f})" if v else "-"
            def g(pre): v = [r[k] for r in rows if r.get("slice") == Z and str(r.get("arm", "")).startswith(pre) and k in r]; return f"{v[0] / f:.2f} (raw {v[0]:.2f})" if v else "-"
            L.append(f"| {Z} | {roi} | {f:.2f} | " + " | ".join(m(a) for a in arms) + f" | {g('GRASP-v2')} | {g('GRASP-Pro')} |")
    open(md, "w", encoding="utf-8").write(s + "\n".join(L) + "\n"); print("appended to", md); print("\n".join(L[4:]))
