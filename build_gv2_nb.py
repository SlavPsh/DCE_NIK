"""build the grasp-v2 twin of NIK_vs_CS_combined_report.ipynb.

clones the EXISTING notebook (not build_report_nb.py, which is stale and no longer emits
sections 4-8) and rewrites only the reference-method bits. BOTH sections are now genuinely v2:
  phantom  -> xph_grasp_v2.py recon, re-aggregated (_gv2 figures + csvs)
  real     -> grasp_v2_real.py recons, re-scored (_gv2 figures + jsons)
K-selection cells are DROPPED: K is grasp-pro's pca rank and classic grasp v2 has no subspace.
then re-executes so every table is recomputed.
"""
import os, re, sys
import nbformat as nbf
from nbclient import NotebookClient

B = "/net/beegfs/users/P101440/DCE_NIK"
PH = f"{B}/results/xcat_physical_nomotion_nik_vs_grasp"
SRC = f"{PH}/NIK_vs_CS_combined_report.ipynb"
DST = f"{PH}/NIK_vs_GRASPv2_combined_report.ipynb"
TAG = "_gv2"

# figures regenerated against grasp v2 (real in-vivo + phantom). anything absent stays as-is.
SWAP_FIGS = {
    # real in-vivo
    "fig1_spoke_frontier", "fig2_pk_cov", "fig3_denoising", "fig4_realdata_montage",
    "fig5_realdata_aorta_curve", "fig7_realdata_kidney_curves", "step1_images",
    "step1_compare", "step2_compare", "step3_compare", "step7_fairgrasp",
    "calib_curves", "outcoil_report",
    # phantom (xph_aggregate re-run with GRASP_NPZ=grasp_v2_recon.npz)
    "fig0_roi_check", "fig1_montage", "fig2_errormaps", "fig6_pk_maps",
}
# NOT swapped: fig6_roi_verification + fig8 motion gif (roi anatomy, shared by both notebooks on
# purpose), fig_bias_locate and outcoil_compare (nik-only, no grasp dependence).
SWAP_FILES = {"task_S.json": f"task_S{TAG}.json", "task2.json": f"task2{TAG}.json",
              "haarpsi_spoke.json": f"haarpsi_spoke{TAG}.json",
              "l3_rebaseline.csv": f"l3_rebaseline{TAG}.csv",
              "pk_metrics.csv": f"pk_metrics{TAG}.csv",
              "compare_recons.npz": f"compare_recons{TAG}.npz"}

# K is grasp-pro's pca rank. classic grasp v2 has no subspace -> drop every K-selection cell.
DROP_PATTERNS = ["choosing k, held-out cv not oracle", "grasp_kcv.csv", "navigator eigenspectrum",
                 "fig_nav_spectrum.png", "k-sweep frontier, per-spoke navigator",
                 "grasp_ksweep_pareto.csv", "note: old k x spokes/frame figure stale"]
DROP_NOTE = ("### k selection, not applicable\n\n"
             "grasp-pro picks a temporal pca rank k. classic grasp v2 has no subspace, so there is no k "
             "to choose and the k-sweep, held-out cv and navigator eigenspectrum cells are dropped. the "
             "v2 knob is the temporal tv weight, left at the published 0.25*max|x0|.")

# prose whose grasp-pro-specific claims do not carry over. anchors contain no "grasp-pro", so they
# survive the generic rename and are applied AFTER it.
FRONTIER_ANCHOR = "cs pca basis now from the retained spokes only."
FRONTIER_V2 = ("classic grasp v2 has no temporal subspace, so the basis leak that affected grasp-pro at "
               "reduced spoke fractions cannot occur here. this frontier is fair by construction.")
SUM_PRO = "level after removing the cs temporal-basis leak; nik ahead at 50%; was wrongly cs-favored"
SUM_V2 = "grasp v2 has no temporal subspace, so no basis leak is possible; frontier fair by construction"
SUM_K_PRO = "| phantom k selection | held-out cv k*=12 (sharp min), captures bolus |"
SUM_K_V2 = "| phantom k selection | n/a, grasp v2 has no temporal subspace |"
# l3_rebaseline row label follows GRASP_LABEL, so notebook lookups by row name must follow too
ROW_PRO = "GRASP-K12 (ref)"
ROW_V2 = "GRASP-v2"
LAB_PRO = "grasp-k12"
LAB_V2 = "grasp v2"
METH_PRO = "methods: cs-file, grasp v2 (k12, cv), nik f0/sub5/sub16/free."
METH_V2 = "methods: cs-file, grasp v2 (classic 2014, no subspace), nik f0/sub5/sub16/free."


def swap_figs(s):
    def sub(m):
        path, base, ext = m.group(0), m.group(1), m.group(2)
        if base in SWAP_FIGS and TAG not in path:
            return path.replace(f"{base}.{ext}", f"{base}{TAG}.{ext}")
        return path
    return re.sub(r"[\w/\.\-]*?([\w\-]+)\.(png|gif)", sub, s)


def rewrite(s):
    s = swap_figs(s)
    for a, b in SWAP_FILES.items():
        s = s.replace(f"'{a}'", f"'{b}'").replace(f'"{a}"', f'"{b}"').replace(f"/{a}", f"/{b}")
    s = s.replace("grasp-pro", "grasp v2").replace("grasp pro", "grasp v2")
    # applied last so our own wording is not rewritten by the rename above
    if FRONTIER_ANCHOR in s:
        s = s.split("### spoke-fraction frontier")[0] + "### spoke-fraction frontier, grasp v2\n\n" + FRONTIER_V2
    for a, b in ((SUM_PRO, SUM_V2), (SUM_K_PRO, SUM_K_V2), (METH_PRO, METH_V2), (ROW_PRO, ROW_V2), (LAB_PRO, LAB_V2)):
        s = s.replace(a, b)
    return s


def drop_k_cells(cells):
    out, dropped, noted = [], 0, False
    for c in cells:
        if any(p in c.source for p in DROP_PATTERNS):
            dropped += 1
            if not noted:
                out.append(nbf.v4.new_markdown_cell(DROP_NOTE)); noted = True
            continue
        out.append(c)
    print(f"dropped {dropped} grasp-pro K-selection cells")
    return out



PH_FIG = f"{PH}/figures/fig_v2_frontier.png"
IV_FIG = f"{B}/results/realdata_nik_vs_cs_figures/figures/fig_v2_frontier_invivo.png"

def frontier_cells():
    """section: spokes/frame is grasp v2's spatial-vs-temporal knob; nik does not have to trade."""
    C = []
    C.append(nbf.v4.new_markdown_cell(
        "# 9. spatial vs temporal frontier, spokes/frame\n\n"
        "grasp v2 trades spatial against temporal through spokes/frame. more spokes per frame gives better "
        "images and fewer frames. nik renders continuously and never rebins, so it does not make that trade.\n\n"
        "the sweep exists to give grasp v2 its OWN best operating point rather than the arbitrary one it was "
        "first run at. rulers are identical at every point: spatial = haarpsi vs the reference averaged over the "
        "same window the frame integrates, temporal = aorta curve nrmse on the fine grid. nik is scored on the "
        "same windows by time-averaging its continuous render.\n\n"
        "grasp v2 lam is the published 0.25*max|x0| at every point, not tuned per spokes/frame."))
    C.append(nbf.v4.new_markdown_cell("### phantom, vs xcat truth"))
    C.append(nbf.v4.new_code_cell(
        "import json, glob, pandas as pd\n"
        f"r = json.load(open('{PH}/v2_sweep/frontier.json'))\n"
        "df = pd.DataFrame(r)[['method','spf','frames','dt','haarpsi','ssim','c_aorta','aorta_pk']]\n"
        "df = df.rename(columns={'spf':'spokes/frame','dt':'s/frame','c_aorta':'aorta nrmse','aorta_pk':'aorta peak'})\n"
        "display(df.round(4).sort_values(['method','spokes/frame']).reset_index(drop=True))"))
    C.append(nbf.v4.new_code_cell(f"display(Image('{PH_FIG}'))"))
    C.append(nbf.v4.new_markdown_cell(
        "phantom read. methods: **NIK-free** (wire_ff, full rank, 3-seed complex average) and **NIK-sub16** "
        "(wire_ff_subspace rank 16, 2-seed) vs **GRASP-v2 classic 2014** (MCNUFFT + temporal TV only, no "
        "subspace, lam = 0.25*max|x0|, published value, NOT tuned per spokes/frame). all three on the SAME "
        "5 of 7 angles per frame (TRAIN_ANG, 71% of acquired), so the spoke budget is matched. reference is "
        "xcat ground truth.\n\n"
        "grasp v2 spatial peaks at 60 spokes/frame (haarpsi 0.921) and its temporal best is at 25 "
        "(aorta nrmse 0.101). nik is flat across the whole sweep (0.884 to 0.890) because it never rebins.\n\n"
        "nik-free wins BOTH axes at 15 spokes/frame and below. in the 2-5 s/frame band grasp v2 is spatially "
        "better (0.902-0.917 vs 0.885-0.886), so there is no combined win there. that band was fixed before "
        "scoring nik.\n\n"
        "aorta peak, truth 0.805: grasp v2 reaches at best 0.651 (25 spokes/frame) and under-reads by 19 to 41 "
        "percent at every operating point. nik is within 4 to 7 percent. no spokes/frame setting recovers it, "
        "which is the amplitude damping of temporal tv without a subspace."))
    C.append(nbf.v4.new_markdown_cell(
        "### single reference setting: 25 spokes/frame, lam 0.25\n\n"
        "one setting for grasp v2, fixed on the phantom by the ideal-corner rule on (haarpsi vs truth, "
        "aorta nrmse vs truth), then carried to in vivo unchanged as NLINE 12 (2.63 vs 2.61 s/frame). "
        "NOT re-selected in vivo, and no best-of across settings. earlier drafts quoted grasp v2's best "
        "haarpsi (60 spf) and best aorta curve (25 spf) together, which was a cherry pick since it cannot "
        "deliver both at once.\n\n"
        "lam stays at the published 0.25. spokes/frame is the knob under study."))
    C.append(nbf.v4.new_code_cell(
        "import json, pandas as pd\n"
        f"j = json.load(open('{PH}/v2_sweep/final_single_setting.json'))\n"
        "print('phantom, vs xcat truth'); display(pd.DataFrame(j['phantom']).round(4))\n"
        "print('in vivo, vs model-free nufft. CURVES ONLY, haarpsi excluded'); display(pd.DataFrame(j['invivo']).round(4))"))
    C.append(nbf.v4.new_code_cell(f"display(Image('{PH}/figures/fig_v2_vs_nik.png'))"))
    C.append(nbf.v4.new_markdown_cell(
        "### image quality panel, phantom vs truth, single setting\n\n"
        "haarpsi/ssim/psnr all reward smoothness, so a blurrier but quieter recon can win them. this "
        "panel adds more full-reference metrics plus a resolution vs noise split and per-band error. "
        "ALL methods on the same body mask and the same window-averaged truth (NIK binned to grasp's "
        "68 frames). sharpness_rel = edge energy in-body relative to truth, 1.0 = matches; bg_noise = "
        "background std. err_lowk/midk/highk = relative error by spatial frequency band."))
    C.append(nbf.v4.new_code_cell(
        "import json, pandas as pd\n"
        f"j = json.load(open('{PH}/v2_sweep/image_metrics_panel.json'))\n"
        "cols = ['method','haarpsi','ssim','ms_ssim','srsim','dss','vif_p','gmsd','mdsi','psnr','nrmse',"
        "'err_lowk','err_midk','err_highk','sharpness_rel','bg_noise']\n"
        "display(pd.DataFrame(j)[[c for c in cols if c in j[0]]].round(4))\n"
        "print('higher better: haarpsi ssim ms_ssim srsim dss vif_p psnr sharpness_rel(->1.0)')\n"
        "print('lower  better: gmsd mdsi nrmse err_* bg_noise')"))
    C.append(nbf.v4.new_markdown_cell(
        "read, at the single setting.\n\n"
        "- **phantom images**: grasp v2 0.9023 vs NIK-sub16 0.8985. essentially a tie.\n"
        "- **phantom curves**: NIK-free clearly better on tissue, cortex 0.0111 vs 0.0720 (6.5x), medulla "
        "0.0142 vs 0.0498 (3.5x). grasp v2 better on aorta, 0.1007 vs 0.1431 for sub16 but WORSE than "
        "NIK-free's 0.0806.\n"
        "- **phantom bolus peak**: NIK +4.1% / +7.4%, grasp v2 **-19.1%**.\n"
        "- **in vivo curves**: grasp v2 better on all three rois and on bolus recovery (0.77 vs 0.73).\n\n"
        "the two datasets disagree and only the phantom has ground truth. in vivo the reference is itself "
        "a reconstruction, so it is the weaker evidence.\n\n"
        "checks done on this comparison: identical input spokes (1720 both, TRAIN_ANG only, val/test never "
        "used by either); the coarse-vs-fine frame count does NOT bias the curve metric (subsampling NIK's "
        "curve to grasp's 68 points moves nrmse by 0.0006-0.005, while the method gaps are 0.02-0.06); "
        "pearson r gives the same ranking; NIK's curves are SMOOTHER than truth (hf wobble 0.0007 vs truth "
        "0.0010-0.0024) so it is not being charged for noise.\n\n"
        "remaining asymmetry, favouring NIK: its phantom numbers are 2-3 seed complex averages, grasp v2 "
        "is a single run.\n\n"
        "in-vivo haarpsi vs the model-free recon is excluded throughout: the reference is a recon, not "
        "truth, and the metric is not informative for ranking."))
    C.append(nbf.v4.new_markdown_cell(
        "### real in-vivo, slice 21, vs model-free nufft reference\n\n"
        "methods: **NIK** (wire_ff_res, full rank, trained on keep_f100 = 1708 of 1710 spokes with NO "
        "heldout) vs **GRASP-v2 classic 2014** (MCNUFFT + temporal TV only, no subspace, lam = 0.25*max|x0|, "
        "all acquired spokes). spoke budgets match to within the binning remainder (1680-1710).\n\n"
        "NOTE this NIK is NOT the one in sections 2-8. those use the output-coil run at v%10<8 = 1368 spokes "
        "(80%, val/test held out), and figs 4-5 use a full-rank run at a random 70% (1197 spokes). the "
        "frontier needed a spoke-matched NIK, so one was retrained.\n\n"
        "reference is the method-neutral model-free nufft recon. LIMIT: it uses a 31-spoke "
        "sliding window, so its effective temporal resolution is about 6.8 s even though it is sampled every "
        "1.54 s. it cannot adjudicate the fine-temporal regime where the phantom both-axes claim sits. this "
        "measures the spatial side of the trade, nik flatness, and bolus recovery. consistency, not accuracy."))
    C.append(nbf.v4.new_code_cell(
        "import json, pandas as pd\n"
        f"r = json.load(open('{B}/v2_sweep_invivo/frontier_invivo.json'))\n"
        "df = pd.DataFrame(r)[['method','NLINE','frames','dt','c_aorta','aorta_pk_ratio']]\n"
        "df = df.rename(columns={'NLINE':'spokes/frame','dt':'s/frame','c_aorta':'aorta nrmse','aorta_pk_ratio':'peak / model-free'})\n"
        "display(df.round(4).sort_values(['method','spokes/frame']).reset_index(drop=True))"))
    C.append(nbf.v4.new_code_cell(f"display(Image('{IV_FIG}'))"))
    C.append(nbf.v4.new_markdown_cell(
        "### in vivo, at the same single setting (NLINE 12)\n\n"
        "setting carried from the phantom (25 spokes/frame = 2.61 s -> NLINE 12 = 2.63 s), NOT re-selected "
        "in vivo. FAIR pair: both on the same 1368 spokes (v%10<8). NIK = full-rank wire_ff_res trained on "
        "those spokes with heldout early stopping + LR schedule (results_sl21_k80); grasp v2 = NLINE 12, "
        "lam 0.25, same 1368 spokes via the keep mask (gv2_slice21_n12_k80)."))
    C.append(nbf.v4.new_code_cell(
        "import json, pandas as pd\n"
        f"j = json.load(open('{B}/v2_sweep_invivo/v2_vs_nik_invivo.json'))\n"
        "display(pd.DataFrame(j['table']).round(4))\n"
        "print('full NLINE x lam sweep:')\n"
        "display(pd.DataFrame(j['sweep']).round(4).sort_values(['lam','NLINE']).reset_index(drop=True))"))
    C.append(nbf.v4.new_code_cell(
        f"display(Image('{B}/results/realdata_nik_vs_cs_figures/figures/fig_v2_vs_nik_invivo.png'))"))
    C.append(nbf.v4.new_markdown_cell(
        "read, in vivo, k80 pair. on the curve-NRMSE-vs-model-free ruler grasp v2 is ahead on all three "
        "rois: aorta 0.153 vs 0.234, cortex 0.084 vs 0.196, medulla 0.058 vs 0.130; bolus recovery is tied "
        "(0.710 vs 0.714).\n\n"
        "DO NOT read this as a temporal win for grasp. the ruler is biased: the model-free reference is an "
        "unregularized gridding recon of the same family as grasp v2, so grasp resembles it by construction. "
        "the physical-bounds check shows it: this same grasp v2 k80 recon has an aorta bolus FWHM of 53.2 s "
        "vs NIK 23.1 s, the model-free 17.3 s, and the phantom ground-truth AIF 12.5 s. a recon 2-4x broader "
        "than physiology scoring better on curve nrmse means the ruler is not measuring temporal fidelity. "
        "in-vivo haarpsi vs the model-free recon is excluded for the same reason.\n\n"
        "what would settle it: held-out spokes (measured data neither method saw), which needs complex-valued "
        "grasp saves; and the FWHM bound extended to slices 18/19/20."))
    return C



PROV_TABLE = """| label | algorithm + key parameters | spokes used |
|---|---|---|
| **phantom, xcat no-motion** | | |
| NIK-F0 | wire_ff_patlak, patlak-free 0 (Patlak basis, no free atoms) | 5 of 7 angles/frame (TRAIN_ANG), 71% |
| NIK-sub5 | wire_ff_subspace, rank 5 | 5 of 7 angles/frame (TRAIN_ANG), 71% |
| NIK-sub12 | wire_ff_subspace, rank 12 | 5 of 7 angles/frame (TRAIN_ANG), 71% |
| NIK-sub16 | wire_ff_subspace, rank 16, 2-seed complex average | 5 of 7 angles/frame (TRAIN_ANG), 71% |
| NIK-free | wire_ff, full rank (no subspace), 3-seed complex average | 5 of 7 angles/frame (TRAIN_ANG), 71% |
| GRASP-Pro | NUFFT-SENSE + temporal PCA subspace K*=12 (held-out CV) + spatial&temporal TV | 5 of 7 angles/frame (TRAIN_ANG), 71% |
| GRASP-v2 (classic, 2014) | MCNUFFT + temporal TV only, NO subspace, lam=0.25*max|x0| (published, untuned) | 5 of 7 angles/frame (TRAIN_ANG), 71%; sweep groups G frames -> 5G/frame |
| CS-file | recon BUNDLED in the XCAT sim HDF5; algorithm not in our code, cannot be verified | 7 of 7 angles/frame, 100%, NOT spoke-matched (+40% vs NIK/GRASP) |
| **real in-vivo, slice 21** | | |
| NIK (output-coil) | outcoil_real.py, subspace rank 16, coil on output heads | v%10<8 = 1368 of 1710, 80%; val/test spokes HELD OUT |
| NIK (full-rank, UNMATCHED 70%) | wire_ff_res, default --subsample-frac 0.7 | random 1197 of 1710, 70%; 30% HELD OUT. name says 'full' but is NOT all spokes |
| NIK (full-rank, spoke-matched) | wire_ff_res, --spoke-keep-file keep_f100, no heldout | 1708 of 1710, 99.9%, same input as GRASP |
| GRASP-Pro f100 | GROG + temporal PCA K=5 + spatial TV 0.001 + temporal TV 0.0005, 14 spokes/frame (122 frames) | 1708 of 1710, 100% of binned |
| GRASP-Pro f80-match | as f100 but spoke-matched to NIK (v%10<8 keep mask into GROG), K=5 | 1368 of 1710, 80%, matches NIK-outcoil exactly |
| GRASP-Pro cs_img | as f100 but 5 spokes/frame (342 frames), K=5 | 1710 of 1710, 100% |
| GRASP-v2 (classic, 2014) | MCNUFFT + temporal TV only, NO subspace, lam=0.25*max|x0| (published, untuned) | all acquired, ~1680-1710 of 1710 depending on binning remainder |
| model-free NUFFT | gridding + ramp DCF, NO regularization, 31-spoke SLIDING window (~6.8 s effective temporal resolution) | all 1710, 240 output samples at 1.54 s spacing |"""

def provenance_cells():
    """methods table. every result statement below names its method AND spoke budget; this is the key."""
    return [nbf.v4.new_markdown_cell(
        "## methods and spoke budgets\n\n"
        "every result in this notebook names the method and the spokes it used. read this table first: "
        "several names are misleading on their own. 'CS-file' is a recon shipped inside the phantom "
        "simulation, not something reconstructed here, and it is NOT spoke-matched. the NIK run labelled "
        "'UNMATCHED 70%' is named 'full' on disk but trained on a random 70% with 30% held out.\n\n"
        + PROV_TABLE)]


def main():
    nb = nbf.read(SRC, as_version=4)
    for c in nb.cells:
        c.source = rewrite(c.source)
        if c.cell_type == "code":
            c.outputs = []; c.execution_count = None
    nb.cells = drop_k_cells(nb.cells)
    nb.cells[2:2] = provenance_cells()
    nb.cells.insert(1, nbf.v4.new_markdown_cell(
        "reference method is **classic grasp (v2, feng/otazo 2014)**, not grasp-pro, in BOTH sections. "
        "identical inputs to the grasp-pro notebook (same slices, same spokes, same binning, same b1, "
        "same coil maps); only the recon algorithm differs. roi anatomy is shared with the grasp-pro "
        "notebook so the two are directly comparable."))
    # frontier section goes just before the summary
    fc = frontier_cells()
    si = next((i for i, c in enumerate(nb.cells) if c.source.lstrip().startswith("# summary")), len(nb.cells))
    nb.cells[si:si] = fc
    print(f"inserted {len(fc)} frontier cells at {si}")
    if "--no-exec" not in sys.argv:
        print("executing...", flush=True)
        os.chdir(B)
        NotebookClient(nb, timeout=1200, kernel_name="python3",
                       resources={"metadata": {"path": B}}).execute()
    nbf.write(nb, DST)
    print("WROTE", DST)


if __name__ == "__main__":
    main()
