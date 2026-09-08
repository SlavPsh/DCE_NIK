"""single source of truth for WHAT each method in the notebooks actually is.

every result statement, table and figure legend should draw its label from here. the ambiguity this
replaces is not cosmetic: "results_spoke_full_slice21" is named "full" and its script comment says
"all spokes", but it trained on a random 70% with 30% held out, and that mislabel produced an unfair
in-vivo comparison. labels below state the spoke budget explicitly, always.

fields: method, detail (algorithm + key hyperparameters), spokes (count, % of acquired, heldout), src
"""
NTV_REAL = 1710          # acquired spokes, real in-vivo slice 21
NTV_PHAN = 7             # angles per frame, phantom (TRAIN 5 / VAL 1 / TEST 1)

REG = {
 # ---------------- phantom, XCAT no-motion, slice ZI=15 ----------------
 "phantom/NIK-F0":    dict(method="NIK-F0", detail="wire_ff_patlak, patlak-free 0 (Patlak basis, no free atoms)",
                           spokes="5 of 7 angles/frame (TRAIN_ANG), 71%", src="checkpoints/w768_ks2.5_s*"),
 "phantom/NIK-sub5":  dict(method="NIK-sub5", detail="wire_ff_subspace, rank 5",
                           spokes="5 of 7 angles/frame (TRAIN_ANG), 71%", src="checkpoints/sub5_w768_s*"),
 "phantom/NIK-sub12": dict(method="NIK-sub12", detail="wire_ff_subspace, rank 12",
                           spokes="5 of 7 angles/frame (TRAIN_ANG), 71%", src="checkpoints/sub12_w768_s*"),
 "phantom/NIK-sub16": dict(method="NIK-sub16", detail="wire_ff_subspace, rank 16, 2-seed complex average",
                           spokes="5 of 7 angles/frame (TRAIN_ANG), 71%", src="checkpoints/sub16_w768_s*"),
 "phantom/NIK-free":  dict(method="NIK-free", detail="wire_ff, full rank (no subspace), 3-seed complex average",
                           spokes="5 of 7 angles/frame (TRAIN_ANG), 71%", src="checkpoints/free_w768_s*"),
 "phantom/GRASP-Pro": dict(method="GRASP-Pro", detail="NUFFT-SENSE + temporal PCA subspace K*=12 (held-out CV) + spatial&temporal TV",
                           spokes="5 of 7 angles/frame (TRAIN_ANG), 71%", src="arrays/grasp_recon.npz"),
 "phantom/GRASP-v2":  dict(method="GRASP-v2 (classic, 2014)", detail="MCNUFFT + temporal TV only, NO subspace, lam=0.25*max|x0| (published, untuned)",
                           spokes="5 of 7 angles/frame (TRAIN_ANG), 71%; sweep groups G frames -> 5G/frame",
                           src="arrays/grasp_v2_recon.npz, v2_sweep/v2_G*.npy"),
 "phantom/CS-file":   dict(method="CS-file", detail="recon BUNDLED in the XCAT sim HDF5; algorithm not in our code, cannot be verified",
                           spokes="7 of 7 angles/frame, 100%, NOT spoke-matched (+40% vs NIK/GRASP)",
                           src="simulation_results_*.mat results/images/Recon/img"),
 # ---------------- real in-vivo, slice 21 ----------------
 "real/NIK-outcoil":  dict(method="NIK (output-coil)", detail="outcoil_real.py, subspace rank 16, coil on output heads",
                           spokes=f"v%10<8 = 1368 of {NTV_REAL}, 80%; val/test spokes HELD OUT",
                           src="results/realdata_* (sections 3-8)"),
 "real/NIK-full70":   dict(method="NIK (full-rank, UNMATCHED 70%)", detail="wire_ff_res, default --subsample-frac 0.7",
                           spokes=f"random 1197 of {NTV_REAL}, 70%; 30% HELD OUT. name says 'full' but is NOT all spokes",
                           src="results_spoke_full_slice21"),
 "real/NIK-matched":  dict(method="NIK (full-rank, spoke-matched)", detail="wire_ff_res, --spoke-keep-file keep_f100, no heldout",
                           spokes=f"1708 of {NTV_REAL}, 99.9%, same input as GRASP",
                           src="results_full_sl21_matched"),
 "real/GRASP-Pro-f100": dict(method="GRASP-Pro f100", detail="GROG + temporal PCA K=5 + spatial TV 0.001 + temporal TV 0.0005, 14 spokes/frame (122 frames)",
                           spokes=f"1708 of {NTV_REAL}, 100% of binned", src="results_spoke_cs/cs_slice21_f100.npy"),
 "real/GRASP-Pro-f80": dict(method="GRASP-Pro f80-match", detail="as f100 but spoke-matched to NIK (v%10<8 keep mask into GROG), K=5",
                           spokes=f"1368 of {NTV_REAL}, 80%, matches NIK-outcoil exactly", src="results_spoke_cs/cs_slice21_f80match.npy"),
 "real/GRASP-Pro-p05": dict(method="GRASP-Pro cs_img", detail="as f100 but 5 spokes/frame (342 frames), K=5",
                           spokes=f"1710 of {NTV_REAL}, 100%", src="results_ref/slice_21.npz['cs_img']"),
 "real/GRASP-v2":     dict(method="GRASP-v2 (classic, 2014)", detail="MCNUFFT + temporal TV only, NO subspace, lam=0.25*max|x0| (published, untuned)",
                           spokes=f"all acquired, ~1680-1710 of {NTV_REAL} depending on binning remainder",
                           src="grasp_v2/results_grasp_v2/gv2_slice21_*.npy"),
 "real/model-free":   dict(method="model-free NUFFT", detail="gridding + ramp DCF, NO regularization, 31-spoke SLIDING window (~6.8 s effective temporal resolution)",
                           spokes=f"all {NTV_REAL}, 240 output samples at 1.54 s spacing",
                           src="step2_slice21.npz['mf']"),
}

def label(key, short=False):
    r = REG[key]
    return r["method"] if short else f"{r['method']} [{r['spokes']}]"

def table_rows(prefix=None):
    return [(k, REG[k]["method"], REG[k]["detail"], REG[k]["spokes"], REG[k]["src"])
            for k in REG if prefix is None or k.startswith(prefix)]

if __name__ == "__main__":
    for k, m, d, s, src in table_rows():
        print(f"{k:24} {m:34} {s}")
