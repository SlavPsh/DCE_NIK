"""TASK 4C Part 1: re-evaluate the EXISTING Task-4 NIK-F0 f25 seed-0 checkpoint on the new
train/val/test masks + truth, to confirm the Task-4 result reproduces before training new models.
Reference Task-4 numbers: aorta recovery 0.495, aorta-curve NRMSE 0.267, intAIF NRMSE 0.130,
held-out(coef-forward) k-NMSE 0.0037."""
import warnings; warnings.filterwarnings("ignore")
import json, numpy as np, torch, sys
import task4c_common as K
sys.path.insert(0, "/scratch/rnga/vvpshenov/DCE_NIK")
CK = "/scratch/rnga/vvpshenov/DCE_NIK/results/task4_xcat_nomotion_pilot/checkpoints/nik_F0_f25_seed0.pt"
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
S, M = K.load()
_, _, _, _, nz, dims = K.build_train(S, dev); C = dims[3]
model = K.make_model(512, 0, C, dev)
ck = torch.load(CK, map_location=dev, weights_only=False)
missing, unexpected = model.load_state_dict(ck["state_dict"], strict=False)
model.eval()
print(f"loaded existing checkpoint (missing {len(missing)}, unexpected {len(unexpected)} keys)", flush=True)
thC = K.extract_pathC(model, nz, S, dev); tm = K.truth_metrics(thC, S)
knm = {s: K.kspace_nmse(model, nz, S, M[s], dev)["nmse"] for s in ("train", "val", "test")}
# combined val+test as the closest analogue to Task-4 held-out (direct model-query estimator)
vt = np.zeros_like(M["val"]); vt |= M["val"] | M["test"]
knm["val+test(direct-query)"] = K.kspace_nmse(model, nz, S, vt, dev)["nmse"]
ref = dict(aorta_recovery=0.495, aorta_curve=0.267, intAIF=0.130, heldout_coefforward=0.0037)
out = dict(reproduced=tm, kspace_nmse=knm, task4_reference=ref,
           delta_aorta_recovery=tm["R_aorta"] - ref["aorta_recovery"], delta_intAIF=tm["intAIF_nrmse"] - ref["intAIF"],
           delta_aorta_curve=tm["aorta_curve_nrmse"] - ref["aorta_curve"])
json.dump(out, open(f"{K.OUT}/arrays/part1_reproduction.json", "w"), indent=1, default=float)
print("REPRO aorta_recovery %.3f (task4 0.495) | curve %.3f (0.267) | intAIF %.3f (0.130) | FP %.3f" % (
    tm["R_aorta"], tm["aorta_curve_nrmse"], tm["intAIF_nrmse"], tm["FP_energy"]))
print("k-space NMSE:", {k: round(v, 5) for k, v in knm.items()})
