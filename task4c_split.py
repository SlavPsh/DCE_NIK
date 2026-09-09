"""TASK 4C Part 2: permanent train/val/test SPOKE split on the Task-4 motion-free f25 dataset.
train = existing Task-4 f25 input spokes (keep_f25 = angles 0,1) -> UNCHANGED.
non-training spokes (angles 2-8) split deterministically by ALTERNATING angle index within each
temporal block (frame): val = {2,4,6,8}, test = {3,5,7}. Whole spokes preserved (never split
individual readout samples). Saves spoke arrays + masks + manifests, verifies disjointness."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, csv, json
T4 = "/net/beegfs/users/P101440/DCE_NIK/results/task4_xcat_nomotion_pilot"
OUT = "/net/beegfs/users/P101440/DCE_NIK/results/task4c_nik_capacity_audit"
S = np.load(f"{T4}/arrays/sim.npz"); kx = S["kx"]; ky = S["ky"]; keep = S["keep_f25"]  # keep [F,9] bool
F, NA, RO = kx.shape
train_mask = keep.copy()                                   # angles 0,1 (EXACT Task-4 f25 input)
nontrain = ~train_mask                                     # angles 2..8
val_mask = np.zeros((F, NA), bool); test_mask = np.zeros((F, NA), bool)
for t in range(F):
    idx = np.where(nontrain[t])[0]                         # non-training angle indices this frame
    for j, a in enumerate(idx):                            # alternating stratified within the block
        (val_mask if j % 2 == 0 else test_mask)[t, a] = True
# disjointness
assert not (train_mask & val_mask).any() and not (train_mask & test_mask).any() and not (val_mask & test_mask).any()
assert (train_mask | val_mask | test_mask).sum() == F * NA
def spokes(m): return np.array([[t, a] for t in range(F) for a in range(NA) if m[t, a]], int)
tr, va, te = spokes(train_mask), spokes(val_mask), spokes(test_mask)
np.save(f"{OUT}/arrays/train_spokes.npy", tr); np.save(f"{OUT}/arrays/validation_spokes.npy", va); np.save(f"{OUT}/arrays/test_spokes.npy", te)
np.savez(f"{OUT}/arrays/spoke_masks.npz", train=train_mask, val=val_mask, test=test_mask)
# distributions
def angle_deg(m):
    ang = []
    for t, a in spokes(m): ang.append(np.degrees(np.arctan2(ky[t, a].mean(), kx[t, a].mean())) % 180)
    return np.array(ang)
def manifest_row(name, m, sp):
    ad = angle_deg(m)
    return dict(set=name, n_complete_spokes=int(m.sum()), frac_of_all=round(float(m.sum()) / (F * NA), 4),
                spokes_per_frame=int(m.sum(1).mean()), frames_covered=int((m.sum(1) > 0).sum()),
                angle_mean_deg=round(float(ad.mean()), 1), angle_std_deg=round(float(ad.std()), 1),
                kz_planes=1, coil_coverage="all-8-per-spoke")
rows = [manifest_row("train", train_mask, tr), manifest_row("validation", val_mask, va), manifest_row("test", test_mask, te)]
with open(f"{OUT}/spoke_split_manifest.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); [w.writerow(r) for r in rows]
# temporal distribution (spokes per frame is constant for each set) + overlap record
meta = dict(train_angles=sorted(set(int(a) for _, a in tr)), val_angles=sorted(set(int(a) for _, a in va)),
            test_angles=sorted(set(int(a) for _, a in te)),
            train_val_overlap=int((train_mask & val_mask).sum()), train_test_overlap=int((train_mask & test_mask).sum()),
            val_test_overlap=int((val_mask & test_mask).sum()), F=int(F), NA=int(NA), RO=int(RO),
            train_matches_task4_f25=bool((train_mask == keep).all()))
json.dump(meta, open(f"{OUT}/arrays/split_meta.json", "w"), indent=1)
print("split done:", {r["set"]: r["n_complete_spokes"] for r in rows})
print("angles: train", meta["train_angles"], "val", meta["val_angles"], "test", meta["test_angles"])
print("overlaps (must be 0):", meta["train_val_overlap"], meta["train_test_overlap"], meta["val_test_overlap"])
print("train == Task-4 f25 keep:", meta["train_matches_task4_f25"])
