"""seed average of a pk arm's recons (magnitude mean over seeds): free noise reduction, curves unchanged (seeds agree to 1 to 2%).
writes <iv>/<arm>_sl<Z>_avg<n>/nik_slice_<Z>_cplx.npy (magnitude stored as complex64 so every reader works unchanged).
usage: python avg_seeds.py --iv invivo_k80_rms1 --arm tofts8 --slice 21 --seeds 0,1,2"""
import os, argparse, numpy as np
B = "/net/beegfs/users/P101440/DCE_NIK/results/tofts_vs_patlak"
ap = argparse.ArgumentParser(); ap.add_argument("--iv", default="invivo_k80_rms1"); ap.add_argument("--arm", default="tofts8"); ap.add_argument("--slice", type=int, default=21); ap.add_argument("--seeds", default="0,1,2"); a = ap.parse_args()
S = [int(x) for x in a.seeds.split(",")]; v = [np.abs(np.load(f"{B}/{a.iv}/{a.arm}_sl{a.slice}_s{s}/nik_slice_{a.slice}_cplx.npy")).astype(np.float32) for s in S]
m = np.mean(v, 0); out = f"{B}/{a.iv}/{a.arm}_sl{a.slice}_avg{len(S)}"; os.makedirs(out, exist_ok=True); np.save(f"{out}/nik_slice_{a.slice}_cplx.npy", m.astype(np.complex64))
sd = np.mean([np.linalg.norm(x - m) / np.linalg.norm(m) for x in v]); print(f"wrote {out}: seeds {S}, mean relative seed deviation {sd:.3f}")
