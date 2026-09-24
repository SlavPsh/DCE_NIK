"""spoke masks for a dataset from its view count: keep = views with v%10 < 8 (the k80 standard), val = v%10 == 8, test = v%10 == 9; view index
v runs over the ntviews kept after the Cut (same convention as the p3 masks: keep_f80match / val_k80_m8 / test_k80_m9 over 1708 views).
usage: DCE_DS=p14 python spoke_masks_build.py   -> spoke_masks/keep_k80_p14.npy, val_k80_m8_p14.npy, test_k80_m9_p14.npy"""
import numpy as np, sys
sys.path.insert(0, "/net/beegfs/users/P101440/DCE_NIK"); import dsp
assert dsp.DS != "p3", "p3 masks exist under their historical names"
v = np.arange(dsp.NTV); keep, val, test = v[v % 10 < 8], v[v % 10 == 8], v[v % 10 == 9]
for p, m in ((dsp.KEEP, keep), (dsp.VAL, val), (dsp.TEST, test)): np.save(p, m.astype(np.int64)); print(p, m.size)
print(f"{dsp.DS}: ntviews {dsp.NTV}, keep {keep.size} val {val.size} test {test.size}, TA {dsp.TA}, nx {dsp.NX}, bas {dsp.BAS}")
