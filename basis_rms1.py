"""unit-rms copies of the tofts basis files: atoms * sqrt(G) (rms 1 instead of 1/sqrt(G)), R_patlak scaled the same so patlak_to_tofts stays
exact. same span, same everything else; the amplitude head then needs the O(1) coefficient scale the patlak class uses (amp track 2026-09-16).
usage: python basis_rms1.py results/tofts_vs_patlak/basis_sl21.npz [...]  -> <name>_rms1.npz next to it"""
import sys, os, numpy as np
for src in sys.argv[1:]:
    dst = src[:-4] + "_rms1.npz"
    if os.path.exists(dst): print("exists", dst); continue
    z = np.load(src, allow_pickle=True); d = {k: z[k] for k in z.files}; G = d["atoms"].shape[0]; f = float(np.sqrt(G))
    d["atoms"] = (d["atoms"] * f).astype(np.float32); d["R_patlak"] = (d["R_patlak"] * f).astype(np.float32); d["atom_rms"] = np.float32(1.0)
    np.savez(dst + ".tmp.npz", **d); os.replace(dst + ".tmp.npz", dst); print("wrote", dst, "rms", float(np.sqrt((d["atoms"] ** 2).mean(0)).mean()))
