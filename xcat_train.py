"""TASK 3 / B4: train NIK on the XCAT sim (validated adapter) and measure bolus timing
ACCURACY vs the known truth (TTP 27.7s, FWHM 46.8s). same fixed recipe as the real-data arm.
out: results_xcat_<tag>/{nik_slice.npy, metrics.json}"""
import argparse, os, json, time, sys
import numpy as np, torch
import torch.nn.functional as F
torch.set_float32_matmul_precision("high")
sys.path.insert(0, "/scratch/rnga/vvpshenov/grasp_pro_py")
import xcat_adapter as X
from nik_model import (WIRE_FF_RES_KXY_COIL_T_REIM, WIRE_FF_SUBSPACE_KXY_COIL_T_REIM, warmstart_phi)
from kspace_normalization import KSpaceNormalizer
from nik_focal_loss import composable_kspace_loss

FIX = dict(hidden=512, depth=12, w0=62.0, s0=15.0, k_freq=256, k_sigma=2.5,
           t_freq=32, t_sigma=1.5, coil_embed_dim=8, env=0.75)


def xcat_pca_phi(d, rank):
    """XCAT navigator PCA for the Phi warm start. temporal cov [nt,nt] from 5-readout x ncc
    observations (mirrors compute_pca_phi); NOT a coil cov. -> Phi [nt, rank]."""
    dd = X.slice_radial(d["meta"]["zi"]); kd = dd["kdata"]; C, Fr, NA, RO = kd.shape; c0 = RO // 2
    kdc = kd.transpose(3, 1, 2, 0).reshape(RO, Fr * NA, C)          # [RO, nspokes, C] frame-major
    nav = np.abs(kdc[c0 - 2:c0 + 3, :Fr * NA, :]).reshape(5, NA, Fr, C, order="F").mean(1)  # [5, Fr, C]
    ds = nav.transpose(0, 2, 1).reshape(5 * C, Fr, order="F")       # [5C observations, Fr]
    w, PC = np.linalg.eigh(np.cov(ds, rowvar=False))               # [Fr, Fr]
    Phi = PC[:, np.argsort(-w)][:, :rank].astype(np.complex64)     # [Fr, rank]
    return d["frame_t"].astype(np.float32), Phi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["wire_ff_res", "wire_ff_subspace"])
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--slice", type=int, default=5)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev} model={args.model} rank={args.rank} slice={args.slice}", flush=True)

    d = X.make_xcat_dataset(args.slice, device=dev)
    x, t, c, yraw, spoke = d["x_all"], d["t_all"], d["coil_all"], d["y_all_raw"], d["spoke_id_all"]
    ncc = d["meta"]["ncc"]
    uniq = torch.unique(spoke); g = torch.Generator(device=dev).manual_seed(args.seed)
    perm = uniq[torch.randperm(uniq.numel(), generator=g, device=dev)]
    ntr = max(1, int(uniq.numel() * 0.7))
    trm = torch.isin(spoke, perm[:ntr]); tri = torch.where(trm)[0]; hei = torch.where(~trm)[0]
    dcf = torch.ones(x.shape[0], device=dev)
    nz = KSpaceNormalizer(); nz.fit(x[tri], yraw[tri], dcf=dcf[tri], envelope_exponent=FIX["env"])
    y = nz.normalize(x, yraw)
    xt, tt, ct, yt = x[tri], t[tri], c[tri], y[tri]; N = xt.shape[0]
    xh, th, ch, yh = x[hei], t[hei], c[hei], y[hei]

    torch.manual_seed(args.seed)
    if args.model == "wire_ff_res":
        model = WIRE_FF_RES_KXY_COIL_T_REIM(n_coils=ncc, coil_embed_dim=FIX["coil_embed_dim"],
                hidden=FIX["hidden"], depth=FIX["depth"], w0=FIX["w0"], s0=FIX["s0"], k_freq=FIX["k_freq"],
                k_sigma=FIX["k_sigma"], t_freq=FIX["t_freq"], t_sigma=FIX["t_sigma"]).to(dev)
    else:
        model = WIRE_FF_SUBSPACE_KXY_COIL_T_REIM(n_coils=ncc, coil_embed_dim=FIX["coil_embed_dim"],
                rank=args.rank, hidden=FIX["hidden"], depth=FIX["depth"], w0=FIX["w0"], s0=FIX["s0"],
                k_freq=FIX["k_freq"], k_sigma=FIX["k_sigma"], t_freq=FIX["t_freq"], t_sigma=FIX["t_sigma"],
                residual=True).to(dev)
        ft, phi = xcat_pca_phi(d, args.rank)
        err = warmstart_phi(model, ft, phi, steps=800, lr=1e-3, device=dev.type)
        print(f"  warmstart Phi<-XCAT navigator PCA (fit MSE {err:.3e})", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=3e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=30, min_lr=1e-7)
    best, best_state = float("inf"), None
    t0 = time.time(); model.train()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, N, (65536,), device=dev)
        opt.zero_grad(set_to_none=True)
        loss = composable_kspace_loss(model(xt[idx], tt[idx], ct[idx]), yt[idx], dcf=torch.ones(65536, device=dev),
                                      use_dcf=False, dcf_power=0.0, use_focal=False, return_diagnostics=False)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 1000 == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                hp = torch.cat([model(xh[i:i+262144], th[i:i+262144], ch[i:i+262144]) for i in range(0, xh.shape[0], 262144)], 0)
                hl = float(F.mse_loss(hp, yh).item())
            model.train(); sched.step(hl)
            if step >= 2000 and hl < best: best = hl; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"    step {step:6d}  train {float(loss):.3e}  heldout {hl:.3e}  ({time.time()-t0:.0f}s)", flush=True)
    if best_state: model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})

    img = X.xcat_reconstruct(model, nz, d["meta"], d["b1"], d["frame_t"], device=dev.type)  # [RO,RO,181]
    np.save(f"{args.save_dir}/nik_slice.npy", img)
    # bolus accuracy vs truth
    roi = X.aorta_roi(X.slice_radial(args.slice)); tim = X.slice_radial(args.slice)["times"]
    cur = np.array([img[..., i][roi].mean() for i in range(img.shape[-1])])
    b = cur[tim < 12].mean(); n = (cur - b) / (cur.max() - b + 1e-9)
    ttp = float(tim[np.argmax(n)]); half = (n > 0.5) & (tim < ttp + 40)
    fwhm = float(tim[half].max() - tim[half].min())
    tr = X.true_kinetics(zi=args.slice)
    res = dict(model=args.model, rank=args.rank, ttp=ttp, fwhm=fwhm, ttp_true=tr["ttp"], fwhm_true=tr["fwhm"],
               ttp_err=ttp - tr["ttp"], fwhm_err=fwhm - tr["fwhm"], heldout=best)
    json.dump(res, open(f"{args.save_dir}/metrics.json", "w"), indent=1, default=float)
    print(f"XCAT {args.model}: TTP {ttp:.1f}s (true {tr['ttp']:.1f}, err {ttp-tr['ttp']:+.1f}) | "
          f"FWHM {fwhm:.1f}s (true {tr['fwhm']:.1f}, err {fwhm-tr['fwhm']:+.1f})", flush=True)


if __name__ == "__main__":
    main()
