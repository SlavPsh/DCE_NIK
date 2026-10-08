"""joint-z nik-tofts (plan item 3, 2026-10-08): ONE model over several slices, z (image-domain slice position, normalized to [-1, 1] over the
slices in the fit) as an extra input of the coefficient backbone A_r(kx, ky, z, coil); the temporal atoms stay fixed and shared across z (one basis
file for every slice). sharing mechanism = a low-bandwidth fourier encoding of z (z_sigma, cycles per unit) with shared weights; controls: a
categorical per-slice embedding (--z-mode embed, no smoothness) and z_sigma variants. everything per slice stays per slice: k-space normalizer,
support mask (prior rendered per z), coil maps and crop (final render per z). compute matched to single-slice runs: per step one micro-batch of
--batch-size samples PER SLICE (gradients accumulated), --steps steps. separate trainer and model class: train_grasp_nik.py and the production
classes are untouched. a --query-slices slice is never trained on: its render at its z is the interpolation test.
out: <out>/tofts8_sl<Z>_s<seed>/nik_slice_<Z>_cplx.npy per slice (the production eval layout), <out>/model_joint.pt, <out>/train.log
usage: DCE_DS=p3 python train_joint_z.py --slices 18,19 --query-slices 20 --basis results/tofts_vs_patlak/basis_sl19_r8_rms1.npz --out <dir> [--z-sigma 1 --z-mode ff --hidden 512]"""
import os, sys, time, json, argparse, numpy as np, torch, torch.nn as nn
sys.path.insert(0, '/net/beegfs/users/P101440/DCE_NIK'); sys.path.insert(0, '/net/beegfs/users/P101440/grasp_pro_py')
import dsp, nik_adapter as A
from nik_model import WIRE_FF_TOFTS_KXY_COIL_T_REIM, FourierFeatures, GaborLayer
from kspace_normalization import KSpaceNormalizer, compute_dcf_radial
from nik_focal_loss import composable_kspace_loss
from nik_output_recon import recon_nik_cart


class JointZTofts(WIRE_FF_TOFTS_KXY_COIL_T_REIM):
    """tofts atoms fixed and shared; coefficient backbone takes (kx, ky, z, coil). z_mode ff: one fourier-feature layer over (kx, ky, z) with the
    z rows scaled to z_sigma (per-dimension bandwidth); embed: fourier features of (kx, ky) plus a learned per-slice embedding (no z smoothness)."""
    def __init__(self, n_coils, atom_tgrid, atoms, R_patlak, z_sigma=1.0, z_mode='ff', n_z=3, z_embed_dim=8, **kw):
        super().__init__(n_coils, atom_tgrid, atoms, R_patlak, coil_mode='input', **kw)
        self.z_mode = z_mode; k_freq = int(self.ff_k.B.shape[1]); k_sigma = float(kw.get('k_sigma', 2.5)); seed = int(kw.get('ff_seed', 0))
        if z_mode == 'ff':
            self.ff_k = FourierFeatures(3, n_freq=k_freq, sigma=k_sigma, seed=seed)
            with torch.no_grad(): self.ff_k.B[2] *= (z_sigma / k_sigma)                                   # z bandwidth independent of the k bandwidth
        else:
            self.z_embed = nn.Embedding(int(n_z), int(z_embed_dim)); nn.init.uniform_(self.z_embed.weight, -1.0, 1.0)
            hidden = self.a_head.in_features // 2
            self.a_first = GaborLayer(self.a_first.linear.in_features + int(z_embed_dim), hidden, w0=self.a_first.w0, s0=self.a_first.s0, is_first=True)

    def amplitudes(self, kcoords, coil_idx):
        ec = self.coil_embed(coil_idx.long())
        if self.z_mode == 'ff': feats = [self.ff_k(kcoords), ec]
        else: feats = [self.ff_k(kcoords[:, :2]), ec, self.z_embed(kcoords[:, 2].round().long())]
        h = self.a_first(torch.cat(feats, dim=-1))
        for blk in self.a_blocks:
            h = h + blk(h) if self.residual else blk(h)
        return self.a_head(h).view(-1, self.rank, 2)


class FixZ(nn.Module):
    """the joint model at one z, with the single-slice signature (kcoords (N,2), t, coil) for reconstruct_cartesian and the diagnostics"""
    def __init__(self, model, zval): super().__init__(); self.model = model; self.zval = float(zval)
    def forward(self, k2, t, c): return self.model(torch.cat([k2, torch.full((k2.shape[0], 1), self.zval, device=k2.device, dtype=k2.dtype)], 1), t, c)
    def amplitudes(self, k2, c): return self.model.amplitudes(torch.cat([k2, torch.full((k2.shape[0], 1), self.zval, device=k2.device, dtype=k2.dtype)], 1), c)
    @property
    def rank(self): return self.model.rank
    def basis(self, t): return self.model.basis(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--slices', required=True); ap.add_argument('--query-slices', default=''); ap.add_argument('--basis', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--z-sigma', type=float, default=1.0); ap.add_argument('--z-mode', default='ff', choices=['ff', 'embed']); ap.add_argument('--z-spacing-mm', type=float, default=1.0, help='only the ratio matters; z normalized over the span')
    ap.add_argument('--hidden', type=int, default=512); ap.add_argument('--depth', type=int, default=12); ap.add_argument('--w0', type=float, default=62.0); ap.add_argument('--s0', type=float, default=15.0)
    ap.add_argument('--k-freq', type=int, default=256); ap.add_argument('--k-sigma', type=float, default=2.5); ap.add_argument('--t-freq', type=int, default=32); ap.add_argument('--t-sigma', type=float, default=1.5)
    ap.add_argument('--coil-embed-dim', type=int, default=8); ap.add_argument('--steps', type=int, default=10000); ap.add_argument('--batch-size', type=int, default=65536); ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--weight-decay', type=float, default=0.01); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--ff-seed', type=int, default=0)
    ap.add_argument('--spoke-keep-file', default=None); ap.add_argument('--envelope-exponent', type=float, default=0.75); ap.add_argument('--dcf-power', type=float, default=0.0)
    ap.add_argument('--support-weight', type=float, default=1.0); ap.add_argument('--support-every', type=int, default=32); ap.add_argument('--support-dilate', type=int, default=6); ap.add_argument('--support-chunk', type=int, default=16384)
    ap.add_argument('--support-radius', type=float, default=1.0); ap.add_argument('--console-every', type=int, default=1000)
    args = ap.parse_args(); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); torch.manual_seed(args.seed); t0 = time.time()
    REFD = dsp.REF; sh = A.load_shared(REFD); nx, bas = int(sh['nx']), int(sh['bas']); os.makedirs(args.out, exist_ok=True)
    SL = [int(z) for z in args.slices.split(',')]; QS = [int(z) for z in args.query_slices.split(',') if z]; ALL = sorted(set(SL + QS))
    zc = 0.5 * (min(ALL) + max(ALL)); zh = max(0.5 * (max(ALL) - min(ALL)), 1e-6); Z2z = {Z: (Z - zc) / zh for Z in ALL}               # z in [-1, 1] over the span
    zi_of = {Z: i for i, Z in enumerate(SL)}                                                                                             # categorical index (embed mode); a query slice gets its nearest trained index
    print(f'joint-z tofts: train {SL} query {QS}; z {Z2z}; mode {args.z_mode} sigma {args.z_sigma} hidden {args.hidden}; device {device}', flush=True)
    bz = np.load(args.basis); atoms = np.asarray(bz['atoms']); print(f'basis {args.basis}: {atoms.shape[1]} atoms', flush=True)
    model = JointZTofts(n_coils=int(sh['ncc']), atom_tgrid=bz['tgrid_model'], atoms=atoms, R_patlak=bz['R_patlak'], z_sigma=args.z_sigma, z_mode=args.z_mode, n_z=len(SL),
                        coil_embed_dim=args.coil_embed_dim, hidden=args.hidden, depth=args.depth, w0=args.w0, s0=args.s0, k_freq=args.k_freq, k_sigma=args.k_sigma,
                        t_freq=args.t_freq, t_sigma=args.t_sigma, ff_seed=args.ff_seed, residual=True).to(device)
    print(f'params {sum(p.numel() for p in model.parameters())}', flush=True)
    # per-slice data, normalizer, dcf; pooled tensors with a z column
    pools = {}; norms = {}; b1s = {}; ncc = int(sh['ncc'])
    kept = torch.as_tensor(np.load(args.spoke_keep_file), device=device) if args.spoke_keep_file else None
    for Z in SL:
        ds = A.make_radial_dataset(REFD, Z, compute_device=device, shared=sh); x, t, c, y_raw, sid = ds['x_all'], ds['t_all'], ds['coil_all'], ds['y_all_raw'], ds['spoke_id_all']
        tr = torch.where(torch.isin(sid, kept.to(sid.dtype)))[0] if kept is not None else torch.arange(x.shape[0], device=device)
        dcf = compute_dcf_radial(x, method='simple_ramp'); nz = KSpaceNormalizer(); nz.fit(x[tr], y_raw[tr], dcf=dcf[tr], envelope_exponent=args.envelope_exponent); y = nz.normalize(x, y_raw)
        zcol = torch.full((tr.numel(), 1), Z2z[Z] if args.z_mode == 'ff' else float(zi_of[Z]), device=device, dtype=x.dtype)
        pools[Z] = dict(x=torch.cat([x[tr], zcol], 1), t=t[tr], c=c[tr], y=y[tr], w=dcf[tr]); norms[Z] = nz; b1s[Z] = ds['b1']
        print(f'  slice {Z}: {tr.numel()} samples ({int(torch.unique(sid[tr]).numel())} spokes), z {zcol[0, 0].item():.3f}', flush=True)
    # support prior per slice (mask rot180 as in production), rendered at that slice's z with its own normalizer
    _s0 = (nx - bas) // 2; crop = torch.zeros(nx, nx, dtype=torch.bool, device=device); crop[_s0:_s0 + bas, _s0:_s0 + bas] = True
    masks = {Z: torch.flip(torch.from_numpy(np.load(dsp.SUPPORT(Z, args.support_dilate)).astype(bool)).to(device), (0, 1)) for Z in SL}
    grid2 = torch.from_numpy(A.cartesian_grid(nx)).to(device); rmask = (torch.sqrt((grid2 ** 2).sum(1)) <= 1.0).float().view(nx, nx, 1)
    from torch.utils.checkpoint import checkpoint as _ckpt
    def support_backward(scale):
        tot = 0.0; Rk = int(model.rank)
        for Z in SL:
            zval = Z2z[Z] if args.z_mode == 'ff' else float(zi_of[Z]); g3 = torch.cat([grid2, torch.full((grid2.shape[0], 1), zval, device=device)], 1); nzZ = norms[Z]; m = masks[Z]; air = crop & ~m
            def _coil_k(cc, cidx):
                cd = torch.full((cc.shape[0],), cidx, dtype=torch.long, device=device); Amp = model.amplitudes(cc, cd)
                return torch.stack([nzZ.denormalize(cc[:, :2], Amp[:, r, :]) for r in range(Rk)], 1)
            for cidx in range(ncc):
                pr = torch.cat([_ckpt(_coil_k, g3[i:i + args.support_chunk], cidx, use_reentrant=False) for i in range(0, g3.shape[0], args.support_chunk)], 0)
                K = torch.complex(pr[..., 0], pr[..., 1]).view(nx, nx, Rk) * rmask
                im = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(K, dim=(0, 1)), dim=(0, 1)), dim=(0, 1)); E = im.abs() ** 2
                pen = (E[air].sum(0) / (E[m].sum(0) + 1e-12)).sum() / (ncc * Rk * len(SL)); (scale * pen).backward(); tot += float(pen)
        return tot
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay); model.train(); sup_last = 0.0; log = []
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True); ltot = 0.0
        for Z in SL:                                                                                       # one micro-batch per slice, gradients accumulated (compute matched)
            P = pools[Z]; idx = torch.randint(0, P['x'].shape[0], (args.batch_size,), device=device)
            pred = model(P['x'][idx], P['t'][idx], P['c'][idx])
            loss = composable_kspace_loss(pred, P['y'][idx], dcf=P['w'][idx], use_dcf=True, dcf_power=args.dcf_power, use_focal=False, focal_warmup_progress=1.0, return_diagnostics=False) / len(SL)
            loss.backward(); ltot += float(loss)
        if args.support_weight > 0 and step % args.support_every == 0: sup_last = support_backward(args.support_weight * args.support_every)
        opt.step()
        if step % args.console_every == 0 or step == args.steps:
            print(f'    step {step:6d}  train {ltot:.3e}  support {sup_last:.3e}  ({time.time() - t0:.0f} s)', flush=True); log.append(dict(step=step, loss=ltot, support=sup_last, wall=time.time() - t0))
    model.eval(); torch.save(dict(state_dict={k: v.cpu() for k, v in model.state_dict().items()}, args=vars(args), z=Z2z, slices=SL, query=QS, basis=args.basis), os.path.join(args.out, 'model_joint.pt'))
    # per-z render: trained slices with their own normalizer and b1; a query slice with its own normalizer (fit on its data, scaling only) and b1
    for Z in ALL:
        if Z not in norms:
            ds = A.make_radial_dataset(REFD, Z, compute_device=device, shared=sh); x, y_raw, sid = ds['x_all'], ds['y_all_raw'], ds['spoke_id_all']
            tr = torch.where(torch.isin(sid, kept.to(sid.dtype)))[0] if kept is not None else torch.arange(x.shape[0], device=device)
            nz = KSpaceNormalizer(); nz.fit(x[tr], y_raw[tr], dcf=compute_dcf_radial(x, method='simple_ramp')[tr], envelope_exponent=args.envelope_exponent); norms[Z] = nz; b1s[Z] = ds['b1']
        zval = Z2z[Z] if args.z_mode == 'ff' else float(zi_of.get(Z, min(zi_of.values(), key=lambda i: abs(SL[i] - Z))))
        wrap = FixZ(model, zval).to(device)
        with torch.no_grad(): cart = A.reconstruct_cartesian(wrap, norms[Z], REFD, device=device.type, shared=sh, support_radius=args.support_radius, verbose=False)
        img, img_cplx = recon_nik_cart(cart, b1s[Z], bas, return_complex=True); od = os.path.join(args.out, f'tofts8_sl{Z}_s{args.seed}'); os.makedirs(od, exist_ok=True)
        np.save(os.path.join(od, f'nik_slice_{Z}_cplx.npy'), img_cplx); np.save(os.path.join(od, f'nik_slice_{Z}.npy'), img)
        print(f'  rendered slice {Z} at z {zval:.3f} ({"trained" if Z in SL else "QUERY, never trained on"}) -> {od}', flush=True)
    json.dump(dict(log=log, z=Z2z, slices=SL, query=QS, params=sum(p.numel() for p in model.parameters()), wall_s=time.time() - t0, args=vars(args)), open(os.path.join(args.out, 'train.json'), 'w'), indent=1)
    print(f'JOINT_Z_DONE {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__': main()
