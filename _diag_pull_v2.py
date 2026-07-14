"""pull joint metrics from the v2 dcf_power sweep."""
import json, glob, os

RUNS=[
 ("p0.5",  "runs/20260601_135522_tiny-stimulating-bug_mct_carteval"),
 ("p0.7",  "runs/20260601_141735_fanatic-toucan-of-certainty_mct_carteval"),
 ("p0.85", "runs/20260601_143927_stereotyped-wombat-from-pluto_mct_carteval"),
 ("p1.0",  "runs/20260601_144813_aardwolf-of-great-eternity_mct_carteval"),
]
KEYS=['model_vs_gt_psnr','model_vs_gt_dists','model_vs_gt_haarpsi',
      'model_high_band_retention','model_sharp_inner','model_sharp_mid','model_sharp_outer',
      'model_radial_falloff','model_minus_cs_psnr']

rows=[]
for lbl,d in RUNS:
    sf=glob.glob(os.path.join(d,'wandb/run-*/files/wandb-summary.json'))
    if not sf: print(f'{lbl}: NO SUMMARY YET'); continue
    s=json.load(open(sf[0]))
    rows.append((lbl, {k:s.get(k) for k in KEYS}))

nice={'model_vs_gt_psnr':'PSNR','model_vs_gt_dists':'DISTS','model_vs_gt_haarpsi':'HaarPSI',
      'model_high_band_retention':'HBret','model_sharp_inner':'sharp_in','model_sharp_mid':'sharp_md',
      'model_sharp_outer':'sharp_ou','model_radial_falloff':'falloff','model_minus_cs_psnr':'mdl-CS'}
hdr=['PSNR','DISTS','HaarPSI','HBret','sharp_in','sharp_md','sharp_ou','falloff','mdl-CS']
key_by_hdr={v:k for k,v in nice.items()}
print(f'{"run":<7s}  ' + '  '.join(f'{h:>9s}' for h in hdr))
for lbl,m in rows:
    cells=[]
    for h in hdr:
        v=m.get(key_by_hdr[h])
        cells.append(f'{v:9.3f}' if isinstance(v,(int,float)) else f'{"?":>9s}')
    print(f'{lbl:<7s}  ' + '  '.join(cells))
print('\nCS ref:  PSNR=21.654  DISTS=0.350  HaarPSI=0.402  HBret=0.547  sharp 0.251/0.355/0.572')
