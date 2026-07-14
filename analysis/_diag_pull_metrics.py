"""pull image-quality metrics from each dcf_power run's wandb summary."""
import json, glob, os, re

RUN_DIRS=[
 ("p0   (dcf off)", "runs/20260529_161643_lyrical-amaranth-kiwi_mct_carteval"),
 ("p0.3",           "runs/20260529_162007_literate-attractive-dodo_mct_carteval"),
 ("p0.5",           "runs/20260529_164030_wooden-impossible-jellyfish_mct_carteval"),
 ("p1.0",           "runs/20260529_164339_dramatic-rare-numbat_mct_carteval"),
]
KEYS=[
 'model_vs_gt_psnr', 'model_vs_gt_dists', 'model_vs_gt_haarpsi',
 'cs_vs_gt_psnr',    'cs_vs_gt_dists',    'cs_vs_gt_haarpsi',
 'nufft_vs_gt_psnr', 'nufft_vs_gt_dists', 'nufft_vs_gt_haarpsi',
 'model_high_band_retention', 'cs_high_band_retention',
 'model_minus_cs_psnr',
]

rows=[]
for lbl,d in RUN_DIRS:
    sfiles=glob.glob(os.path.join(d, 'wandb/run-*/files/wandb-summary.json'))
    if not sfiles: print(f'{lbl}: no summary'); continue
    s=json.load(open(sfiles[0]))
    rows.append((lbl, {k: s.get(k) for k in KEYS}))

# print as table
hdr_keys=['model_vs_gt_psnr','model_vs_gt_dists','model_vs_gt_haarpsi',
          'model_high_band_retention','model_minus_cs_psnr']
nice={'model_vs_gt_psnr':'mdl PSNR','model_vs_gt_dists':'mdl DISTS','model_vs_gt_haarpsi':'mdl HaarPSI',
      'model_high_band_retention':'mdl HB ret','model_minus_cs_psnr':'mdl-CS PSNR'}
print(f'{"run":<14s}  ' + '  '.join(f'{nice[k]:>11s}' for k in hdr_keys))
for lbl,m in rows:
    cells=[]
    for k in hdr_keys:
        v=m.get(k)
        cells.append(f'{v:11.4f}' if isinstance(v,(int,float)) else f'{"?":>11s}')
    print(f'{lbl:<14s}  ' + '  '.join(cells))
print('\nreference rows (constant across runs since CS/NUFFT/GT identical):')
if rows:
    m=rows[0][1]
    print(f'  CS  vs GT:  PSNR={m.get("cs_vs_gt_psnr"):.3f}  DISTS={m.get("cs_vs_gt_dists"):.4f}  HaarPSI={m.get("cs_vs_gt_haarpsi"):.4f}  HBret={m.get("cs_high_band_retention"):.4f}')
    print(f'  NUFFT GT :  PSNR={m.get("nufft_vs_gt_psnr"):.3f}  DISTS={m.get("nufft_vs_gt_dists"):.4f}  HaarPSI={m.get("nufft_vs_gt_haarpsi"):.4f}')
