#!/bin/bash
# retrieval only: 118 step lines (nan check) after the first logged steps
cd /net/beegfs/users/P101440/DCE_NIK; date '+%T'; export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
j=results/tofts_vs_patlak/delay_prior/dly0.1/tofts8_sl21_s0/wandb_runs/slice_21.json; ls -la $j 2>&1 | awk '{print $5, $9}'
micromamba run -n torch29 python -c "
import json; d=json.load(open('$j')); h=d.get('history', d)
rows=[r for r in (h if isinstance(h,list) else []) if 'delay_tv' in r]; print('logged', len(rows)); print([(r.get('step'), round(r['delay_tv'],4), round(r.get('support',-1),4), round(r.get('loss',-1),4)) for r in rows[-4:]])" 2>&1 | tail -3
grep -E "step +[0-9]+ " jobs/log/118_delay_prior_3471851.out | tail -2
