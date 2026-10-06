#!/bin/bash
# retrieval only: 118 progress
cd /net/beegfs/users/P101440/DCE_NIK; date '+%T'; squeue -u $USER -o "%.12i %.12j %.2t %.8M %R" -r | grep -E "JOBID|34718" | head -6
for f in $(ls -t jobs/log/118_delay_prior_34*.out 2>/dev/null | head -2); do echo "== $f"; grep -E "task [0-9]|model=|step +[0-9]+ |Traceback|Error|nan" "$f" | tail -3 | cut -c1-160; done
for W in 0.1 1; do j=results/tofts_vs_patlak/delay_prior/dly$W/tofts8_sl21_s0/wandb_runs/slice_21.json; [ -f $j ] && python3 -c "import json,sys; d=json.load(open('$j')); h=d.get('history',d); print('$W', [ (r.get('step'), round(r.get('delay_tv',-1),4), round(r.get('support',-1),4)) for r in (h[-3:] if isinstance(h,list) else [])])" 2>/dev/null; done
