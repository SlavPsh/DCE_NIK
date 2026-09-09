#!/bin/bash
#SBATCH -J fairswap
#SBATCH -p defq
#SBATCH -c 4
#SBATCH --mem 20G
#SBATCH -t 2:00:00
#SBATCH -o /net/beegfs/users/P101440/DCE_NIK/fairswap_%j.log
cd /net/beegfs/users/P101440/DCE_NIK
P=${PY:-/net/beegfs/users/P101440/micromamba/envs/torch29/bin/python}
RD=results/realdata_nik_vs_cs_figures/figures

echo "=== archive the leaked versions ==="
for f in haarpsi_spoke.json task_S.json; do [ -f "$f" ] && cp -n "$f" "${f%.json}_leaked.json" && echo "  kept ${f%.json}_leaked.json"; done
for f in fig1_spoke_frontier fig3_denoising; do [ -f "$RD/$f.png" ] && cp -n "$RD/$f.png" "$RD/${f}_leaked.png" && echo "  kept ${f}_leaked.png"; done

# canonical outputs now come from the FAIR recon set. TAG empty on purpose: these become the notebook's numbers.
export CSD=/net/beegfs/users/P101440/grasp_pro_py/results_spoke_cs_fairphi
export CSPRE=cs
export TAG=

echo; echo "=== haarpsi_spoke (fair) ==="; $P -u haarpsi_spoke.py || echo FAILED
echo; echo "=== task_S (fair f25) ==="; $P -u task_S.py || echo FAILED
echo; echo "=== figures (fair) ==="; $P -u task_realdata_figures.py || echo FAILED
echo; echo "=== refresh notebook in place ==="; $P -u refresh_nb.py || echo FAILED
echo FAIRSWAP_DONE
