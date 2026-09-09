#!/bin/bash
#SBATCH -J tbasis
#SBATCH -p defq
#SBATCH -c 4
#SBATCH --mem 16G
#SBATCH -t 0:30:00
#SBATCH -o /net/beegfs/users/P101440/DCE_NIK/results/tofts_vs_patlak/basis_%j.log
cd /net/beegfs/users/P101440/DCE_NIK; P=${PY:-/net/beegfs/users/P101440/micromamba/envs/torch29/bin/python}; O=results/tofts_vs_patlak
echo "### phantom, linear signal proxy (PRIMARY)"; $P -u nik_tofts_basis.py --aif aif_xph.npz --out $O/basis_xph.npz
echo; echo "### phantom, analytic signal via SPGR (secondary check: cost of the linear approximation)"
$P -u nik_tofts_basis.py --aif aif_xph.npz --out $O/basis_xph_spgr.npz --spgr 1400,4.66,18,3.5
for Z in 18 19 21; do echo; echo "### in vivo slice $Z, linear signal proxy"; $P -u nik_tofts_basis.py --aif aif_slice$Z.npz --out $O/basis_sl$Z.npz; done
echo ALL_BASIS_DONE
