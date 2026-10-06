#!/bin/bash
# retrieval only: the parametric aif fit numbers and figure (114 prep)
cd /net/beegfs/users/P101440/DCE_NIK; grep -E "fit rms|params" jobs/log/114_arrival_fixes_3469931.out | cut -c1-300
ls -la results/realdata_nik_vs_cs_figures/figures/aif_param_slice21.png && git add results/realdata_nik_vs_cs_figures/figures/aif_param_slice21.png && echo staged
grep -E "^ *[0-9]+ +[0-9]+ " results/tofts_vs_patlak/basis_sl21_r8_param.log results/tofts_vs_patlak/basis_sl21_r5.log results/tofts_vs_patlak/basis_sl21_r6.log results/tofts_vs_patlak/basis_sl21_r8.log 2>/dev/null | cut -c1-160
