#!/bin/bash
# stage the 109 gifs; the single sub16 gifs got the bare name "<ds>_sl<Z>_sub16.gif" (label split at the bracket), renamed to *_sub16_wd3e-3_prior.gif
cd /net/beegfs/users/P101440/DCE_NIK/results/realdata_nik_vs_cs_figures/figures
for f in recon_gifs_single/*_sub16.gif; do [ -f "$f" ] && mv -v "$f" "${f%.gif}_wd3e-3_prior.gif"; done
ls -la recon_gifs_single/*wd3e-3* | awk '{printf "%.1f MB %s\n", $5/1e6, $9}'
git add recon_gif_v2_*.gif recon_gifs_single/*wd3e-3*.gif && echo staged
