#!/bin/bash
# stage the 109 gifs (the eval's git add ran before the agent cycle; some may be unstaged)
cd /net/beegfs/users/P101440/DCE_NIK; ls -la results/realdata_nik_vs_cs_figures/figures/recon_gif_v2*.gif results/realdata_nik_vs_cs_figures/figures/recon_gifs_single/*sub16_wd*.gif | awk '{printf "%.1f MB %s\n", $5/1e6, $9}'
git add results/realdata_nik_vs_cs_figures/figures/recon_gif_v2*.gif results/realdata_nik_vs_cs_figures/figures/recon_gifs_single/*sub16_wd*.gif && echo staged
