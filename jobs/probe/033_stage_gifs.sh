#!/bin/bash
# stage the 102 gifs for the agent commit (agent syncs png/md/json only by default); sizes printed
cd /net/beegfs/users/P101440/DCE_NIK; ls -la results/realdata_nik_vs_cs_figures/figures/recon_gif*.gif | awk '{printf "%.1f MB %s\n", $5/1e6, $9}'
git add results/realdata_nik_vs_cs_figures/figures/recon_gif*.gif && echo "staged"
