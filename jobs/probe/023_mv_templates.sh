#!/bin/bash
# move the painting templates under results/ so the agent syncs them
cd /net/beegfs/users/P101440/DCE_NIK && mkdir -p results/rois_manual && mv -f rois_manual/template_*.png results/rois_manual/ 2>/dev/null; rmdir rois_manual 2>/dev/null; ls -la results/rois_manual/ | awk '{print $5, $9}'
