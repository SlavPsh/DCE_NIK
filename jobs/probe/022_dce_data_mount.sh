#!/bin/bash
# why did two array tasks not see the raw file? nodes of the 070 tasks, the dce_data path type, mounts on the login node
sacct -j 3373535,3373536,3373537,3373461 -o JobID%14,NodeList%14,State%12,Elapsed --noheader 2>/dev/null
ls -la /net/beegfs/users/P101440/ | grep dce_data; ls -la /net/beegfs/users/P101440/dce_data/ | head; readlink -f /net/beegfs/users/P101440/dce_data/orig; df -h /net/beegfs/users/P101440/dce_data/orig 2>/dev/null | tail -1
stat -c "%s %n" /net/beegfs/users/P101440/dce_data/orig/meas_topqmri_p14.dat
sinfo -p defq -o "%N %T %c %m" | head
