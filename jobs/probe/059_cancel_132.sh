#!/bin/bash
# cancel the joint-z campaign (132, array 3501995) and its chained eval; the runs can be resubmitted later from jobs/queue/132_jointz.sh
cd /net/beegfs/users/P101440/DCE_NIK; date '+%T'
scancel 3501995 && echo "cancelled 3501995"
for j in $(squeue -u $USER -h -o "%i %j" | grep -E "jointz" | awk '{print $1}'); do scancel $j && echo "cancelled $j"; done
squeue -u $USER -o "%.12i %.10j %.2t %.8M %R" -r | head -12
