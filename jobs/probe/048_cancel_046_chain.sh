#!/bin/bash
# cancel the array + post chained by the flawed 046 submit (GLAMS split to 0.02 only); 047 replaces them
scancel 3318981 3318982 && echo "cancelled 3318981 3318982"; sleep 5; squeue -u P101440 -h -o "%i %j %T" | grep -E "gv2" | sort | uniq -c | sort -rn | head
