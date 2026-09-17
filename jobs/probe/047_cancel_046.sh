#!/bin/bash
# cancel queue 046 submit job (3318980): it carries the comma-split --export bug, the fixed copy is queue 047
scancel 3318980 && echo "cancelled 3318980"; squeue -u P101440 -h -o "%i %j %T" | grep -E "gv2|3318980" | head
