#!/bin/bash
# cancel the sequential p14 prep (067, job 3373461): resubmitted as a per-slice array (070)
scancel 3373461 && echo "cancelled 3373461"; squeue -u $USER -h -o "%i %j %T" | head
