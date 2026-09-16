#!/bin/bash
# raise the running 043 slice array from 12 to 24 concurrent tasks (qos allows 512 cpus)
scontrol update JobId=3308510 ArrayTaskThrottle=24 && echo "throttle set"
squeue -u P101440 -h -o "%i %j %T %M" | grep gv2 | sort | head -30
