#!/bin/bash
# the 118 eval's arrival diag crashed (checkpoint 'rank' is the cli default 12 for tofts models; the diag now takes the rank from the atoms buffer): rerun the eval stage
cd /net/beegfs/users/P101440/DCE_NIK; git log --oneline -1 -- arrival_artifact_diag.py | cut -c1-80
sbatch --export=ALL,STAGE=eval --array=0 -J dlyprior_eval2 --partition=gpu --gres=gpu:1g.12gb:1 -c 4 --mem 48G -t 3:00:00 --output=jobs/log/118_delay_prior_eval2_%j.out --error=jobs/log/118_delay_prior_eval2_%j.out jobs/queue/118_delay_prior.sh
