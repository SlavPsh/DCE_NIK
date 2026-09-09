# laptop loop

laptop (claude) -> git -> helios agent (slurm, defq) -> sbatch -> wandb + git -> laptop.

- `jobs/probe/<name>.sh`: run inline by the agent on its own node, 60 s cap, output `jobs/log/probe_<name>.out`.
- `jobs/queue/<name>.sh`: full sbatch script (own `#SBATCH` lines; `--output` is overridden to `jobs/log/<name>_<jid>.out`).
- `jobs/done/<name>`: marker written by the agent (jid, state, exit). a script runs once; rename to rerun.
- `jobs/agent/status.md`: active + last done, rewritten on change. `jobs/agent/active`: jids being tracked.
- `jobs/agent/STOP`: commit it to stop the agent (it exits without resubmitting; delete it and `sbatch helios_agent.sh` to restart).

agent commits json/md/png/csv/txt/out under `jobs/` and `results/` (<= 5 mb, never npy/npz/pt); logs of running jobs are committed when the job ends.

bootstrap on helios (citrix, once): `wandb login`, then `cd /net/beegfs/users/P101440/DCE_NIK && sbatch helios_agent.sh`.
laptop: `micromamba run -n dce python helios_fetch.py` (git pull + jobs table + wandb summaries in `results/_wandb/`).
