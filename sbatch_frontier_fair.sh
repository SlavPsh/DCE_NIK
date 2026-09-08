#!/bin/bash
#SBATCH -J frontfair
#SBATCH -p luna-cpu-tiny
#SBATCH -c 4
#SBATCH --mem 16G
#SBATCH -t 1:30:00
#SBATCH -o /scratch/rnga/vvpshenov/DCE_NIK/frontfair_%j.log
cd /scratch/rnga/vvpshenov/DCE_NIK
P=${PY:-/scratch/rnga/vvpshenov/micromamba/envs/torch29/bin/python}
echo "########## how much did the recon itself move?"
$P -u - <<'PY'
import numpy as np
L="/scratch/rnga/vvpshenov/grasp_pro_py/results_spoke_cs"
F="/scratch/rnga/vvpshenov/grasp_pro_py/results_spoke_cs_fairphi"
def c(u,v):
    u=u.ravel()-u.mean(); v=v.ravel()-v.mean(); return float((u@v)/np.sqrt((u@u)*(v@v)))
print(f"{'recon':18} {'corr(leak,fair)':>16} {'rel L2 diff':>12}")
for n in ["cs_slice13_f100","cs_slice13_f70","cs_slice13_f50","cs_slice13_f35","cs_slice13_f25",
          "cs_slice18_f25","cs_slice19_f25","cs_slice20_f25","cs_slice21_f25"]:
    a=np.abs(np.load(f"{L}/{n}.npy")); b=np.abs(np.load(f"{F}/{n}.npy"))
    print(f"{n:18} {c(a,b):16.4f} {np.linalg.norm(a-b)/np.linalg.norm(a):12.4f}")
PY
echo; echo "########## fair-phi spoke frontier (CS re-scored)"
CSD=/scratch/rnga/vvpshenov/grasp_pro_py/results_spoke_cs_fairphi CSPRE=cs TAG=_fairphi $P -u haarpsi_spoke.py
echo; echo "########## original (leaked) frontier for reference"
cat haarpsi_spoke.json
echo FRONTIER_DONE
