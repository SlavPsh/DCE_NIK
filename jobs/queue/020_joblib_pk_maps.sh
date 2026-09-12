#!/bin/bash
#SBATCH -J pkmaps3
#SBATCH -p defq
#SBATCH -c 16
#SBATCH --mem 32G
#SBATCH -t 4:00:00
# joblib into torch29 (DCE-NET fitter dependency), then the fitted ext-kety parameter maps on the phantom
cd /net/beegfs/users/P101440/DCE_NIK
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH XPH_SIM=nomotion OMP_NUM_THREADS=1
micromamba install -y -n torch29 -c conda-forge joblib 2>&1 | tail -3
micromamba run -n torch29 python -c "import joblib; print('joblib', joblib.__version__)" || { micromamba run -n torch29 python -m pip install joblib 2>&1 | tail -1; }
micromamba run -n torch29 python -u pk_maps_phantom.py --jobs 16
echo "PK exit $?"
