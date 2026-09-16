#!/bin/bash
# helios facts for the DCE_Rerun grasp v2 job: which env has twixtools, defq node sizes, existing prep dirs, the target file
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
echo "== envs"; micromamba env list 2>&1 | tail -n +1
for e in torch29 anon; do printf "%s: " $e; timeout 25 micromamba run -n $e python -c "import twixtools, numpy; print('twixtools', twixtools.__version__ if hasattr(twixtools,'__version__') else 'ok', 'numpy', numpy.__version__)" 2>&1 | tail -1; done
printf "torch29 finufft/nibabel: "; timeout 20 micromamba run -n torch29 python -c "import finufft, nibabel, matplotlib, PIL; print(finufft.__version__, nibabel.__version__)" 2>&1 | tail -1
echo "== defq"; sinfo -p defq -o "%n %c %m %G %t" 2>&1 | head -20; scontrol show partition defq 2>&1 | grep -E "Max|Default" | head -5
echo "== qos"; sacctmgr show qos -P format=Name,MaxTRESPU,MaxJobsPU,MaxSubmitPU,GrpTRES 2>&1 | head -6
echo "== data"; ls -la /net/beegfs/users/P101440/dce_data/orig/; ls /net/beegfs/users/P101440/tmp/dce_rerun 2>&1 | head -20; du -sh /net/beegfs/users/P101440/tmp/dce_rerun 2>&1
echo "== quota"; df -h /net/beegfs/users/P101440 2>&1 | tail -1
