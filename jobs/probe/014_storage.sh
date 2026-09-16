#!/bin/bash
# storage tiers on helios: mounts, quotas, scratch/project dirs, node-local disk, env vars
echo "== mounts"; df -h /net/beegfs /appdata /scratch /project /tmp /local /data 2>/dev/null
echo "== candidates"; ls -d /scratch* /project* /data* /local* /net/* /appdata/users/P101440 2>/dev/null
echo "== env"; env | grep -i "tmpdir\|scratch\|project\|beegfs\|appdata" | head
echo "== quota"; (beegfs-ctl --getquota --uid $USER 2>/dev/null || quota -s 2>/dev/null) | head -8
echo "== slurm tmp"; scontrol show config 2>/dev/null | grep -i "tmpfs\|TmpFS\|JobContainer" ; scontrol show node $(sinfo -h -p gpu -o %N | head -1 | cut -d, -f1 | sed 's/\[.*//')* 2>/dev/null | grep -i "TmpDisk\|Gres=" | head -3
echo "== motd"; head -40 /etc/motd 2>/dev/null; ls /etc/profile.d 2>/dev/null | head; cat /net/beegfs/README* /appdata/README* 2>/dev/null | head -40
