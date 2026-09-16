#!/bin/bash
#SBATCH -J arcverify
#SBATCH -p defq
#SBATCH -c 2
#SBATCH --mem 4G
#SBATCH -t 4:00:00
# read-only: for each migration archive, list members and check that each regular file exists on disk with the same size (prefixes tried:
# beegfs root, dirname of the archive, NEWROOT). report DCE_NIK/results/_admin/archive_verify.md. nothing is deleted.
R=/net/beegfs/users/P101440; O=$R/DCE_NIK/results/_admin; mkdir -p $O; M=$O/archive_verify.md
echo "# archive verification $(date '+%F %T') (read-only)" > $M; echo >> $M; echo "| archive | members (files) | found same size | found other size | missing | example missing |" >> $M; echo "|---|---|---|---|---|---|" >> $M
for A in $R/02_DCE_NIK.tar $R/03_dce_data.tar $R/04_XCAT-ERIC.tar $R/01_essentials.tgz $R/05_tmp.tar $R/claude_port.tgz $R/delta_20260908/delta_20260908.tgz $R/delta_20260908/delta_heavy_20260908.tgz $R/delta_20260908/claude_port_20260908.tgz $R/grasp_v2.zip; do
  [ -f "$A" ] || continue; n=0; ok=0; diff=0; miss=0; ex=""
  case "$A" in *.zip) LIST=$(unzip -l "$A" 2>/dev/null | awk 'NR>3 && $1 ~ /^[0-9]+$/ && $4 != "" {print $1"\t"$4}');;
               *) LIST=$(tar -tvf "$A" 2>/dev/null | awk '$1 ~ /^-/ {print $3"\t"$6}');; esac
  while IFS=$'\t' read sz p; do
    [ -z "$p" ] && continue; n=$((n+1)); f=""
    for pre in "$R" "$(dirname "$A")" "$R/DCE_NIK" "$R/tmp" "/"; do c="$pre/${p#./}"; [ -f "$c" ] && { f="$c"; break; }; done
    if [ -z "$f" ]; then miss=$((miss+1)); [ -z "$ex" ] && ex="$p"; elif [ "$(stat -c %s "$f")" = "$sz" ]; then ok=$((ok+1)); else diff=$((diff+1)); fi
  done <<< "$LIST"
  echo "| ${A#$R/} | $n | $ok | $diff | $miss | $ex |" >> $M; echo "done $A: $n files, $ok same, $diff differ, $miss missing"
done
echo >> $M; echo "same size = safe duplicate; other size = the tree copy changed since (newer work, keep the tree); missing = only in the archive (extract before deleting)" >> $M
echo "VERIFY_DONE"
