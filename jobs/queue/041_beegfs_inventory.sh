#!/bin/bash
#SBATCH -J inventory
#SBATCH -p defq
#SBATCH -c 2
#SBATCH --mem 4G
#SBATCH -t 2:00:00
# beegfs inventory under /net/beegfs/users/P101440: per-directory totals, archives (tar/tgz/zip/7z) with size, date and whether an
# extracted sibling exists, largest files. read-only; nothing is deleted. report: DCE_NIK/results/_admin/beegfs_inventory.md
R=/net/beegfs/users/P101440; O=$R/DCE_NIK/results/_admin; mkdir -p $O; M=$O/beegfs_inventory.md
{
echo "# beegfs inventory $(date '+%F %T'), root $R (read-only)"; echo
echo "## top-level totals"; echo '```'; du -sh $R/* $R/.[a-zA-Z]* 2>/dev/null | sort -rh | head -40; echo '```'; echo
echo "## second level, largest 40"; echo '```'; du -sh $R/*/* 2>/dev/null | sort -rh | head -40; echo '```'; echo
echo "## archives (tar, tar.gz, tgz, zip, 7z, gz > 100 MB), with an extracted sibling if one exists"
echo "| size | date | archive | sibling dir (size) |"; echo "|---|---|---|---|"
find $R -type f \( -iname "*.tar" -o -iname "*.tar.gz" -o -iname "*.tgz" -o -iname "*.tar.bz2" -o -iname "*.tar.xz" -o -iname "*.zip" -o -iname "*.7z" -o -iname "*.gz" \) -size +100M -printf "%s\t%TY-%Tm-%Td\t%p\n" 2>/dev/null | sort -rn | while IFS=$'\t' read sz dt p; do
  b=$(basename "$p"); d=$(dirname "$p"); stem="${b%%.tar*}"; stem="${stem%.zip}"; stem="${stem%.7z}"; stem="${stem%.tgz}"; stem="${stem%.gz}"
  sib=""; for c in "$d/$stem" "$R/$stem" "$d/../$stem"; do [ -d "$c" ] && { sib="$c ($(du -sh "$c" 2>/dev/null | cut -f1))"; break; }; done
  printf "| %s | %s | %s | %s |\n" "$(numfmt --to=iec $sz)" "$dt" "${p#$R/}" "${sib#$R/}"
done
echo; echo "## largest 60 files anywhere"; echo '```'; find $R -type f -size +2G -printf "%s\t%p\n" 2>/dev/null | sort -rn | head -60 | while IFS=$'\t' read sz p; do printf "%8s  %s\n" "$(numfmt --to=iec $sz)" "${p#$R/}"; done; echo '```'
echo; echo "## checkpoints and snapshots (pt / snap npy) by directory, largest 30"; echo '```'; find $R -type f \( -name "*.pt" -o -name "snap_slice_*.npy" \) -printf "%s\t%h\n" 2>/dev/null | awk -F'\t' '{a[$2]+=$1} END {for (d in a) printf "%d\t%s\n", a[d], d}' | sort -rn | head -30 | while IFS=$'\t' read sz d; do printf "%8s  %s\n" "$(numfmt --to=iec $sz)" "${d#$R/}"; done; echo '```'
} > $M 2>&1
echo "INVENTORY_DONE $(wc -l < $M) lines"
