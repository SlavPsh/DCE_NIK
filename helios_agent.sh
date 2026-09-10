#!/bin/bash
#SBATCH --job-name=helios-agent
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=0-12:00
#SBATCH --dependency=singleton
#SBATCH --output=/net/beegfs/users/P101440/DCE_NIK/logs/agent_%j.out
# helios side of the laptop loop. every PERIOD s: git pull (3 repos), probes inline with a 60 s cap,
# sbatch new queue scripts, commit small results (json/md/png/csv/txt/out, <= MAXKB kb), push.
# resubmits itself MARGIN s before the time limit; singleton = the successor pends until this one ends.
# bootstrap (citrix, once): cd /net/beegfs/users/P101440/DCE_NIK && sbatch helios_agent.sh
# stop: commit jobs/agent/STOP from the laptop. status: jobs/agent/status.md. job logs: jobs/log/<name>_<jid>.out
set -uo pipefail
ROOT=/net/beegfs/users/P101440
REPOS=(DCE_NIK grasp_v2 grasp_pro_py)
D=$ROOT/DCE_NIK; J=$D/jobs
PERIOD=${AGENT_PERIOD:-120}
MARGIN=${AGENT_MARGIN:-900}
MAXKB=${AGENT_MAXKB:-5120}
SMALL='\.(json|md|png|csv|txt|out)$'
LIMIT_DEFAULT=43200
ACTIVE_RE='^$'
mkdir -p "$J/probe" "$J/queue" "$J/done" "$J/log" "$J/agent" "$D/logs"
touch "$J/agent/active"
export GIT_TERMINAL_PROMPT=0

log() { echo "[$(date '+%F %T')] $*"; }

# git with an identity fallback (helios config may be empty)
gitc() {
  local r=$1; shift
  if git -C "$r" config user.email >/dev/null; then git -C "$r" "$@"
  else git -C "$r" -c user.name=helios-agent -c user.email=P101440@helios "$@"; fi
}

pull_all() {
  local r out
  for r in "${REPOS[@]}"; do
    [ -d "$ROOT/$r/.git" ] || { log "no repo $r"; continue; }
    out=$(gitc "$ROOT/$r" pull --rebase --autostash -q origin 2>&1) || log "pull FAILED $r: $out"
  done
}

# time limit of this job in s (squeue %l: [D-]HH:MM:SS | MM:SS | UNLIMITED)
limit_s() {
  local tl d=0 a b c s
  tl=$(squeue -h -j "${SLURM_JOB_ID:-0}" -o %l 2>/dev/null) || tl=""
  case "$tl" in ""|UNLIMITED|NOT_SET) echo "$LIMIT_DEFAULT"; return;; esac
  [[ $tl == *-* ]] && { d=${tl%%-*}; tl=${tl#*-}; }
  IFS=: read -r a b c <<<"$tl"
  if [ -z "${c:-}" ]; then s=$((10#$a*60+10#${b:-0})); else s=$((10#$a*3600+10#$b*60+10#$c)); fi
  echo $((s + d*86400))
}

# jobs/probe/*.sh not in jobs/done: run inline, 60 s cap
run_probes() {
  local f n st rc
  for f in "$J"/probe/*.sh; do
    [ -e "$f" ] || continue; n=$(basename "$f" .sh); [ -e "$J/done/$n" ] && continue
    st=$(date '+%F %T'); log "probe $n"
    ( cd "$D" && timeout -k 5 60 bash "$f" ) > "$J/log/probe_$n.out" 2>&1; rc=$?
    printf 'probe %s\nhost %s\nstart %s\nend %s\nexit %s\nlog jobs/log/probe_%s.out\n' \
      "$n" "$(hostname)" "$st" "$(date '+%F %T')" "$rc" "$n" > "$J/done/$n"
    log "probe $n exit $rc"; changed=1
  done
}

# jobs/queue/*.sh not in jobs/done: sbatch, log to jobs/log/<name>_<jid>.out
run_queue() {
  local f n jid err
  for f in "$J"/queue/*.sh; do
    [ -e "$f" ] || continue; n=$(basename "$f" .sh); [ -e "$J/done/$n" ] && continue
    err=$(mktemp)
    if jid=$(cd "$D" && sbatch --parsable --output="$J/log/${n}_%j.out" --error="$J/log/${n}_%j.out" "$f" 2>"$err") && [ -n "$jid" ]; then
      jid=${jid%%;*}
      printf 'queue %s\njid %s\nsubmitted %s\nstate SUBMITTED\nlog jobs/log/%s_%s.out\n' \
        "$n" "$jid" "$(date '+%F %T')" "$n" "$jid" > "$J/done/$n"
      echo "$jid $n" >> "$J/agent/active"; log "sbatch $n -> $jid"
    else
      printf 'queue %s\nsubmitted %s\nstate SBATCH_FAILED\nerror %s\n' \
        "$n" "$(date '+%F %T')" "$(tr '\n' ' ' < "$err")" > "$J/done/$n"
      log "sbatch FAILED $n: $(cat "$err")"
    fi
    rm -f "$err"; changed=1
  done
}

# active jids: state from squeue, final state from sacct when gone
track_jobs() {
  local keep="" jid n st cur fin
  [ -s "$J/agent/active" ] || return 0
  while read -r jid n; do
    [ -n "${jid:-}" ] || continue
    st=$(squeue -h -j "$jid" -o '%T %r' 2>/dev/null | sort | uniq -c | awk '{$1=$1; print}' | paste -sd, - | sed 's/,/, /g'); st=${st//&/\&}   # arrays: per state counts
    if [ -n "$st" ]; then
      keep+="$jid $n"$'\n'
      cur=$(sed -n 's/^state //p' "$J/done/$n" 2>/dev/null)
      [ "$cur" = "$st" ] || { sed -i "s|^state .*|state $st|" "$J/done/$n"; changed=1; }
    else
      fin=$(sacct -n -X -j "$jid" -o State,Elapsed 2>/dev/null | awk '{$1=$1; print}' | sort | uniq -c | awk '{$1=$1; print}' | paste -sd, - | sed 's/,/, /g'); [ -n "$fin" ] || fin="ENDED (no sacct)"
      sed -i "s|^state .*|state $fin|" "$J/done/$n"; echo "ended $(date '+%F %T')" >> "$J/done/$n"
      log "ended $n ($jid): $fin"; changed=1
    fi
  done < "$J/agent/active"
  printf '%s' "$keep" > "$J/agent/active"
  ACTIVE_RE=$(awk '{printf "^jobs/log/%s_|", $2}' "$J/agent/active")'^$'      # by script name: array tasks and chained jobs too
}

write_status() {
  local jid n
  {
    echo "# helios agent"; echo
    echo "updated $(date '+%F %T') on $(hostname), job ${SLURM_JOB_ID:-none}, cycle $cycle"; echo
    echo "## active"; echo; echo "| jid | script | state |"; echo "|---|---|---|"
    while read -r jid n; do [ -n "${jid:-}" ] && echo "| $jid | $n | $(sed -n 's/^state //p' "$J/done/$n") |"; done < "$J/agent/active"
    echo; echo "## done (last 20)"; echo; echo "| name | state |"; echo "|---|---|"
    ls -t "$J/done" 2>/dev/null | grep -v '^\.' | head -20 | while read -r n; do
      echo "| $n | $(sed -n 's/^state //p;s/^exit /exit /p' "$J/done/$n" | head -1) |"; done
  } > "$J/agent/status.md"
}

# stage small files under the given paths (allowed extensions, size cap, logs of running jobs deferred), commit, push
commit_push() {
  local r=$1; shift; local f kb n
  [ -d "$r/.git" ] || return 0
  cd "$r" || return 0
  git ls-files -z --others --modified --exclude-standard -- "$@" 2>/dev/null | while IFS= read -r -d '' f; do
    case "$f" in jobs/done/*|jobs/agent/*) ;; *) printf '%s\n' "$f" | grep -qE "$SMALL" || continue;; esac   # markers have no extension
    printf '%s\n' "$f" | grep -qE "$ACTIVE_RE" && continue
    if [ -f "$f" ]; then kb=$(( $(stat -c %s "$f") / 1024 )); [ "$kb" -le "$MAXKB" ] || { log "skip large $f (${kb} kb)"; continue; }; fi
    git add -- "$f"
  done
  git diff --cached --quiet && return 0
  n=$(git diff --cached --name-only | wc -l)
  gitc "$r" commit -q -m "agent $(date '+%F %H:%M') $(hostname): $n files" || { log "commit FAILED $r"; return 1; }
  gitc "$r" push -q origin HEAD 2>/dev/null \
    || { gitc "$r" pull --rebase --autostash -q origin && gitc "$r" push -q origin HEAD; } \
    || { log "push FAILED $r"; return 1; }
  log "pushed $r: $n files"
}

main() {
  local t_start limit successor=0 elapsed sid
  t_start=$(date +%s); cycle=0; changed=0
  limit=$(limit_s); log "start job ${SLURM_JOB_ID:-none} on $(hostname), limit ${limit}s, period ${PERIOD}s"
  while :; do
    cycle=$((cycle+1)); changed=0
    pull_all
    if [ -e "$J/agent/STOP" ]; then
      log "STOP present, exit without resubmit"; write_status; commit_push "$D" jobs results; exit 0
    fi
    run_probes; run_queue; track_jobs
    [ "$changed" = 1 ] || [ "$cycle" = 1 ] && write_status
    commit_push "$D" jobs results
    commit_push "$ROOT/grasp_v2" results
    commit_push "$ROOT/grasp_pro_py" results
    if [ "$successor" = 0 ] && [ -n "${SLURM_JOB_ID:-}" ]; then          # after one good cycle only
      if [ "$(squeue -h -u "$USER" -n helios-agent -t PD 2>/dev/null | wc -l)" = 0 ]; then
        if sid=$(cd "$D" && sbatch --parsable helios_agent.sh 2>&1); then log "successor $sid (pends on singleton)"
        else log "successor sbatch FAILED: $sid"; fi
      else log "successor already pending"; fi
      successor=1
    fi
    elapsed=$(( $(date +%s) - t_start ))
    if [ $((elapsed + PERIOD + MARGIN)) -ge "$limit" ]; then log "near limit (${elapsed}s of ${limit}s), exit for successor"; exit 0; fi
    sleep "$PERIOD"
  done
}
main
