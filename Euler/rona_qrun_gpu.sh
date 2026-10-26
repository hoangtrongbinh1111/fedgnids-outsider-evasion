#!/bin/bash
# GPU variant of rona_qrun2.sh: every job gets "--cpu 0" appended (argparse keeps the last value).
# rona_qrun.sh plus a shared memory lock, so that several lanes can run side by side.
# A lane takes Exps/.memlock, waits until NEED_GB of commit memory is free, starts its job and
# keeps the lock for HOLD seconds while the job allocates its memory; only then may another lane
# start a job. Same queue format as rona_qrun.sh.
#   PY=... NEED_GB=4 bash Euler/rona_qrun2.sh Exps/queue_E.txt E
source "$(dirname "$0")/rona_fed_common.sh"
PY=${PY:-python}
Q=$1; L=$2; FAIL=Exps/qfail_$L.txt; touch $FAIL
LOCK=Exps/.memlock
HOLD=${HOLD:-150}
nfail() { grep -cxF "$1" $FAIL; }
NEED_GB=${NEED_GB:-0}
free_gb() { powershell -NoProfile -c "\$o=Get-CimInstance Win32_OperatingSystem; [int](\$o.FreeVirtualMemory/1MB)"; }
# take the lock (a lock older than HOLD+60 s is stale: its holder died), then wait for memory
claim() {
  while ! mkdir $LOCK 2>/dev/null; do
    age=$(( $(date +%s) - $(stat -c %Y $LOCK 2>/dev/null || date +%s) ))
    [ $age -gt $((HOLD + 60)) ] && rmdir $LOCK 2>/dev/null
    sleep $((20 + RANDOM % 20))
  done
  [ "$NEED_GB" = "0" ] && return
  while true; do
    f=$(free_gb); [ "${f//[^0-9]/}" -ge "$NEED_GB" ] 2>/dev/null && return
    touch $LOCK; sleep 60
  done
}
release_later() { ( sleep $HOLD; rmdir $LOCK 2>/dev/null ) & }
while true; do
  ran=0; pending=0
  while read -r kind T rest; do
    rest=${rest%$'\r'}
    [ -z "$kind" ] && continue
    case $kind in \#*) continue;; esac
    if [ "$kind" = "END" ]; then continue; fi
    key="$kind $T $rest"
    [ "$(nfail "$key")" -ge 2 ] && continue
    if [ "$kind" = "train" ]; then
      done_train $T && continue
      ran=1; claim; release_later
      rm -rf $R/$T; mkdir -p $R/$T
      echo "== [$L] train $T $(date '+%m-%d %H:%M')"
      $PY -u -W ignore $rest --tag $T < /dev/null > $R/$T/log.txt 2>&1
      if done_train $T; then grep RESULT $R/$T/log.txt | cut -c1-160; else echo "   FAILED"; tail -3 $R/$T/log.txt; echo "$key" >> $FAIL; fi
      break
    elif [ "$kind" = "job" ]; then
      set -- $rest; OUT=$1; LOG=$2; shift 2
      [ -f $R/$T/$OUT ] && continue
      if ! done_train $T; then pending=1; continue; fi
      # another lane may be running this job already
      [ -f $R/$T/$LOG ] && [ $(( $(date +%s) - $(stat -c %Y $R/$T/$LOG) )) -lt 1800 ] && { pending=1; continue; }
      ran=1; claim
      [ -f $R/$T/$OUT ] && { rmdir $LOCK; continue; }
      release_later
      echo "== [$L] $T -> $OUT $(date '+%m-%d %H:%M')"
      $PY -u -W ignore "$@" --cpu 0 < /dev/null > $R/$T/$LOG 2>&1
      if [ -f $R/$T/$OUT ]; then summ $R/$T/$LOG; grep -h "^RESULT\|^SUMMARY" $R/$T/$LOG | cut -c1-200; else echo "   FAILED"; tail -3 $R/$T/$LOG; echo "$key" >> $FAIL; fi
      break
    fi
  done < $Q
  if [ $ran -eq 0 ]; then
    if grep -qx "END" $Q && [ $pending -eq 0 ]; then echo "== [$L] queue empty $(date '+%m-%d %H:%M')"; exit 0; fi
    sleep 120
  fi
done
