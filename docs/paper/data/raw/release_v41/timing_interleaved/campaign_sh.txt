#!/usr/bin/env bash
# R8 v4.1.0 release energy campaign driver (benchmark host, from ~/jsr-data/opponents/energy).
# R7 protocol unchanged (runs/r7/PREREG.md): one pinned core (CPU 16), 60 s idle windows, QUIET.lock per arm,
# TIMING.lock only when no CI VM runs, one rerun of a core-unclean arm. Five arms on panel73 in three
# counterbalanced blocks; every block also runs RC2 live +LM, Zipformer INT8, v4.1.0 live +LM and v4.1.0 final on
# the full 730-utterance panel (three paired full-panel timing repetitions; block 1's RC2/Zipformer pair is the R7 anchor check). The whole campaign holds the
# shared xbabe0 meter lease (/dev/shm/jsr-h172-meter.lease) so it never overlaps another metered run.
# usage: r8_campaign.sh            (all three blocks)
set -u
cd "$HOME/jsr-data/opponents/energy"
R=runs/r8/run-energy.sh
LOG=runs/r8/campaign.log
V41=models/isosyll-v41-h247-int8
ids=()  # panel73 arms dropped 00:45 MST: full-panel paired blocks only (time budget)
declare -A cmdline=(
 [rc2-live-lm]="r7-live-lm @N panel73"
 [v41-live-lm]="r8-live-lm @N panel73 $V41"
 [r0-int8-r6]="r7-zipformer @N panel73"
 [rc2-final]="r7-final @N panel73"
 [v41-final]="r8-final @N panel73 $V41"
 [anchor730-live]="r7-live-lm @N panel730"
 [anchor730-zip]="r7-zipformer @N panel730"
 [v41-live-lm-730]="r8-live-lm @N panel730 $V41"
 [v41-final-730]="r8-final @N panel730 $V41"
)
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
exec 9>>/dev/shm/jsr-h172-meter.lease
say "waiting for the shared meter lease"
flock 9
say "meter lease held"
TL=$HOME/jsr-data/opponents/TIMING.lock
HAVE_TL=0
take_timing_lock() {
  HAVE_TL=0
  local vms; vms=$(systemctl list-units 'neverhuman-xbabe0-lan-*' --state=active --no-legend --plain 2>/dev/null | grep -c neverhuman-xbabe0-lan- || true)
  if [[ ${vms:-0} -eq 0 ]] && mkdir "$TL" 2>/dev/null; then
    printf 'r8-energy pid=%s start=%s until=%s\n' "$$" "$(date -u +%FT%TZ)" "$(date -u -d '+40 min' +%FT%TZ)" > "$TL/owner"
    HAVE_TL=1
    say "TIMING.lock taken (no CI VM running): $(bash $HOME/jsr-data/opponents/timing-quiet/timing-quiet-wait.sh 300)"
  else
    say "TIMING.lock not taken (CI VMs active=${vms:-?} or lock held); QUIET.lock only"
  fi
}
release_timing_lock() { if [[ $HAVE_TL == 1 ]]; then rm -f "$TL/owner"; rmdir "$TL" 2>/dev/null; HAVE_TL=0; fi; }
trap release_timing_lock EXIT
for b in 1 2 3; do
  order=()
  case $b in
   1) order+=(anchor730-live anchor730-zip v41-live-lm-730 v41-final-730 "${ids[@]}") ;;
   2) order+=(v41-final-730 v41-live-lm-730 anchor730-zip anchor730-live); for ((i=${#ids[@]}-1;i>=0;i--)); do order+=("${ids[i]}"); done ;;
   3) order+=(v41-live-lm-730 anchor730-live v41-final-730 anchor730-zip); for ((i=0;i<${#ids[@]};i++)); do order+=("${ids[(i+2)%${#ids[@]}]}"); done ;;
  esac
  say "[b$b] BLOCK START order: ${order[*]}"
  for id in "${order[@]}"; do
    name="r8/b$b-r8-$id"
    if [[ -s runs/$name.json ]] && python3 -c "import json,sys;sys.exit(0 if json.load(open(sys.argv[1]))['clean_repetitions']['core']==1 else 1)" runs/$name.json; then say "[b$b] SKIP $id"; continue; fi
    for attempt in 1 2; do
      rm -rf "runs/out/$name"
      read -r -a args <<< "${cmdline[$id]//@N/$name}"
      while [[ -e $TL ]]; do say "[b$b] TIMING.lock present, waiting"; sleep 30; done
      take_timing_lock
      until bash $R quiet 45 "${args[@]}" >> runs/r8/arm-$b-$id.log 2>&1; do
        rc=$?; say "[b$b] quiet wrapper rc=$rc for $id, retry in 60 s"; sleep 60
        [[ -s runs/$name.json ]] && break
      done
      release_timing_lock
      clean=$(python3 - "runs/$name.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
print(d.get("clean_repetitions",{}).get("core"), d.get("joules_per_audio_s"), d.get("cpu_s_per_audio_s"), ",".join(d.get("flags",[])))
PY
)
      say "[b$b] DONE $id attempt=$attempt core_clean/J/cpu/flags: $clean"
      if [[ ${clean%% *} == 1 ]]; then break; fi
      if [[ $attempt == 1 ]]; then
        say "[b$b] UNCLEAN core rep for $id; keeping it as .unclean and rerunning once"
        mv "runs/$name.json" "runs/$name.unclean.json"
        mv "runs/out/$name" "runs/out/$name.unclean"; mv "runs/series/b$b-r8-$id.json" "runs/series/b$b-r8-$id.unclean.json" 2>/dev/null
      fi
    done
  done
  say "[b$b] BLOCK DONE"
done
say "CAMPAIGN DONE"
