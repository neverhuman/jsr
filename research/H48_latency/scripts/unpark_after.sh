#!/usr/bin/env bash
# x9: another agent SIGSTOPped the H48b s4321 re-run decode set (guard 899747 + cells, group 897565) at ~08:06Z to give
# H48c-a the card. Resume them by PID / PGID the moment H48c-a's trainer exits; H48c-a's own end decodes wait for them
# (fulldev.sh never overlaps two decode sets on one card).
set -u
R=$HOME/jsr-data/runs/train/H48c_a_s1234
until ! kill -0 "$(cat $R/PID)" 2>/dev/null; do sleep 10; done
ps -o args= -p 899747 | grep -q "fleet_guard --pgid 897565" && kill -CONT 899747 && echo "$(date -u +%T) guard 899747 resumed"
kill -CONT -- -897565 2>/dev/null && echo "$(date -u +%T) decode group 897565 resumed"
