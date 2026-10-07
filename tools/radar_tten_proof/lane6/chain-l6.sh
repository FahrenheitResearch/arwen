#!/usr/bin/env bash
# chain-l6.sh ARM...: the lane 6 cycled arms on box N, one 8-card mutex request per arm, in order.
# Before each request it waits until no other lane 6 arm runs and no da-obs-full job (waiter or run) exists
# on the box (lead order 01:10Z: conv-full goes between lane 6 arms).
yield() {
  while pgrep -f "cycle-l6.sh " >/dev/null || pgrep -f "run.sh da-obs-full" >/dev/null; do sleep 60; done
}
for ARM in "$@"; do
  yield
  /opt/gpu-mutex/run.sh da-l6 --cards 8 --min-cards 4 --wait 21600 /work/da-l6/cycle-l6.sh $ARM
done
