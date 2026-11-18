#!/usr/bin/env bash
# After the H1 tracker: gate G-A2 (and its held-out-NAO reading), Cassie's mirror, Cassie's
# tracker. Waits for "done" in logs/pact/train_trackers.status before starting.
# Run from a snapshot copy with PACT_ROOT=<repo>. Status: logs/pact/queue_after_h1.status
set -uo pipefail
root="${PACT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$root"
python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
status=logs/pact/queue_after_h1.status
note() { echo "$1 $(date +%Y-%m-%d_%H:%M)" >> "$status"; }

note "waiting for the H1 tracker"
until grep -q "^done" logs/pact/train_trackers.status 2>/dev/null; do sleep 60; done
note "H1 queue done: $(grep -E 'pact_h1' logs/pact/train_trackers.status | tail -1)"

"$python" -u transfer/scripts/pact_gate_a2.py --bodies g1,nao,h1,cassie \
  > logs/pact/gate_a2.log 2>&1 && note "G-A2: $(tail -1 logs/pact/gate_a2.log)" || note "G-A2 FAILED"
"$python" -u transfer/scripts/pact_gate_a2.py --bodies g1,h1,cassie,nao --hold-out nao \
  --arms pact_k --output transfer/results/icml_gate_a2_holdout_nao.json \
  > logs/pact/gate_a2_holdout_nao.log 2>&1 && note "G-A2 held-out NAO written" || note "G-A2 hold-out FAILED"

"$python" -u transfer/scripts/measure_mirror.py --headless --task-id CassieTrack-Play-v0 \
  --prefix left_ right_ --suffix _left _right --output transfer/bodies/cassie/mirror.json \
  > logs/pact/mirror_cassie.log 2>&1
if [ -f transfer/bodies/cassie/mirror.json ]; then
  note "cassie mirror measured"
  # the trackers' queue rewrites its own status; start a fresh run of it for Cassie
  mv logs/pact/train_trackers.status "logs/pact/train_trackers_h1.status"
  cp transfer/scripts/pact_train_trackers.sh logs/pact/.train_trackers.run.sh
  PACT_ROOT="$root" bash logs/pact/.train_trackers.run.sh cassie 3000 41
  note "cassie: $(tail -2 logs/pact/train_trackers.status | head -1)"
else
  note "cassie mirror FAILED, see logs/pact/mirror_cassie.log"
fi
note "done"
