#!/usr/bin/env bash
# After gate G-X (logs/pact/gate_x.status "done"): gate G-A3 on the co-execution map
# captures, the NAO held out. Run from a snapshot copy with PACT_ROOT=<repo>.
# Status: logs/pact/gate_a3.status; log: logs/pact/gate_a3.log
set -uo pipefail
root="${PACT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$root"
python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
status=logs/pact/gate_a3.status
note() { echo "$1 $(date +%Y-%m-%d_%H:%M)" >> "$status"; }

note "waiting for gate G-X"
until grep -q "^done" logs/pact/gate_x.status 2>/dev/null; do sleep 60; done
missing=""
for body in g1 nao h1 cassie; do
  [ -f "logs/pact/coexec/${body}_map.npz" ] || missing="$missing $body"
done
if [ -n "$missing" ]; then
  note "G-A3 not run: no map capture for$missing"
else
  "$python" -u transfer/scripts/pact_gate_a3.py --bodies g1,nao,h1,cassie --hold-out nao \
    > logs/pact/gate_a3.log 2>&1 && note "G-A3: $(grep -E 'PASSES|FAILS' logs/pact/gate_a3.log | tail -1)" \
    || note "G-A3 FAILED, see logs/pact/gate_a3.log"
fi
note "done"
