#!/usr/bin/env bash
# After the co-executions (logs/pact/coexecute.status "done"): gate G-X on the four bodies'
# replay captures. Run from a snapshot copy with PACT_ROOT=<repo>.
# Status: logs/pact/gate_x.status; log: logs/pact/gate_x.log
set -uo pipefail
root="${PACT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$root"
python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
status=logs/pact/gate_x.status
note() { echo "$1 $(date +%Y-%m-%d_%H:%M)" >> "$status"; }

note "waiting for the co-executions"
until grep -q "^done" logs/pact/coexecute.status 2>/dev/null; do sleep 60; done
missing=""
for body in g1 nao h1 cassie; do
  [ -f "logs/pact/coexec/${body}_replay.npz" ] || missing="$missing $body"
done
if [ -n "$missing" ]; then
  note "G-X not run: no replay capture for$missing"
else
  "$python" -u transfer/scripts/pact_gate_x.py --bodies g1,nao,h1,cassie \
    > logs/pact/gate_x.log 2>&1 && note "G-X: $(tail -1 logs/pact/gate_x.log)" \
    || note "G-X FAILED, see logs/pact/gate_x.log"
fi
note "done"
