#!/usr/bin/env bash
# W3 and W4, after W1 (w1_resample.sh) has finished.
#
#   bash transfer/scripts/w3_w4_queue.sh            # every stage, decisions by their criteria
#   W34_STAGES="probes" bash transfer/scripts/w3_w4_queue.sh
#
# Stages, in order; each skips what already exists, so the queue resumes:
#   probes   step 1: the frozen incumbents on the transfer probes (both bodies)
#   g1       step 3, only if step 1's prediction holds on the G1: smoke train, then three seeds
#            of r3_6194 continued as S's with the shaped family, laddered as S's were; and the
#            transfer probes on S's own seeds (the control)
#   w4       only if W1 passed (transfer/results/w1_resample.json): B1-NAO seed 31 retrained
#            under the generator space, both skills
#   nao      step 4, only if step 1 holds on the NAO: three seeds of r1_3498 as S's, shaped
set -uo pipefail

python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
status=logs/contract/w3_w4.status
note() { echo "$1 $(date +%Y-%m-%d_%H:%M)" | tee -a "$status"; }
stages="${W34_STAGES:-probes g1 w4 nao}"
want() { [[ " $stages " == *" $1 "* ]]; }
results=transfer/results

room() {
  local free gpu busy
  free=$(powershell -NoProfile -Command \
    "[math]::Floor((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB)")
  gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  # No brackets or backslashes in the pattern (bash -> PowerShell quoting).
  busy=$(powershell -NoProfile -Command \
    "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'capture_contract|scripts.train.py|capture_velocity' }).Count")
  [[ "$free" =~ ^[0-9]+$ && "$gpu" =~ ^[0-9]+$ && "$busy" =~ ^[0-9]+$ ]] || return 1
  [ "$free" -ge 10 ] && [ "$gpu" -le 2500 ] && [ "$busy" -eq 0 ]
}
wait_for_room() {
  local waited=0
  until room; do
    [ $waited -eq 0 ] && note "waiting for room before $1"
    waited=1
    sleep 120
  done
}
assets_online() {
  local server=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1 url
  for url in "$server/Isaac/IsaacLab/Robots/Unitree/G1/g1_minimal.usd" \
    "$server/Isaac/Environments/Grid/default_environment.usd"; do
    [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 -I "$url")" = 200 ] || return 1
  done
}
rate() {
  "$python" -c "
import sys, torch
print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['optimizer_state_dict']['param_groups'][0]['lr'])" "$1"
}
latest() {
  echo "model_$(ls "$1" | grep -E '^model_[0-9]+\.pt$' | sed 's/model_//; s/\.pt//' | sort -n | tail -1)"
}
holds() {  # <json> <python expression on d>: prints yes/no
  "$python" -c "
import json, sys
d = json.load(open(sys.argv[1]))
print('yes' if ($2) else 'no')" "$1" 2>/dev/null || echo no
}
conditions() {  # <body>: S's training conditions
  export HUMANOID_PROCEDURAL=0.7 HUMANOID_SYMMETRY=loss HUMANOID_HEADS=0 HUMANOID_PLAN_TRUST=0
  if [ "$1" = g1 ]; then
    export HUMANOID_CONTRACT="$root/logs/contract/walk_dagger3e/generator.pt" HUMANOID_LOAD_SPACE=log
    unset HUMANOID_POSTURE_CONTRACT
  else
    export HUMANOID_CONTRACT="$root/logs/contract/walk_dagger2e/generator.pt" HUMANOID_LOAD_SPACE=linear
    export HUMANOID_POSTURE_CONTRACT="$HUMANOID_CONTRACT"
  fi
}
# <body> <run name> <seed> <from run> <checkpoint> <iterations> [num envs]
train() {
  local body="$1" name="$2" seed="$3" from="$4" checkpoint="$5" iterations="$6" envs="${7:-4096}"
  local task=G1Track-v0
  [ "$body" = nao ] && task=NaoTrack-v0
  wait_for_room "train $name"
  until assets_online; do sleep 60; done
  note "$name training start"
  "$python" -u scripts/train.py --task "$task" --headless --num_envs "$envs" \
    --max_iterations "$iterations" --seed "$seed" --run_name "$name" --resume \
    --load_run "$(basename "$from")" --checkpoint "$checkpoint.pt" \
    "agent.algorithm.learning_rate=$(rate "$from/$checkpoint.pt")" \
    > "logs/contract/train_${name}.log" 2>&1 || { note "$name training FAILED"; return 1; }
  note "$name trained"
}
# <body> <sym policy> <tag>: the transfer probes, true state, autonomous (as the map runs)
probes() {
  local body="$1" policy="$2" tag="$3" task_id=G1Track-Play-v0 walk=walk_dagger3e
  local rollouts=g1/rollouts
  if [ "$body" = nao ]; then task_id=NaoTrack-Play-v0; walk=walk_dagger2e; rollouts=nao/rollouts; fi
  local out="$results/transfer_probes_${body}_${tag}.json"
  [ -f "$out" ] && { note "probes $body $tag: already done"; return 0; }
  wait_for_room "probes $body $tag"
  until assets_online; do sleep 60; done
  local extra=()
  [ "$body" = nao ] && extra=(env "HUMANOID_POSTURE_CONTRACT=$root/logs/contract/$walk/generator.pt")
  "${extra[@]}" env HUMANOID_CONTRACT="$root/logs/contract/$walk/generator.pt" \
    "$python" -u transfer/scripts/capture_contract.py --headless --task-id "$task_id" \
    --driver-policy "$policy" --steps 900 --settle-steps 100 --num-envs 216 \
    --contract-from-truth --reference autonomous --gait-probes transfer/probes/transfer_probes.json \
    --output "$rollouts/zs_${tag}_transfer.npz" > "logs/contract/capture_zs_${tag}_transfer.log" 2>&1 \
    || { note "probes $body $tag capture FAILED"; return 1; }
  "$python" transfer/scripts/transfer_probe_effect.py "$rollouts/zs_${tag}_transfer.npz" \
    --output "$out" > /dev/null || { note "probes $body $tag evaluate FAILED"; return 1; }
  note "probes $body $tag done"
}
# <body> <run dir> <tag> <unsym tag>: laddered as S's seeds were
ladder() {
  local body="$1" run="$2" tag="$3" unsym="$4" checkpoint task_id=G1Track-Play-v0
  [ "$body" = nao ] && task_id=NaoTrack-Play-v0
  checkpoint="$(latest "$run")"
  if [ ! -f "$results/competence_${body}_${unsym}_corners.json" ]; then
    wait_for_room "map $unsym"
    bash transfer/scripts/zero_shot_ladder.sh "$body" "$run" "$checkpoint" "$unsym" map corners \
      > "logs/contract/ladder_${unsym}_map.log" 2>&1 || note "$unsym map FAILED"
  fi
  for task in rough flat; do
    local generator=flat_dagger2 stages=(l4)
    [ "$task" = rough ] && { generator=rough_b4; stages=(encoder walk l4); }
    [ -f "$results/zero_shot_${body}_${tag}_${generator}_auto_kin_l4_turning.json" ] && continue
    wait_for_room "l4 $tag $task"
    until assets_online; do sleep 60; done
    ZS_REFERENCE=autonomous ZS_ESTIMATOR=kinematic ZS_SYMMETRIZE=1 ZS_TASK_ID="$task_id" \
      ZS_TASK="$task" ZS_GENERATOR="logs/contract/${generator}/generator.pt" \
      bash transfer/scripts/zero_shot_ladder.sh "$body" "$run" "$checkpoint" "$tag" "${stages[@]}" \
      > "logs/contract/ladder_${tag}_${task}.log" 2>&1 || note "$tag $task ladder FAILED"
  done
  probes "$body" "$run/exported/${checkpoint}_sym.pt" "$tag"
  note "evaluated $tag"
}
smoke() {  # <body> <from run> <checkpoint>: 3 iterations, then the saved env.yaml must carry the shape
  local body="$1" from="$2" checkpoint="$3" name="w3smoke_${1}" run
  if ! ls -d logs/rsl_rl/"${body}"_track/*_"$name" > /dev/null 2>&1; then
    train "$body" "$name" 1 "$from" "$checkpoint" 3 256 || return 1
  fi
  run="$(ls -d logs/rsl_rl/"${body}"_track/*_"$name" | tail -1)"
  if grep -A3 "procedural_load_shape" "$run/params/env.yaml" | grep -q "0.5"; then
    note "smoke $body: env.yaml carries the load shape"
  else
    note "smoke $body: env.yaml lacks procedural_load_shape FAILED"
    return 1
  fi
}

# The previous queue first: one Isaac process at a time.
until grep -q "^w1 done" logs/contract/w1_resample.status 2>/dev/null; do sleep 120; done
note "w3/w4 start: $stages"
"$python" transfer/scripts/w1_compare.py --output "$results/w1_resample.json" \
  > logs/contract/w1_compare.out 2>&1 || note "w1 compare FAILED"

g1_r3=logs/rsl_rl/g1_track/2026-10-14_15-38-39_zs_g1_r3_loss
g1_incumbent=logs/rsl_rl/g1_track/2026-10-24_17-31-00_warm_r3_logload
nao_incumbent=logs/rsl_rl/nao_track/2026-10-14_19-59-50_zs_nao_r1

if want probes; then
  probes g1 "$g1_incumbent/exported/model_7193_sym.pt" warm_r3_logload_sym
  probes nao "$nao_incumbent/exported/model_3498_sym.pt" r1_3498_sym_v3
fi

g1_go=$(holds "$results/transfer_probes_g1_warm_r3_logload_sym.json" "any(d['prediction_holds'].values())")
nao_go=$(holds "$results/transfer_probes_nao_r1_3498_sym_v3.json" "any(d['prediction_holds'].values())")
w1_go=$(holds "$results/w1_resample.json" "d.get('passes')")
note "decisions: g1 step 3 $g1_go, nao step 4 $nao_go, w4 $w1_go"

if want g1 && [ "$g1_go" = yes ]; then
  conditions g1
  export HUMANOID_LOAD_SHAPE=0.5,2
  smoke g1 "$g1_r3" model_6194 && for seed in 41 42 43; do
    name="warm_r3_shape_s${seed}"
    ls -d logs/rsl_rl/g1_track/*_"$name" > /dev/null 2>&1 \
      || train g1 "$name" "$seed" "$g1_r3" model_6194 1000 || continue
    ladder g1 "$(ls -d logs/rsl_rl/g1_track/*_"$name" | tail -1)" "${name}_sym" "$name"
  done
  unset HUMANOID_LOAD_SHAPE
  # The control: S's own seeds on the transfer probes.
  probes g1 "$g1_incumbent/exported/model_7193_sym.pt" warm_r3_logload_sym
  for seed in 42 43; do
    run="$(ls -d logs/rsl_rl/g1_track/*_warm_r3_s${seed}_log | tail -1)"
    probes g1 "$run/exported/$(latest "$run")_sym.pt" "warm_r3_s${seed}_log_sym"
  done
fi

if want w4 && [ "$w1_go" = yes ]; then
  export HUMANOID_RESAMPLE=generator HUMANOID_PROCEDURAL=0 HUMANOID_SYMMETRY=loss HUMANOID_HEADS=0
  export HUMANOID_PLAN_TRUST=0 HUMANOID_LOAD_SPACE=linear
  export HUMANOID_POSTURE_CONTRACT="$root/logs/contract/walk_dagger2e/generator.pt"
  for task in rough flat; do
    generator=rough_b4
    [ "$task" = flat ] && generator=flat_dagger2
    name="nao_task_${task}_gen"
    tag="${name}_sym"
    if ! ls -d logs/rsl_rl/nao_track/*_"$name" > /dev/null 2>&1; then
      # The skill's generator with the NAO's encoder, as B1 trained (paper_queue.sh b1).
      export HUMANOID_CONTRACT="$root/logs/contract/${generator}_zs_nao_r1_3498_sym_v3/generator.pt"
      train nao "$name" 31 "$nao_incumbent" model_3498 1000 || continue
    fi
    run="$(ls -d logs/rsl_rl/nao_track/*_"$name" | tail -1)"
    [ -f "$results/zero_shot_nao_${tag}_${generator}_auto_kin_l4_turning.json" ] && continue
    wait_for_room "l4 $tag"
    ZS_REFERENCE=autonomous ZS_ESTIMATOR=kinematic ZS_SYMMETRIZE=1 ZS_TASK_ID=NaoTrack-Play-v0 \
      ZS_TASK="$task" ZS_GENERATOR="logs/contract/${generator}/generator.pt" \
      bash transfer/scripts/zero_shot_ladder.sh nao "$run" "$(latest "$run")" "$tag" encoder l4 \
      > "logs/contract/ladder_${tag}_${task}.log" 2>&1 || note "$tag ladder FAILED"
    note "evaluated $tag"
  done
  unset HUMANOID_RESAMPLE
fi

if want nao && [ "$nao_go" = yes ]; then
  conditions nao
  export HUMANOID_LOAD_SHAPE=0.5,2
  smoke nao "$nao_incumbent" model_3498 && for seed in 31 32 33; do
    name="warm_nao_shape_s${seed}"
    ls -d logs/rsl_rl/nao_track/*_"$name" > /dev/null 2>&1 \
      || train nao "$name" "$seed" "$nao_incumbent" model_3498 1000 || continue
    ladder nao "$(ls -d logs/rsl_rl/nao_track/*_"$name" | tail -1)" "${name}_sym" "$name"
  done
  unset HUMANOID_LOAD_SHAPE
  for seed in 31 32 33; do
    # S's seed 31 is the run named warm_nao_r1; the others carry their seed.
    run=logs/rsl_rl/nao_track/2026-10-24_02-47-22_warm_nao_r1
    [ "$seed" = 31 ] || run="$(ls -d logs/rsl_rl/nao_track/*_warm_nao_r1_s${seed} | tail -1)"
    probes nao "$run/exported/$(latest "$run")_sym.pt" "warm_nao_r1_s${seed}_sym"
  done
fi
note "w3/w4 done"
