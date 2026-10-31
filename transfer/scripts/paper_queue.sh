#!/usr/bin/env bash
# The paper's experiments, in order (transfer/paper/README.md). Every stage skips itself when its
# last result file exists, so the queue resumes where it stopped.
#
#   S    seeds of each body's final training stage, deployed symmetrized as the incumbents are;
#   B1   the tracker trained on the skill's own plans (N x M), from each body's incumbent;
#   B2   the oracle (the NAO trained by RL on the skill's task) -- added once implemented;
#   B3   joint retargeting -- added once implemented.
#
#   bash transfer/scripts/paper_queue.sh [stages]     (default: "s_g1 b1_nao s_nao b1_g1")
#
# B1 trains with the incumbent's seed (31 on the NAO, 41 on the G1); B1_SEEDS_NAO and
# B1_SEEDS_G1 list the seeds to run instead, e.g. "32 33". Another seed's runs and results
# carry it: nao_task_rough_s32, zero_shot_nao_nao_task_rough_s32_sym_*.
set -uo pipefail
python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
status=logs/contract/paper_queue.status
note() { echo "$1 $(date +%H:%M)" | tee -a "$status"; }
stages="${1:-s_g1 b1_nao s_nao b1_g1}"
want() { [[ " $stages " == *" $1 "* ]]; }
results=transfer/results

g1_incumbent=logs/rsl_rl/g1_track/2026-10-24_17-31-00_warm_r3_logload
g1_checkpoint=model_7193
nao_incumbent=logs/rsl_rl/nao_track/2026-10-14_19-59-50_zs_nao_r1
nao_checkpoint=model_3498

rate() {  # the learning rate a checkpoint's PPO had reached
  "$python" -c "
import sys, torch
print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['optimizer_state_dict']['param_groups'][0]['lr'])" "$1"
}
latest() {  # <run directory>: its last checkpoint's name
  echo "model_$(ls "$1" | grep -E '^model_[0-9]+\.pt$' | sed 's/model_//; s/\.pt//' | sort -n | tail -1)"
}
conditions() {  # <body>: the conditions each body's trackers train in
  export HUMANOID_PROCEDURAL=0.7 HUMANOID_SYMMETRY=loss HUMANOID_HEADS=0 HUMANOID_PLAN_TRUST=0
  if [ "$1" = g1 ]; then
    export HUMANOID_CONTRACT="$root/logs/contract/walk_dagger3e/generator.pt" HUMANOID_LOAD_SPACE=log
    unset HUMANOID_POSTURE_CONTRACT
  else
    export HUMANOID_CONTRACT="$root/logs/contract/walk_dagger2e/generator.pt" HUMANOID_LOAD_SPACE=linear
    export HUMANOID_POSTURE_CONTRACT="$HUMANOID_CONTRACT"
  fi
}
train() {  # <body> <run name> <seed> <load run dir> <checkpoint> [extra hydra overrides]
  local body="$1" name="$2" seed="$3" from="$4" checkpoint="$5"
  shift 5
  local task=G1Track-v0
  [ "$body" = nao ] && task=NaoTrack-v0
  "$python" -u scripts/train.py --task "$task" --headless --num_envs 4096 --max_iterations 1000 \
    --seed "$seed" --run_name "$name" --resume --load_run "$(basename "$from")" \
    --checkpoint "$checkpoint.pt" "agent.algorithm.learning_rate=$(rate "$from/$checkpoint.pt")" "$@" \
    > "logs/contract/train_${name}.log" 2>&1 || { note "$name training FAILED"; return 1; }
  note "$name trained"
}
ladder() {  # <body> <run dir> <tag> <tasks> <rough stages...>: symmetrized, as the incumbents are
  local body="$1" run="$2" tag="$3" tasks="$4"
  shift 4
  local task_id=G1Track-Play-v0
  [ "$body" = nao ] && task_id=NaoTrack-Play-v0
  for task in $tasks; do
    local generator stages
    if [ "$task" = rough ]; then generator=rough_b4; stages=("$@"); else generator=flat_dagger2; stages=(l4); fi
    # L4 reads the body's refitted encoder: a ladder of one skill alone (B1) fits it first.
    [ -f "logs/contract/walk_zs_${body}_${tag}/generator.pt" ] || stages=(encoder "${stages[@]}")
    ZS_REFERENCE=autonomous ZS_ESTIMATOR=kinematic ZS_SYMMETRIZE=1 ZS_TASK_ID="$task_id" \
      ZS_TASK="$task" ZS_GENERATOR="logs/contract/${generator}/generator.pt" \
      bash transfer/scripts/zero_shot_ladder.sh "$body" "$run" "$(latest "$run")" "$tag" "${stages[@]}" \
      > "logs/contract/ladder_${tag}_${task}.log" 2>&1 || note "$tag $task ladder FAILED"
    note "evaluated $tag $task"
  done
}
done_already() {  # <body> <tag> <task>: its L4's last command exists
  local generator=rough_b4
  [ "$3" = flat ] && generator=flat_dagger2
  [ -f "$results/zero_shot_${1}_${2}_${generator}_auto_kin_l4_turning.json" ]
}
b1() {  # <body> <incumbent run> <checkpoint> <encoder tag> <incumbent's seed> <seeds...>
  # The body's incumbent continued on each skill's own plans, no procedural family.
  local body="$1" incumbent="$2" checkpoint="$3" encoder="$4" first="$5"
  shift 5
  conditions "$body"
  for seed in "$@"; do
    for task in rough flat; do
      local generator=rough_b4
      [ "$task" = flat ] && generator=flat_dagger2
      local name="${body}_task_${task}"
      [ "$seed" = "$first" ] || name="${name}_s${seed}"
      local tag="${name}_sym"
      done_already "$body" "$tag" "$task" && { note "$tag: already done"; continue; }
      if ! ls -d logs/rsl_rl/"${body}"_track/*_"$name" > /dev/null 2>&1; then
        # The skill's generator with this body's encoder, as the ladder deploys it (L4).
        ( export HUMANOID_CONTRACT="$root/logs/contract/${generator}_${encoder}/generator.pt"
          export HUMANOID_PROCEDURAL=0
          train "$body" "$name" "$seed" "$incumbent" "$checkpoint" ) || continue
      fi
      ladder "$body" "$(ls -d logs/rsl_rl/"${body}"_track/*_"$name" | tail -1)" "$tag" "$task" encoder l4
    done
  done
}

note "paper queue start: $stages"

# S, the G1: the log-load continuation's seeds 42 and 43 (seed 41 is the incumbent).
if want s_g1; then
  conditions g1
  for seed in 42 43; do
    tag="warm_r3_s${seed}_log_sym"
    done_already g1 "$tag" flat && { note "$tag: already done"; continue; }
    run="$(ls -d logs/rsl_rl/g1_track/*_warm_r3_s${seed}_log | tail -1)"
    ladder g1 "$run" "$tag" "rough flat" encoder walk l4
  done
fi

# B1, the NAO.
if want b1_nao; then
  b1 nao "$nao_incumbent" "$nao_checkpoint" zs_nao_r1_3498_sym_v3 31 ${B1_SEEDS_NAO:-31}
fi

# S, the NAO: its incumbent continued in its own conditions, seeds 31 (warm_nao_r1), 32 and 33.
if want s_nao; then
  conditions nao
  for seed in 31 32 33; do
    name="warm_nao_r1_s${seed}" tag="warm_nao_r1_s${seed}_sym"
    [ "$seed" = 31 ] && name=warm_nao_r1
    done_already nao "$tag" flat && { note "$tag: already done"; continue; }
    if ! ls -d logs/rsl_rl/nao_track/*_"$name" > /dev/null 2>&1; then
      train nao "$name" "$seed" "$nao_incumbent" "$nao_checkpoint" || continue
    fi
    ladder nao "$(ls -d logs/rsl_rl/nao_track/*_"$name" | tail -1)" "$tag" "rough flat" encoder walk l4
  done
fi

# B1, the G1.
if want b1_g1; then
  b1 g1 "$g1_incumbent" "$g1_checkpoint" zs_g1_warm_r3_logload_sym 41 ${B1_SEEDS_G1:-41}
fi

note "paper queue done: $stages"
