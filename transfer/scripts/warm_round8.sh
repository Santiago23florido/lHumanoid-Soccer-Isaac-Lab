#!/usr/bin/env bash
# Round 8 of the warm study (transfer/docs/skill_space.md §9.1): the physics arm from round 6's
# distillation, which passed its gate (student 50, incumbent 58 terminations with PPO's noise),
# and the control, both with PPO from the incumbent's adapted learning rate; then both ladders.
#
#   bash transfer/scripts/warm_round8.sh
set -uo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"
export HUMANOID_FEEDFORWARD_OBS=1
WARM_ARMS="v2t r3" WARM_SUFFIX=_r8 \
  WARM_DISTILLED=logs/rsl_rl/g1_track_physics/2026-10-22_21-29-34_distill_v2t \
  bash transfer/scripts/warm_study.sh
