#!/usr/bin/env bash
# The loads scored in log space (transfer/docs/skill_space.md §9.1): the incumbent's PPO continued
# exactly as round 8's control was (its adapted rate, 1000 iterations, seed 41), the reward
# scoring the feet's loads in the generator's log space; then the same ladder. Against
# warm_r3_r8 only the load's space differs.
#
#   bash transfer/scripts/logload_study.sh
set -uo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"
export HUMANOID_LOAD_SPACE=log
WARM_ARMS=r3 WARM_SUFFIX=_logload bash transfer/scripts/warm_study.sh
