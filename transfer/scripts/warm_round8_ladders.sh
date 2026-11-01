#!/usr/bin/env bash
# Round 8's ladders again: on 2026-10-23 from 03:09 the asset server's outage failed every capture
# of both arms' ladders (USD file not found); the trained runs stand.
#
#   bash transfer/scripts/warm_round8_ladders.sh
set -uo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"
export HUMANOID_FEEDFORWARD_OBS=1
WARM_ARMS="v2t r3" WARM_SUFFIX=_r8 WARM_LADDER_ONLY=1 bash transfer/scripts/warm_study.sh
