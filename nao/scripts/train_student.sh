#!/usr/bin/env bash
# The student: the teacher's actor, frozen, on an estimate of the robot's parameters.
#
#   bash nao/scripts/train_student.sh <teacher checkpoint> [name] [-- extra arguments]
#        (default name: s2r_student)
#
# Runs nao/scripts/train_adaptation.py on NaoTrack-Sim2RealStudent-v0 with the plans and
# robot model the teacher was trained with (nao/scripts/train_teacher.sh). Arguments after
# "--" go to train_adaptation.py (e.g. -- --rounds 3 --num_envs 256 for a smoke run).
# Progress: logs/contract/train_<name>.log; the module and its per-round record land in
# logs/rsl_rl/nao_track/<date>_<name>/.
set -euo pipefail

teacher="$1"
name="${2:-s2r_student}"
shift $(( $# < 2 ? $# : 2 ))
[ "${1:-}" = "--" ] && shift
python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
[ -f "$teacher" ] || { echo "missing $teacher"; exit 1; }

export HUMANOID_CONTRACT="$root/logs/contract/walk_dagger2e/generator.pt"
export HUMANOID_POSTURE_CONTRACT="$HUMANOID_CONTRACT"
export HUMANOID_PROCEDURAL=0.7
export HUMANOID_SIM2REAL_HISTORY="${HUMANOID_SIM2REAL_HISTORY:-3}"
"$python" -u nao/scripts/train_adaptation.py --headless --teacher "$teacher" \
  --run_name "$name" "$@" > "logs/contract/train_${name}.log" 2>&1
echo "trained $(date +%H:%M)"
