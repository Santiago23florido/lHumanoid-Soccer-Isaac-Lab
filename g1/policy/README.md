# `g1/policy/` — the pinned teacher

This directory holds the one trained artifact the repository tracks.

| File | What it is |
| --- | --- |
| `teacher_policy.pt` | The teacher gait, exported TorchScript. 357 KB, 85 925 parameters, 123 observations → 37 joint targets. |
| `teacher_policy.json` | Provenance, hashes and the measured gait. Checked by `tests/test_teacher_policy.py`. |
| `teacher_env.yaml` | The environment configuration the run actually used, dumped by the run itself. |
| `teacher_agent.yaml` | The PPO configuration, likewise. |

## Why a binary is tracked here when nothing else is

The repository's standing rule is that derived artifacts stay out: the NAO
meshes are untracked for licensing, the generated USD because it embeds them,
`g1/rollouts/` because it is large and regenerable, and `logs/` because it is
training output.

The teacher is the exception, for one reason. **It is not reproducible.** The
run cost 1 h 58 m of GPU time and, more importantly, a rerun under a different
seed produces a different gait. Every comparison the transfer study makes —
three signals, masked against unmasked, three seeds — is a comparison *against
this teacher*. If the teacher drifts between experiments, the study measures the
teacher instead of the transfer, and it does so without any error appearing.

So the teacher is pinned rather than regenerated, and the checkpoint under
`logs/` is no longer the authority: this file is. 357 KB is a cheap price for a
result that can be reproduced by someone who was not here.

The export is plain TorchScript and loads with `torch` alone — no Isaac Lab, no
Isaac Sim, no Nucleus access:

```python
import torch

policy = torch.jit.load("g1/policy/teacher_policy.pt", map_location="cpu")
policy.eval()
action = policy(observation)          # (N, 123) -> (N, 37)
```

## What it is not

The 37 outputs are **all** of the loaded articulation's joints, including the
fourteen hand joints, because that is what Isaac Lab's shipped G1 action term
commands. Only the 23 body joints matter for a gait; see `g1/TRAINING.md` §5 for
the measured composition and why the `G1_29DOF_CFG` name is wrong.

The observation vector is Isaac Lab's `G1FlatEnvCfg` policy group and is only
meaningful inside that environment. This file is a teacher to *imitate*, not a
controller to drop into another environment — feeding it observations assembled
differently is the same class of error as the correspondence bugs recorded in
`g1/TRAINING.md` §5 and `transfer/docs/plan.md`.

## Replacing it

Deliberately awkward: `tests/test_teacher_policy.py` fails until
`teacher_policy.json` is updated to match, so replacing the teacher is a visible
decision rather than a side effect.

```powershell
# 1. Train
python scripts\train.py --task G1Walk-Teacher-v0 --headless `
    --num_envs 4096 --max_iterations 1500 --seed 1 --run_name teacher

# 2. Export TorchScript
python scripts\play.py --task G1Walk-Teacher-Play-v0 --headless --num_envs 8 `
    --checkpoint logs\rsl_rl\g1_flat\<run>\model_1499.pt

# 3. Verify it walks, rather than assuming (reward does not show a walk)
python g1\scripts\measure_gait.py --headless `
    --checkpoint logs\rsl_rl\g1_flat\<run>\exported\policy.pt

# 4. Pin it, and update every number in teacher_policy.json from step 3
copy logs\rsl_rl\g1_flat\<run>\exported\policy.pt g1\policy\teacher_policy.pt
copy logs\rsl_rl\g1_flat\<run>\params\*.yaml g1\policy\

# 5. Anything captured from the old teacher is now stale
del g1\rollouts\teacher.npz
python g1\scripts\capture_rollouts.py --headless `
    --checkpoint g1\policy\teacher_policy.pt
```

Step 5 is not optional. `g1/rollouts/teacher.npz` is the recorded behaviour of
*this* policy; keeping it beside a new one is how a comparison ends up spanning
two teachers.

Full training record: [`../TRAINING.md`](../TRAINING.md).
