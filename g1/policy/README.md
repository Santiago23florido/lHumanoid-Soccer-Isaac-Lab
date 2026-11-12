# `g1/policy/`: the pinned walking teacher

The one trained artifact the repository tracks: the G1's walking teacher
(`G1Walk-Teacher-v0`). Every walking plan the trackers train on comes from its rollouts, so it is
pinned and hashed rather than retrained; a run with another seed would be another gait.

| File | Contents |
| --- | --- |
| `teacher_policy.pt` | TorchScript, 85 925 parameters, 123 observations → 37 joint targets |
| `teacher_policy.json` | Provenance, hashes, command range and the measured gait (checked by `tests/test_teacher_policy.py`) |
| `teacher_env.yaml`, `teacher_agent.yaml` | The environment and PPO configurations of the run |

It loads with `torch` alone:

```python
import torch

policy = torch.jit.load("g1/policy/teacher_policy.pt", map_location="cpu").eval()
action = policy(observation)  # (N, 123) -> (N, 37)
```

The 37 outputs are all the articulation's joints, the hands included, as Isaac Lab's G1 action
term commands them; the observation is Isaac Lab's `G1FlatEnvCfg` policy group.

## Replacing it

`tests/test_teacher_policy.py` fails until `teacher_policy.json` matches the new file, so a
replacement is always deliberate:

```powershell
python scripts\train.py --task G1Walk-Teacher-v0 --headless --num_envs 4096 --max_iterations 1500 --seed 1
python scripts\play.py --task G1Walk-Teacher-Play-v0 --headless --num_envs 8 --checkpoint <run>\model_1499.pt
python g1\scripts\measure_gait.py --headless --checkpoint <run>\exported\policy.pt
python g1\scripts\capture_rollouts.py --headless --checkpoint g1\policy\teacher_policy.pt
```

Rollouts captured from the previous teacher must be recaptured.
