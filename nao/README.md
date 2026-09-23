# `nao/` — the target embodiment

**SoftBank Robotics / Aldebaran NAO H25 V5.0.** The robot a skill has to end up
on, and the reason the research question exists: 5.3 kg, 0.58 m, weak actuators,
and a zero-step capturable envelope between 0.443 and 0.827 m/s depending on the
direction of the push. Learning a dynamic skill directly on it is hard for
physical rather than algorithmic reasons.

![NAO in its nominal standing posture](img/nao_stand.png)

## What is here

| Path | Contents |
| --- | --- |
| `assets/urdf/` | The vendored URDF. BSD, verbatim upstream, tracked. |
| `assets/meshes/`, `assets/texture/` | Geometry. CC BY-NC-ND, **untracked**, fetched by `scripts/fetch_meshes.py`. |
| `assets/generated/` | Derived URDF and USD. **Untracked**: embeds the licensed geometry. |
| `assets/licenses/` | Upstream licences and full attribution. |
| `docs/` | Task specification and controller documentation. |
| `scripts/` | Every NAO-specific entry point. |
| `results/` | Measured results, as JSON. |
| `img/` | Figures. |

The Python code lives in `source/humanoid_transfer/humanoid_transfer/nao/`,
because Isaac Lab requires the extension layout. Everything else about the robot
is here.

## Documentation

| Document | Covers |
| --- | --- |
| [`docs/task_balance.md`](docs/task_balance.md) | The zero-step balance task: specification, observations, rewards, curriculum, training and evaluation. |
| [`docs/baseline_joint_pd.md`](docs/baseline_joint_pd.md) | The joint PD baseline and its measurements. |
| [`docs/baseline_capture_point.md`](docs/baseline_capture_point.md) | The capture-point controller, its derivation and its failure modes. |
| [`docs/asset_smoke_test.md`](docs/asset_smoke_test.md) | Loading the articulation, options, troubleshooting. |
| [`docs/asset_pipeline.md`](docs/asset_pipeline.md) | URDF to USD conversion and asset policy. |

## Scripts

| Script | Does |
| --- | --- |
| `fetch_meshes.py` | One-time licensed geometry bootstrap. **Run this first.** |
| `view_asset.py` | Smoke test: loads the passive articulation and lets it fall. |
| `render_pose.py` | Renders the standing posture to `img/`. |
| `derive_gains.py` | Reproduces the joint gain table from the URDF. |
| `run_baseline.py` | The model-based controllers. `--controller {joint_pd,dcm}`. |
| `compare_controllers.py` | Scores every controller on one environment instance. |
| `sweep_thresholds.py` | Largest push survived per direction, against the bound. |
| `diagnose_recovery.py` | Checks the assumptions a capturability comparison needs. |
| `check_env.py` | Environment instantiation and stepping check. |
| `metrics.py` | Metric export shared by the baseline scripts. |

## Quick start

```powershell
python nao\scripts\fetch_meshes.py        # once, prompts for licence acceptance
python nao\scripts\view_asset.py --headless --max-steps 1
python nao\scripts\derive_gains.py
```

## Balance results

Four controllers, one environment instance, n = 2048, pushes at 90 % of the
direction-dependent capturability bound.

| Controller | Fall-free | 95 % CI | Ankle torque |
| --- | --- | --- | --- |
| **PPO policy** | **82.3 %** | [80.6, 84.0] | 44.0 % |
| PPO, arms frozen | 83.0 % | [81.4, 84.6] | 47.9 % |
| Joint PD | 77.6 % | [75.8, 79.4] | 34.3 % |
| Capture-point PD | 27.8 % | [25.9, 29.7] | 30.2 % |

PPO beats the joint PD by 4.7 points (z = 3.76). The arms contribute nothing
measurable (z = 0.59). Raw data in
[`results/comparison_directional.json`](results/comparison_directional.json).

## Derived parameters

Every number below comes from the URDF via
`humanoid_transfer.nao.assets.nao_kinematics`, and each has a test.

| Symbol | Quantity | Value |
| --- | --- | --- |
| `M` | Total mass | 5.3054 kg |
| `z_c` | CoM height above the soles | 0.268900 m |
| `ω₀` | `√(g/z_c)` | 6.0400 rad/s |
| `ln2/ω₀` | Error doubling time | 114.8 ms |
| `Mgl` | Gravitational stiffness | 11.9526 N·m/rad |
| `I` | Whole-body inertia about the ankle | 0.390045 kg·m² |
| — | Ratio to the distal subtree | **543×** |
| — | Foot | 164.0 × 92.3 × 65.9 mm |

## Licensing

The geometry is CC BY-NC-ND 4.0 and is never committed; see
[`assets/licenses/README.md`](assets/licenses/README.md) for the full reasoning
and upstream commit SHAs. **Non-commercial only.**
