# NAO standing baseline: PD control and measurements

`scripts/stand_nao.py` commands the nominal joint posture using the implicit
PD actuators in `NAO_STAND_CFG`. The references update at 100 Hz; PhysX solves
the articulation and joint drives at 200 Hz. There is no vision, learned policy,
inverse dynamics, step planner or CoM/DCM feedback in this baseline.

## Run

From the repository root, with the existing Isaac Lab environment and meshes:

```powershell
& C:\Users\USER\miniconda3\envs\env_isaaclab\python.exe scripts\stand_nao.py --headless --device cuda:0 --duration 6 --settle-time 0.5 --push-at 1.5 --push-velocity 0.3 --metrics-file outputs\pd_trial.json
```

For this machine's GUI, remove `--headless` and add
`--rendering_mode performance --kit_args=--/app/vulkan=false`.
If geometry is missing, follow the existing mesh bootstrap in the main README.

| Option | Meaning |
| --- | --- |
| `--duration` | Simulated seconds; default 10. |
| `--settle-time` | Initial interval excluded from metrics; default 0.5 s. Falls still count. |
| `--push-velocity` | Velocity increment along world x, in m/s; 0 disables, negative pushes backward. |
| `--push-at` | Push time; default halfway through the interval after settling. |
| `--metrics-file` | Optional JSON containing aligned time series, parameters and result. |
| `--rebuild-usd` | Regenerate the derived URDF and USD. |

The push directly changes root linear velocity. It is an impulse surrogate,
not a force applied at a specified point for a measured duration. World x is
forward only for the nominal initial heading used here.

## Measurement conditions

Isaac Sim 5.1.0.0, Isaac Lab checkout `b4c3210` (extension 0.54.4), Windows,
CUDA. Physics 5 ms; reference/render interval two physics steps. Settling 0.5 s;
push at 1.5 s. These are the current recorded trials, replacing the earlier
protocol's figures in this guide.

| Quantity | No push, 6 s | +0.30 m/s, 6 s | +0.70 m/s, 4 s |
| --- | ---: | ---: | ---: |
| Maximum CoM forward offset | 9.22 mm | 31.38 mm | 223.96 mm |
| Maximum DCM forward offset | 8.72 mm | 44.56 mm | 378.93 mm |
| Minimum nominal sagittal DCM margin | 69.25 mm | 58.74 mm | -275.63 mm |
| Minimum base height | 0.3215 m | 0.3215 m | 0.0660 m |
| Peak estimated ankle torque / limit | 1.5% | 74.7% | 100.0% |
| First fall after push | -- | -- | 0.245 s |
| Result / process exit code | Upright / 0 | Upright / 0 | Fell / 1 |

The failed trial's extrema include motion after falling. These three runs do
not establish a maximum recoverable push or a statistical success rate. No
controller gains were retuned during this review.

## What is measured

- **CoM:** mass-weighted average of body centers of mass and their velocities,
  using simulator states. These are not observations from a real sensor stack.
- **DCM:** horizontal `xi = c + c_dot / omega`, with nominal
  `omega = sqrt(g/h) = 6.04003 /s`. This is the divergent variable of the
  constant-height linear inverted pendulum model (LIPM).
- **CoP estimate:** reconstructed from incoming ankle joint wrenches, accounting
  for the foot's weight and translating the wrench to the ground plane. Foot
  acceleration and angular inertia are omitted: it is a quasi-static estimate,
  unreliable during impacts, flight and falls. The assumed joint/body frame
  alignment also needs checking if the USD or importer changes.
- **Ankle torque estimate:** Isaac Lab's clipped implicit-PD estimate,
  not an exact measurement of the motor effort integrated by PhysX.
- **Fall:** base height below 0.20 m or projected gravity z above -0.7
  (approximately 45.6 degrees from upright), checked from the first step.

Offsets use the midpoint of the feet. The reported margin is the minimum over
all recorded sagittal DCM samples relative to the nominal interval
`[-0.0607, 0.1033]` m. It is not a live polygon reconstructed from loaded contact
points, and it does not evaluate lateral balance.

JSON uses SI units and `null` for unavailable CoP samples. State timestamps
refer to the end of each physics step; estimated actuator torques were computed
when preparing that step. Interrupted or unmeasurable runs cannot pass.

## Model-based estimates and their scope

The corrected nominal LIPM velocity bounds are approximately 0.548 m/s forward,
0.443 m/s backward and 0.620 m/s laterally. These use CoM and polygon coordinates
relative to the same sole midpoint. The earlier 0.554/0.437 values mixed the
base origin with the sole origin by approximately 1 mm.

These are ideal fixed-support, constant-height bounds with controllable CoP;
they are not guarantees for this joint PD, and do not rule out other strategies
such as changing support or centroidal angular momentum. The rough torque-only
CoP reach is 116.2 mm with both ankle-pitch actuators and 58.1 mm with one;
contact geometry, shear forces and load sharing also matter.

`scripts/derive_gains.py` reproduces the scalar gain-sizing calculation. It
uses whole-body axis inertia for the legs and distal subtree inertia for the
arms. The assumed scalar gravitational stiffness and equal load sharing are
approximations, not a constrained multibody stability proof. The small held
hand/wrist gains include engineering floors.

## Tests and remaining work

The suite runs without Isaac Sim. On a Windows installation whose global
pytest temporary folder is inaccessible:

```powershell
& C:\Users\USER\miniconda3\envs\env_isaaclab\python.exe -m pytest --basetemp outputs\pytest_review
```

Use a directory dedicated to pytest. PhysX still reports issues with continuous
finger mimic joints lacking finite limits; successful standing does not
validate their mechanical fidelity.

This controller is the reference a learned policy has to beat. The task it is
measured on, the training procedure and the current comparison are in
[task_nao_stand.md](task_nao_stand.md); the headline numbers are in the
repository README.
