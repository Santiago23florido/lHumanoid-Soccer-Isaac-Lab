# Balance Baseline — Holding The NAO Upright With A PD

`scripts/stand_nao.py` makes the NAO stand. It is the first thing in this
project that keeps the robot off the floor, and it does it with a fixed linear
feedback law: every joint is commanded to hold the nominal standing posture,
and the derived gains do the rest.

There is no learning here. That is the point — a learned policy needs a baseline
to be measured against, and this is it.

## One-time asset bootstrap

The NAO meshes are licensed CC BY-NC-ND 4.0 and are never committed. Fetch them
once per checkout:

```powershell
python scripts\fetch_nao_meshes.py
```

## Run it

```powershell
& "C:\Users\USER\miniconda3\shell\condabin\conda-hook.ps1"
conda activate env_isaaclab
cd C:\IsaacLab
.\isaaclab.bat -p "C:\Users\USER\Documents\AppsPlayGround\lHumanoid-Soccer-Isaac-Lab\scripts\stand_nao.py" --device cuda:0 --headless
```

For the GUI on this machine, add `--rendering_mode performance
--kit_args=--/app/vulkan=false`; the RTX/Vulkan path crashes at startup here.

| Flag | Meaning |
| --- | --- |
| `--duration` | Seconds of simulated time. Default 10. |
| `--push-velocity` | Forward CoM velocity in m/s to impose as a shove. 0 disables. |
| `--push-at` | When to push, in seconds. Defaults to halfway. |
| `--settle-time` | Seconds to ignore before measuring. Default 0.5. |
| `--rebuild-usd` | Regenerate the derived URDF and USD first. |

## What it measures, and why those quantities

The script reports the quantities the balance theory is actually written in,
not just "did it fall".

**Centre of mass.** Assembled from the simulator's own body poses and masses,
because Isaac Lab exposes per-body states but no whole-body centre of mass.

**Divergent component of motion**, `ξ = x + ẋ/ω₀`. The linear inverted pendulum
splits into a stable mode and an unstable one. This is the unstable one, and it
obeys `ξ̇ = ω₀(ξ − p)` where `p` is the centre of pressure. Regulating `ξ` is
the whole of the balance problem. Regulating the centre of mass alone is not: a
centre of mass sitting still with the wrong velocity is already falling.

**Centre of pressure.** Where the ground reaction can be replaced by a single
force with no horizontal moment. Unilateral contact confines it to the support
polygon — a foot can push on the ground but never pull — which is precisely
what bounds how much the ankle can do.

The simulator does not report the centre of pressure, so it is reconstructed
from the ankle joint reaction wrench in three steps: rotate the wrench to world
frame, recover the ground reaction by applying Newton's law to the foot alone
(including the foot's own 0.17 kg, which is worth a millimetre or two), then
slide the wrench down to the contact plane and solve for the point where the
horizontal moment vanishes.

**Ankle torque** as a fraction of the URDF limit. This is what saturates first.

## What the robot can do

Measured on Isaac Sim 5.1.0 / Isaac Lab 0.54.4, Windows 11, RTX 4070.

Standing still, no perturbation:

| Quantity | Value |
| --- | --- |
| Base height | 0.3215 m, flat |
| Tilt (`projected_gravity_b[2]`) | −0.9997 (−1 is exactly upright) |
| CoM forward offset | +9.0 mm |
| DCM forward offset | +8.7 mm |
| Ankle torque used | 3.2 % of the limit |

Recovering from a 0.30 m/s forward push, which is 54 % of the zero-step
capturable limit:

| Quantity | Value |
| --- | --- |
| Peak CoM offset | +31.4 mm |
| Peak DCM offset | +50.5 mm |
| Peak CoP offset | +74.8 mm |
| Ankle torque used | 84.3 % of the limit |
| Outcome | stayed upright |

Two things in that table are worth reading twice.

The centre of pressure leads the divergent component — 74.8 mm against 50.5 mm.
That is not a coincidence, it is the control law: to pull `ξ` back the foot must
push the pressure centre *past* it, because `ξ̇ = ω₀(ξ − p)` only becomes
negative when `p > ξ`. Watching those two numbers is watching the ankle
strategy work.

And 54 % of the capturable limit already costs 84 % of the ankle torque. The
relationship is not linear, and it is why the margin runs out quickly.

Pushed past the limit, at 0.70 m/s — 126 % of the predicted 0.554 m/s — it
fails exactly the way the theory says it should:

| Quantity | Value |
| --- | --- |
| Peak DCM offset | +378.9 mm, far outside the 103.3 mm polygon |
| Ankle torque used | 100 % — saturated |
| Fell at | 0.48 s after the push |
| Outcome | fell, did not recover |

So the measured boundary sits between 0.30 m/s (recovers, comfortably) and
0.70 m/s (falls, unrecoverably), bracketing the 0.554 m/s that capturability
predicts from geometry alone. The script prints both diagnostic notes in this
case: the divergent component left the support polygon, and the ankle strategy
was exhausted. Those are the two distinct ways a standing biped runs out of
options, and the robot hit both at once.

This is the baseline a learned policy has to beat. It cannot beat it by pushing
the pressure centre further — the foot is already the binding constraint. It has
to do something the PD cannot: swing the arms to trade angular momentum, or
crouch to buy time. That is the interesting part.

## The authority limits it prints

Before stepping, the script prints where the robot's authority ends:

```
  AUTHORITY LIMITS
    ankle pitch torque       : 3.023 N m each, 6.046 total
    CoP reach from torque    : 116.2 mm (two feet)
    CoP reach from geometry  : 103.3 mm (toe edge)
    binding constraint       : foot geometry
```

In double support the foot runs out before the motors do: the toe is 103.3 mm
ahead of the ankle, while the two ankles between them could push the pressure
centre 116.2 mm. On one foot that reverses — a single ankle reaches only
58.1 mm, well inside the same 103.3 mm of foot — so the binding constraint
changes with the stance. Any controller that assumes one or the other is wrong
half the time.

```
  ZERO-STEP CAPTURABILITY (no stepping, ankle strategy only)
    forward                  : 0.554 m/s
    backward                 : 0.437 m/s
    lateral                  : 0.620 m/s
```

Past these the robot has to take a step or throw its arms; no ankle torque will
save it. They come from `ω₀ d`, where `d` is the distance from the centre of
mass to the polygon edge. Backward is tighter than forward because the centre of
mass sits 11.6 mm ahead of the ankle axis and the heel is nearer than the toe.

These numbers are what size the perturbation curriculum for the learning task:
there is nothing to learn from a push that no controller could recover.

## Exit code

`0` if the robot stayed upright for the whole run, `1` otherwise — so it can be
used as a regression check on the actuator gains.

## Known noise

PhysX prints errors at startup about the finger joints needing finite limits to
be used by the mimic joint feature. The upstream URDF declares them
`continuous`, which carries no limit, and they are `mimic` joints driven by
`LHand` / `RHand`. The errors are non-fatal and the articulation simulates
correctly; the finger drives are held at zero precisely so this cannot matter.
