# The Second Baseline — Capture-Point Balance

`scripts/stand_nao.py --controller dcm` runs a balance controller that knows
what the joint PD does not: whether the robot is falling.

The joint PD reacts to joint error. It holds the nominal posture, and the only
reason that keeps the robot upright is that the posture happens to be an
equilibrium. Push it hard enough and it holds the wrong posture very accurately
all the way down.

This one closes a loop on the quantity that decides the outcome.

## The law

Under the linear inverted pendulum the horizontal centre of mass obeys
`ẍ = ω₀²(x − p)`, with `p` the centre of pressure and `ω₀ = √(g/z_c)`. That
second-order system factors into two first-order ones through the **divergent
component of motion**:

```
ξ = x + ẋ/ω₀
ξ̇ = ω₀(ξ − p)        ← unstable, and the whole problem
ẋ = −ω₀(x − ξ)        ← stable, takes care of itself
```

So balance is not about where the centre of mass *is*. A robot standing
perfectly still with the wrong velocity is already falling, and no amount of
joint-space feedback can see it.

Place the centre of pressure at

```
p = ξ_ref + k(ξ − ξ_ref),    k > 1
```

and the closed loop becomes `ξ̇ = ω₀(1−k)(ξ − ξ_ref)`, stable at rate
`ω₀(k−1)`. At `k = 2.5` that is 9.06 rad/s against a natural divergence of
6.04 rad/s.

**`k > 1` is the entire design.** The pressure centre has to *overtake* the
divergent component, not follow it. A controller that chases the centre of mass
never catches it, and the constructor rejects `k ≤ 1` for that reason.

## From pressure centre to joint angles

Static equilibrium of one foot with normal load `F` and the pressure centre `d`
ahead of the ankle axis gives the moment the shank must transmit:

```
m_y = +d·F        m_x = −e·F
```

These signs were **checked against the simulator, not assumed**. With
`d = 7.5 mm` and `F = 24.4 N` the model predicts `m_y = 0.183 N·m`; the
reported joint reaction wrench reads `0.1842`. Deriving a sign and trusting it
is how balance controllers end up pushing the robot over.

The actuators are implicit position PDs, so a torque is requested by displacing
the target:

```
q_des = q + (τ_des + K_d·q̇) / K_p
```

which inverts the drive law `τ = K_p(q_des − q) − K_d·q̇`. The point of going
through the position target rather than `set_joint_effort_target` is that PhysX
then integrates the PD at the full 200 Hz physics rate instead of holding one
torque across the whole 10 ms control period.

## When the ankle runs out

Unilateral contact confines `p` to the support polygon — a foot can push on the
ground but never pull. For the NAO the **foot runs out before the motors do**:
103.3 mm of toe against 116.2 mm of torque reach in double support.

Past that the only remaining source of horizontal force is a change of
centroidal angular momentum: accelerate the trunk and arms one way and the
ground pushes the body the other. The controller keeps the *unclipped* command
and feeds the part it could not realise into hip and shoulder pitch, in
opposite directions so the pair is a momentum exchange rather than a lean.

Measured on the bench: at 0.6 m/s with the ankle saturated it commands hip
+0.19 rad and shoulder −0.40 rad. At 0.2 m/s, unsaturated, **both are exactly
zero** — the momentum strategy stays out of the way until it is needed.

This is the piece a fixed joint PD structurally cannot express, because the
whole purpose of a position loop is to hold the arms still.

## Two versions that were wrong

Both are kept in the tests, because they were more instructive than the
version that worked.

### Asking each foot for a point outside itself

The first version asked *each* ankle to put its own pressure centre on the
midline. The feet sit 100 mm apart and are 92 mm wide, so **the midline is
outside both of them**. No ankle torque can put pressure where there is no
foot.

Lateral authority in double support comes from how the load *divides* between
the two feet, not from either one reaching a point it does not own. The fix was
to clip the global command into each foot's own sole, which the left foot
mirrors from the right.

### A reference that was not an equilibrium

The second version used `ξ_ref = 0`, the geometric centre of the support
polygon. But the nominal posture balances with the centre of mass **12.6 mm
ahead of that**, and that posture is what the rest of the body's position loops
are holding.

Asking for the geometric centre therefore demanded a *permanent* ankle torque
that those loops fought. The ankle saturated and the robot **fell backwards in
0.76 s with no push at all**.

The lesson generalises: the reference of a balance controller has to be the
posture's actual equilibrium, not a convenient geometric point.

## Standing still

| Quantity | Joint PD | Capture-point PD |
| --- | --- | --- |
| CoM forward offset | +9.0 mm | **−1.8 mm** |
| DCM forward offset | +8.7 mm | **−2.1 mm** |
| Base height | 0.3215 m | 0.3204 m |
| Ankle torque used | 3.2 % | 25.7 % |

The capture-point controller regulates to its reference instead of drifting to
wherever the posture settles, and pays about eight times the standing torque to
do it. That is a real trade, not a free improvement: holding a torque through a
position target means there is no longer a "free" equilibrium the joint can
rest at.

## Under perturbation

Same robot, same gains, same nominal posture, same measurement. The only thing
that differs is the control law.

| Push | LIPM prediction | Joint PD | Capture-point PD |
| --- | --- | --- | --- |
| 0.40 m/s | recoverable | recovers | recovers |
| 0.50 m/s | recoverable | **falls** | recovers |
| 0.55 m/s | bound is 0.548 | falls | **recovers** |
| 0.60 m/s | not recoverable | falls | fell, then recovered |
| 0.70 m/s | not recoverable | falls | falls |

Three things worth reading carefully.

**The threshold moves.** At 0.50 m/s one falls and the other does not. Nothing
about the hardware changed — same actuators, same gains, same posture. The
difference is that one controller knows where its pressure centre is.

**It exceeds the capturable bound.** 0.55 m/s against a predicted 0.548 m/s.
This does not contradict the theory, it *violates its assumption*: `ω₀ d` is
derived from the linear inverted pendulum, which assumes centroidal angular
momentum is constant, and the hip and arm strategy exists precisely to break
that. The bound is the ceiling for an ankle-only strategy, not for the robot.

**The 0.60 m/s result is ambiguous, and is reported as such.** The fall detector
fired and then cleared. That is either a genuine recovery from a large lean or a
transient threshold crossing, and **one repetition cannot distinguish them**.
Calling it "recovers at 0.60" would be overstating it.

Reproduce with:

```powershell
foreach ($c in "joint_pd","dcm") {
  foreach ($v in 0.40,0.50,0.55,0.60) {
    python scripts\stand_nao.py --headless --controller $c `
           --duration 6 --push-velocity $v --push-at 3
  }
}
```

## Running it

```powershell
# no perturbation
python scripts\stand_nao.py --headless --controller dcm --duration 4

# same push protocol as the joint PD, for a like-for-like comparison
python scripts\stand_nao.py --headless --controller dcm `
       --duration 6 --push-velocity 0.30 --push-at 3
```

`--controller joint_pd` (the default) selects the first baseline. Everything
else — actuators, nominal posture, control rate, measurement — is identical
between the two, so the only difference is the control law.

## Why two baselines

A learned policy has to be measured against a *competent* controller, not a
straw man. The joint PD establishes what a fixed posture loop can do; the
capture-point controller establishes what a controller that knows the theory
can do. Anything a policy adds has to be visible on top of the second one, and
the reasons it might are set out in
[nao_stand_plan.md](nao_stand_plan.md).
