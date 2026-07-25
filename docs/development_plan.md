# Development Plan

## Phase 0: Structure

- Keep the repository installable as an Isaac Lab external extension.
- Define task, robot, field, ball, and training config placeholders.
- Add structure tests that run without Isaac Lab.

## Phase 1: Locomotion Foundation

- Implement humanoid standing and walking without the ball.
- Add contact sensors, fall termination, and command tracking.
- Train a baseline PPO policy.

## Phase 2: Ball Interaction

- Add the ball asset and reset distributions.
- Reward controlled ball approach and short dribbles.
- Add curriculum stages for ball speed and target distance.

## Phase 3: Soccer Skills

- Add shooting, defending, and goal-conditioned policies.
- Track success metrics: shots on target, possession time, recovery after contact.

## Phase 4: Team Play

- Introduce multi-agent layouts, opponent policies, and team-level rewards.
- Keep this separate from the single-agent task until base skills are reliable.
