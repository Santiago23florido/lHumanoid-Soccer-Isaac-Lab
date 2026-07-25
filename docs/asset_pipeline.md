# Asset Pipeline

## Robot

1. Select the humanoid model and license.
2. Import or convert the robot to USD.
3. Verify articulation roots, joint names, masses, inertias, limits, and actuator
   settings in Isaac Sim.
4. Add the Isaac Lab asset config in `humanoid_soccer_lab/assets/humanoids.py`.
5. Add a joint-order map that matches the policy action vector.

## Field

1. Define the field dimensions and goal geometry.
2. Keep collision geometry simple for training speed.
3. Add visual materials only after the physics scene is stable.

## Ball

1. Start with a sphere primitive config for fast iteration.
2. Calibrate mass, friction, restitution, and damping.
3. Replace with a USD asset only if visuals or custom collision are needed.
