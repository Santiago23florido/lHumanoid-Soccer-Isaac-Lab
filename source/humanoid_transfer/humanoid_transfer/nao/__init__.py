"""NAO H25 V5.0 --- the constrained target embodiment.

The NAO is the robot this project wants a skill to end up on, and the reason the
transfer question exists: 5.3 kg, 0.58 m, weak actuators, and a zero-step
capturable speed between 0.443 and 0.827 m/s depending on the direction of the
push. Learning a dynamic skill directly on it is hard for reasons that are
physical rather than algorithmic.

Contents:

``assets``
    Model, derived parameters and the URDF-to-USD pipeline.
``controllers``
    Model-based baselines, which set the bar a learned policy has to clear.
``tasks``
    Reinforcement learning environments.
"""
