"""Unitree G1 --- the capable source embodiment.

The G1 is where a skill is learned first, because it can actually perform it:
29 actuated joints, roughly 35 kg, 1.3 m tall, and actuators with the torque
density to walk dynamically. Isaac Lab ships the model and a velocity-tracking
locomotion task, so nothing here has to re-derive a robot from a URDF the way
the NAO package does.

It is deliberately swappable. ``SOURCE_ROBOT_CFG`` in :mod:`g1.assets.g1`
selects which Unitree model backs the source embodiment, and the H1 is a
one-line change if the 29 degrees of freedom turn out to be more than the
transfer study needs.

Contents:

``assets``
    Articulation configuration, wrapping the Isaac Lab asset.
``tasks``
    Locomotion environments used to produce the teacher policy.
"""
