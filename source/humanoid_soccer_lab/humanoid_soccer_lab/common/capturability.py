"""Linear inverted pendulum theory, expressed for any legged robot.

The NAO package derived these quantities from its own URDF and kept the
derivation and the theory in the same module. That was fine while there was one
robot. With a second embodiment the theory has to be stated once, in terms that
do not mention a particular machine: a centre-of-mass height, a support polygon
and a point inside it.

Everything here takes those as arguments and returns numbers. No robot, no
simulator, no Isaac Lab.

References
----------
Kajita et al. (2001), the linear inverted pendulum model.
Pratt et al. (2006), capture points.
Koolen et al. (2012), N-step capturability; the direction dependence of the
zero-step bound is their point, not an observation of this project.
Englsberger et al. (2015), divergent component of motion.
"""

from __future__ import annotations

import math

import numpy as np

GRAVITY = 9.81
"""Standard gravity, in m/s^2."""


def lipm_omega(com_height: float, gravity: float = GRAVITY) -> float:
    """Natural frequency of the linear inverted pendulum, in rad/s.

    ``omega_0 = sqrt(g / z_c)``. It sets every timescale in balance control: the
    error doubling time is ``ln(2) / omega_0``, and the control rate has to be
    fast against it.

    Note the direction of the dependence, because it is counter-intuitive and
    matters when comparing controllers: a *lower* centre of mass gives a
    *larger* ``omega_0``, hence a larger capturable velocity. A robot that
    crouches makes itself harder to push over, and a controller that crouches
    while being measured will beat a bound derived at nominal height without
    doing anything clever.
    """
    if com_height <= 0.0:
        raise ValueError(f"com_height must be positive, got {com_height}")
    return math.sqrt(gravity / com_height)


def dcm(com_xy: np.ndarray, com_vel_xy: np.ndarray, omega: float) -> np.ndarray:
    """Divergent component of motion, ``xi = x + x_dot / omega``.

    The linear inverted pendulum splits into a stable part and an unstable part,
    and this is the unstable one::

        xi_dot = omega * (xi - p)

    with ``p`` the centre of pressure. All of the instability of standing lives
    in this single first-order equation, which is why balance control reduces to
    steering ``xi`` by placing ``p`` -- and why ``p`` being confined to the
    support polygon is the entire constraint.
    """
    return np.asarray(com_xy, dtype=float) + np.asarray(com_vel_xy, dtype=float) / omega


def ray_box_distance(
    origin: tuple[float, float],
    angle: float,
    box_x: tuple[float, float],
    box_y: tuple[float, float],
) -> float:
    """Distance from ``origin`` to the edge of an axis-aligned box along ``angle``.

    The slab method. For each axis the ray is clipped against the two bounding
    planes and the nearest positive exit is kept; a direction component of zero
    contributes no constraint on that axis.

    Args:
        origin: Point inside the box, in metres.
        angle: Heading in radians, measured from the +x axis.
        box_x: ``(low, high)`` bounds on x.
        box_y: ``(low, high)`` bounds on y.

    Returns:
        Distance to the boundary, in metres.

    Raises:
        ValueError: If ``origin`` lies outside the box, where the question has
            no meaning.
    """
    x0, y0 = origin
    if not (box_x[0] <= x0 <= box_x[1] and box_y[0] <= y0 <= box_y[1]):
        raise ValueError(
            f"origin {origin} is outside the box x={box_x} y={box_y}; "
            "the distance to the edge along a ray is only defined from inside"
        )

    direction = (math.cos(angle), math.sin(angle))
    distance = math.inf
    for position, component, (low, high) in zip(
        (x0, y0), direction, (box_x, box_y), strict=True
    ):
        if abs(component) < 1e-12:
            continue
        bound = high if component > 0.0 else low
        distance = min(distance, (bound - position) / component)
    return distance


def capturable_velocity(distance_to_edge: float, omega: float) -> float:
    """Zero-step capturable centre-of-mass speed, in m/s.

    ``v_max = omega_0 * d``. Above it the divergent component leaves the support
    polygon and no admissible centre of pressure brings it back, so the robot
    must step, widen its stance or change its centroidal angular momentum.

    Two standing assumptions are worth restating every time this is used,
    because forgetting them turns the bound into a false ceiling:

    1. Centroidal angular momentum is constant. A controller that swings its
       arms or trunk breaks this deliberately and can exceed the bound.
    2. Contacts do not change. It is a *zero-step* bound, so it says nothing
       about a robot that is allowed to take a step -- comparing a stepping
       controller against it is not a result, it is a category error.
    """
    return omega * distance_to_edge


def capturable_velocity_in_direction(
    angle: float,
    com_xy: tuple[float, float],
    com_height: float,
    polygon_x: tuple[float, float],
    polygon_y: tuple[float, float],
    gravity: float = GRAVITY,
) -> float:
    """Zero-step capturable speed for a push along ``angle``, in m/s.

    The bound is usually quoted as a single number. It is not one: the support
    polygon is rarely symmetric and the centre of mass is rarely at its centre,
    so ``d`` -- and therefore the bound -- is a function of the direction of the
    push.

    The size of that variation is not a detail. On the NAO the ratio between the
    strongest and weakest direction is 1.87, which is large enough that a
    perturbation curriculum specified as an absolute speed will be trivial in
    one direction and physically unwinnable in another.
    """
    omega = lipm_omega(com_height, gravity)
    distance = ray_box_distance(com_xy, angle, polygon_x, polygon_y)
    return capturable_velocity(distance, omega)
