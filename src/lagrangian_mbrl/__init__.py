"""lagrangian_mbrl — Physically-Informed Neural Network for Franka arm dynamics.

Subpackages
-----------
- :mod:`lagrangian_mbrl.models` : dynamics models (Deep Lagrangian Network +
  unstructured MLP baseline).
- :mod:`lagrangian_mbrl.envs`   : analytic rigid-body simulators.
- :mod:`lagrangian_mbrl.eval`   : accuracy and physical-consistency metrics.
- :mod:`lagrangian_mbrl.theory` : complexity proxy κ and LQR surrogate.
- :mod:`lagrangian_mbrl.utils`  : seeding and logging helpers.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
