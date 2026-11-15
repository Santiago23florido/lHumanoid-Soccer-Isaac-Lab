"""The tracker's actor in NumPy: normalise, a multilayer perceptron, and its mirror average.

Read from an ``.npz`` written by ``nao/scripts/export_runtime.py`` from the
exported TorchScript actor:

* ``mean``, ``scale``: the input normaliser, ``(o - mean) / scale``;
* ``weight_<i>``, ``bias_<i>``: the layers, ELU between them;
* optionally ``mirror_linear``, ``mirror_offset``, ``action_joints``,
  ``action_offset``: the sagittal mirror, for the symmetrised actor
  ``½ (π(o) + P π(A o + b) + c)``.

A student (``nao/scripts/student_bundle.py``) adds an adaptation module that
estimates the robot's parameters from a longer history, which its observation
carries after the actor's own inputs:

* ``adapt_mean``, ``adapt_scale``, ``adapt_weight_<i>``, ``adapt_bias_<i>``,
  ``adapt_mirror_linear``, ``adapt_mirror_offset``: ``φ`` and its mirror, the
  estimate ``½ (φ(h) + φ(M h + m))`` clipped to ``[-1, 1]``;
* ``params_start``: where the actor's ``robot_params`` inputs begin; the estimate,
  or parameters given instead (:meth:`Actor.__call__`), is written there.

``tests/test_nao_runtime.py`` holds it to the TorchScript actor it came from.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def elu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0.0, x, np.expm1(np.minimum(x, 0.0)))


def _layers(arrays: dict, prefix: str = "") -> list:
    layers = []
    i = 0
    while f"{prefix}weight_{i}" in arrays:
        layers.append(
            (
                np.asarray(arrays[f"{prefix}weight_{i}"], np.float64),
                np.asarray(arrays[f"{prefix}bias_{i}"], np.float64),
            )
        )
        i += 1
    return layers


def _mlp(x: np.ndarray, layers: list) -> np.ndarray:
    for k, (weight, bias) in enumerate(layers):
        x = x @ weight.T + bias
        if k < len(layers) - 1:
            x = elu(x)
    return x


class Actor:
    def __init__(self, arrays: dict[str, np.ndarray]) -> None:
        self.mean = np.asarray(arrays["mean"], dtype=np.float64)
        self.scale = np.asarray(arrays["scale"], dtype=np.float64)
        self.layers = _layers(arrays)
        self.mirror = None
        if "mirror_linear" in arrays:
            self.mirror = tuple(
                np.asarray(arrays[k], dtype=np.float64)
                for k in ("mirror_linear", "mirror_offset", "action_joints", "action_offset")
            )
        self.adaptation = None
        if "adapt_weight_0" in arrays:
            self.adaptation = {
                "mean": np.asarray(arrays["adapt_mean"], dtype=np.float64),
                "scale": np.asarray(arrays["adapt_scale"], dtype=np.float64),
                "layers": _layers(arrays, "adapt_"),
                "linear": np.asarray(arrays["adapt_mirror_linear"], dtype=np.float64),
                "offset": np.asarray(arrays["adapt_mirror_offset"], dtype=np.float64),
                "start": int(np.asarray(arrays["params_start"])),
            }
        self.last_params = None
        """The robot parameters the last call acted on (a student only), normalised."""

    @classmethod
    def load(cls, path: str | Path) -> Actor:
        with np.load(path, allow_pickle=False) as data:
            return cls({k: data[k] for k in data.files})

    @property
    def width(self) -> int:
        """The actor's own inputs (a student's observation carries its history after them)."""
        return self.mean.shape[0]

    @property
    def inputs(self) -> int:
        extra = 0 if self.adaptation is None else self.adaptation["mean"].shape[0]
        return self.width + extra

    @property
    def outputs(self) -> int:
        return self.layers[-1][1].shape[0]

    def raw(self, observation: np.ndarray) -> np.ndarray:
        return _mlp((observation - self.mean) / self.scale, self.layers)

    def estimate_params(self, observation: np.ndarray) -> np.ndarray:
        """A student's estimate of the robot's parameters from its observation's history."""
        a = self.adaptation
        history = np.asarray(observation, dtype=np.float64)[..., self.width :]

        def phi(x: np.ndarray) -> np.ndarray:
            return _mlp((x - a["mean"]) / a["scale"], a["layers"])

        reflected = history @ a["linear"].T + a["offset"]
        return np.clip(0.5 * (phi(history) + phi(reflected)), -1.0, 1.0)

    def __call__(self, observation: np.ndarray, params: np.ndarray | None = None) -> np.ndarray:
        """The action; a student acts on ``params`` (normalised) if given, else its estimate."""
        observation = np.asarray(observation, dtype=np.float64)
        if self.adaptation is not None:
            if params is None:
                params = self.estimate_params(observation)
            params = np.asarray(params, dtype=np.float64)
            params = np.broadcast_to(params, (*observation.shape[:-1], params.shape[-1]))
            observation = observation[..., : self.width].copy()
            start = self.adaptation["start"]
            observation[..., start : start + params.shape[-1]] = params
            self.last_params = np.array(params)
        if self.mirror is None:
            return self.raw(observation)
        linear, offset, joints, action_offset = self.mirror
        reflected = observation @ linear.T + offset
        return 0.5 * (self.raw(observation) + self.raw(reflected) @ joints.T + action_offset)


__all__ = ["Actor", "elu"]
