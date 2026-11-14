"""The tracker's actor in NumPy: normalise, a multilayer perceptron, and its mirror average.

Read from an ``.npz`` written by ``nao/scripts/export_runtime.py`` from the
exported TorchScript actor:

* ``mean``, ``scale``: the input normaliser, ``(o - mean) / scale``;
* ``weight_<i>``, ``bias_<i>``: the layers, ELU between them;
* optionally ``mirror_linear``, ``mirror_offset``, ``action_joints``,
  ``action_offset``: the sagittal mirror, for the symmetrised actor
  ``½ (π(o) + P π(A o + b) + c)``.

``tests/test_nao_runtime.py`` holds it to the TorchScript actor it came from.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def elu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0.0, x, np.expm1(np.minimum(x, 0.0)))


class Actor:
    def __init__(self, arrays: dict[str, np.ndarray]) -> None:
        self.mean = np.asarray(arrays["mean"], dtype=np.float64)
        self.scale = np.asarray(arrays["scale"], dtype=np.float64)
        layers = []
        i = 0
        while f"weight_{i}" in arrays:
            layers.append(
                (
                    np.asarray(arrays[f"weight_{i}"], np.float64),
                    np.asarray(arrays[f"bias_{i}"], np.float64),
                )
            )
            i += 1
        self.layers = layers
        self.mirror = None
        if "mirror_linear" in arrays:
            self.mirror = tuple(
                np.asarray(arrays[k], dtype=np.float64)
                for k in ("mirror_linear", "mirror_offset", "action_joints", "action_offset")
            )

    @classmethod
    def load(cls, path: str | Path) -> Actor:
        with np.load(path, allow_pickle=False) as data:
            return cls({k: data[k] for k in data.files})

    @property
    def inputs(self) -> int:
        return self.mean.shape[0]

    @property
    def outputs(self) -> int:
        return self.layers[-1][1].shape[0]

    def raw(self, observation: np.ndarray) -> np.ndarray:
        x = (observation - self.mean) / self.scale
        for k, (weight, bias) in enumerate(self.layers):
            x = x @ weight.T + bias
            if k < len(self.layers) - 1:
                x = elu(x)
        return x

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float64)
        if self.mirror is None:
            return self.raw(observation)
        linear, offset, joints, action_offset = self.mirror
        reflected = observation @ linear.T + offset
        return 0.5 * (self.raw(observation) + self.raw(reflected) @ joints.T + action_offset)


__all__ = ["Actor", "elu"]
