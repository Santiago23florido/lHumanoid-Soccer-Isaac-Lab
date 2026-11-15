"""A deployment bundle of the student: the teacher's actor and the adaptation module, in NumPy.

Takes the teacher's rsl-rl checkpoint (``nao/scripts/train_teacher.sh``), the
adaptation module (``nao/scripts/train_adaptation.py``'s ``adaptation.pt``) and an
existing bundle for what does not depend on the tracker (body model, plans, units),
and writes:

* ``actor.npz`` -- the teacher's normaliser, MLP and mirror over its own inputs
  (:mod:`humanoid_transfer.nao.deploy.policy`), plus ``φ``, its normaliser and its
  mirror over the history, and where the robot parameters go (``params_start``);
* ``runtime.json`` -- the policy terms with ``robot_params`` and the adaptation history,
  the history lengths, and ``robot_params``' names and ranges, so measured values can be
  given in their own units;
* with ``--torchscript``, the same student as TorchScript, which reads
  ``NaoTrack-Sim2RealStudent-Play-v0``'s policy group (its true ``robot_params``
  ignored): for ``transfer/scripts/capture_contract.py`` and
  ``nao/scripts/export_runtime.py --actor-npz``, and checked against the NumPy actor.

Usage::

    python nao/scripts/student_bundle.py logs/rsl_rl/nao_track/<teacher>/model_<n>.pt \
        logs/rsl_rl/nao_track/<student>/adaptation.pt --like nao/bundles/s2r_r1_4997 \
        --output nao/bundles/<name> --torchscript logs/rsl_rl/nao_track/<student>/student.pt
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "humanoid_transfer"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from checkpoint_bundle import EPSILON, HISTORY_TERMS, mirror_arrays  # noqa: E402

PARAM_WIDTH = 5


def checkpoint_arrays(path: Path) -> dict[str, np.ndarray]:
    import torch

    saved = torch.load(path, map_location="cpu", weights_only=False)
    state = {k: v.double().numpy() for k, v in saved["actor_state_dict"].items() if v.ndim > 0}
    arrays = {
        "mean": state["obs_normalizer._mean"][0],
        "scale": state["obs_normalizer._std"][0] + EPSILON,
    }
    layers = sorted(
        int(k.split(".")[1]) for k in state if k.startswith("mlp.") and k.endswith(".weight")
    )
    for i, layer in enumerate(layers):
        arrays[f"weight_{i}"] = state[f"mlp.{layer}.weight"]
        arrays[f"bias_{i}"] = state[f"mlp.{layer}.bias"]
    return arrays


def adaptation_arrays(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    import torch

    saved = torch.load(path, map_location="cpu", weights_only=False)
    state = {k: v.double().numpy() for k, v in saved["state_dict"].items()}
    arrays = {"adapt_mean": state["mean"], "adapt_scale": state["scale"]}
    layers = sorted(
        int(k.split(".")[1]) for k in state if k.startswith("mlp.") and k.endswith(".weight")
    )
    for i, layer in enumerate(layers):
        arrays[f"adapt_weight_{i}"] = state[f"mlp.{layer}.weight"]
        arrays[f"adapt_bias_{i}"] = state[f"mlp.{layer}.bias"]
    return arrays, saved


def torchscript(arrays: dict[str, np.ndarray], path: Path) -> None:
    """The student as TorchScript, the NumPy actor's computation over a batch."""
    import torch
    from torch import nn

    def mlp(prefix: str) -> nn.Sequential:
        layers: list[nn.Module] = []
        i = 0
        while f"{prefix}weight_{i}" in arrays:
            weight = torch.as_tensor(arrays[f"{prefix}weight_{i}"], dtype=torch.float32)
            linear = nn.Linear(weight.shape[1], weight.shape[0])
            linear.weight.data.copy_(weight)
            linear.bias.data.copy_(
                torch.as_tensor(arrays[f"{prefix}bias_{i}"], dtype=torch.float32)
            )
            layers += [linear, nn.ELU()]
            i += 1
        return nn.Sequential(*layers[:-1])

    class Student(nn.Module):
        # No class annotations: under ``from __future__ import annotations`` TorchScript cannot
        # read them on a local class; it types the attributes from their values.
        def __init__(self) -> None:
            super().__init__()
            for key in ("mean", "scale", "mirror_linear", "mirror_offset", "action_joints",
                        "action_offset", "adapt_mean", "adapt_scale", "adapt_mirror_linear",
                        "adapt_mirror_offset"):  # fmt: skip
                self.register_buffer(key, torch.as_tensor(arrays[key], dtype=torch.float32))
            self.actor = mlp("")
            self.phi = mlp("adapt_")
            self.width = int(arrays["mean"].shape[0])
            self.start = int(arrays["params_start"])
            self.count = PARAM_WIDTH

        def estimate(self, history: torch.Tensor) -> torch.Tensor:
            reflected = history @ self.adapt_mirror_linear.T + self.adapt_mirror_offset
            direct = self.phi((history - self.adapt_mean) / self.adapt_scale)
            mirrored = self.phi((reflected - self.adapt_mean) / self.adapt_scale)
            return (0.5 * (direct + mirrored)).clamp(-1.0, 1.0)

        def raw(self, observation: torch.Tensor) -> torch.Tensor:
            return self.actor((observation - self.mean) / self.scale)

        def forward(self, observation: torch.Tensor) -> torch.Tensor:
            base = observation[:, : self.width]
            params = self.estimate(observation[:, self.width :])
            base = torch.cat(
                [base[:, : self.start], params, base[:, self.start + self.count :]], dim=-1
            )
            reflected = base @ self.mirror_linear.T + self.mirror_offset
            mirrored = self.raw(reflected) @ self.action_joints.T + self.action_offset
            return 0.5 * (self.raw(base) + mirrored)

    torch.jit.script(Student().eval()).save(str(path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("teacher", type=Path, help="The teacher's rsl-rl checkpoint.")
    parser.add_argument("adaptation", type=Path, help="train_adaptation.py's adaptation.pt.")
    parser.add_argument("--like", type=Path, required=True, help="A bundle to take the rest from.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--torchscript", type=Path, default=None)
    args = parser.parse_args()

    arrays = checkpoint_arrays(args.teacher)
    adapt, saved = adaptation_arrays(args.adaptation)
    config = json.loads((args.like / "runtime.json").read_text(encoding="utf-8"))
    terms = [(str(n), int(w)) for n, w in saved["policy_terms"]]
    names = [n for n, _ in terms]
    split = names.index("robot_params") + 1
    teacher_terms, history_terms = terms[:split], terms[split:]
    width = sum(w for _, w in teacher_terms)
    if arrays["mean"].shape[0] != width:
        raise ValueError(f"the teacher reads {arrays['mean'].shape[0]} inputs, the layout {width}")
    if adapt["adapt_mean"].shape[0] != sum(w for _, w in history_terms):
        raise ValueError("the adaptation module does not read the layout's history")
    expected = [f"{n}_adapt_history" for n in HISTORY_TERMS]
    if [n for n, _ in history_terms] != expected:
        raise ValueError(f"the history terms are {history_terms}, expected {expected}")
    arrays.update(mirror_arrays(config, teacher_terms))
    history = mirror_arrays(config, history_terms)
    arrays.update(adapt)
    arrays["adapt_mirror_linear"] = history["mirror_linear"]
    arrays["adapt_mirror_offset"] = history["mirror_offset"]
    arrays["params_start"] = np.int64(width - PARAM_WIDTH)

    config["policy_terms"] = [list(t) for t in terms]
    lengths = {}
    for n, w in terms:
        if n.endswith("_history"):
            base = n.removesuffix("_history").removesuffix("_adapt")
            lengths[n] = w // dict(terms)[base]
    config["history"] = lengths
    config["robot_params"] = {"names": saved["params"], "ranges": saved["param_ranges"]}
    config["policy"] = str(args.teacher)
    config["adaptation"] = str(args.adaptation)

    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.like / "body_model.json", args.output / "body_model.json")
    shutil.copytree(args.like / "plans", args.output / "plans", dirs_exist_ok=True)
    (args.output / "runtime.json").write_text(json.dumps(config, indent=1), encoding="utf-8")
    np.savez(args.output / "actor.npz", **arrays)
    print(f"wrote {args.output}: actor {width} inputs + history {arrays['adapt_mean'].shape[0]}")
    if args.torchscript is not None:
        import torch
        from humanoid_transfer.nao.deploy.policy import Actor

        torchscript(arrays, args.torchscript)
        total = width + arrays["adapt_mean"].shape[0]
        rng = np.random.default_rng(0)
        probe = rng.normal(size=(8, total)) * arrays["scale"].mean()
        probe[:, width:] = (
            arrays["adapt_mean"] + rng.normal(size=(8, total - width)) * adapt["adapt_scale"]
        )
        traced = torch.jit.load(str(args.torchscript))(torch.as_tensor(probe, dtype=torch.float32))
        gap = float(np.abs(traced.detach().numpy() - Actor(arrays)(probe)).max())
        print(f"wrote {args.torchscript} (against the NumPy actor: {gap:.1e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
