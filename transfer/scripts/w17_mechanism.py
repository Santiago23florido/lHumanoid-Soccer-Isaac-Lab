"""Where W17's deviation came from (``skill_space.md`` §10.18, read after its result).

Three measurements, all offline, on captures already on disk:

1. **Coverage** -- for each L4 command of rough and of the base tasks, which expert's training
   steps lie near it in the dimensionless command space (within 0.02 on every axis, as a
   Euclidean radius on the scaled axes), with the command the expert was given and with the one
   its future achieved (``latent/hindsight.py``, span 50); and the height and loaded share of
   those steps.
2. **Postures** -- each expert's mean height (in leg lengths) and loaded share over its DART
   capture, and the rough and flat teachers' over their play captures.
3. **What the deployed path asked** -- on one tracker per body (G1 s41, NAO s31), for each W17
   arm and rough command, the mean height and loaded share of the plans (``contract_due``) and of
   the body, over the recorded steps (after the ladder's 100 settle steps).

    python transfer/scripts/w17_mechanism.py --output transfer/results/w17_mechanism.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(REPO_ROOT / "source" / "humanoid_transfer"))

from humanoid_transfer.common.similarity import UnitSystem  # noqa: E402
from humanoid_transfer.common.skill_state import DEFAULT_CONTACT_THRESHOLD  # noqa: E402
from humanoid_transfer.transfer.latent.hindsight import SPAN, hindsight_commands  # noqa: E402
from humanoid_transfer.transfer.latent.sequences import load_streams  # noqa: E402

EXPERTS = {
    "walk": "g1/rollouts/teacher_dart.npz",
    "flat": "g1/rollouts/flat_dart.npz",
    "backward": "g1/rollouts/base_backward_dart.npz",
    "sidestep": "g1/rollouts/base_sidestep_dart.npz",
    "jog": "g1/rollouts/base_jog_dart.npz",
}
TEACHERS = {
    "rough": "g1/rollouts/rough_play_forward.npz",
    "flat": "g1/rollouts/flat_play_forward.npz",
}
COMMANDS = {  # G1 units, as the ladders run them
    "rough_forward": (0.25, 0.0, 0.0),
    "rough_fast": (0.6, 0.0, 0.0),
    "rough_turning": (0.3, 0.0, 0.3),
    "sidestep_left": (0.0, 0.25, 0.0),
    "sidestep_left_fast": (0.0, 0.45, 0.0),
    "sidestep_right": (0.0, -0.35, 0.0),
    "backward_mid": (-0.35, 0.0, 0.0),
    "jog_mid": (1.25, 0.0, 0.0),
}
RADIUS = 0.02
HEIGHT = 0
LOAD = slice(22, 24)
ARMS = ("mt16_c", "mt17_c", "mt17_m", "rough_b4")
DEPLOYED = {"g1": "mt_g1_s41_sym", "nao": "mt_nao_s31_sym"}


def posture(z: np.ndarray) -> dict:
    return {
        "height": float(z[..., HEIGHT].mean()),
        "loaded": float((z[..., LOAD] > DEFAULT_CONTACT_THRESHOLD).mean()),
    }


def coverage() -> dict:
    names = list(EXPERTS)
    streams = load_streams([REPO_ROOT / EXPERTS[n] for n in names])
    labels = {
        "given": streams.command.numpy(),
        "relabelled": hindsight_commands(streams, SPAN).numpy(),
    }
    capture = streams.capture.numpy()
    skill = streams.skill.numpy()
    g1 = np.load(REPO_ROOT / EXPERTS["walk"])
    units = UnitSystem(float(g1["leg_length"]), float(g1["total_mass"]))
    scale = np.array([1.0 / units.velocity, 1.0 / units.velocity, units.time])
    out: dict = {}
    for command_name, command in COMMANDS.items():
        target = np.asarray(command) * scale
        entry: dict = {"command": target.tolist()}
        for label, values in labels.items():
            near = np.linalg.norm(values - target, axis=1) < RADIUS
            counts = np.bincount(capture[near], minlength=len(names))
            total = int(near.sum())
            entry[label] = {
                "steps": total,
                "share": {n: float(c / max(total, 1)) for n, c in zip(names, counts, strict=True)},
                **(posture(skill[near]) if total else {}),
            }
        out[command_name] = entry
    return out


def postures() -> dict:
    out = {}
    for name, path in {**EXPERTS, **{f"{k}_play": v for k, v in TEACHERS.items()}}.items():
        out[name] = posture(np.load(REPO_ROOT / path)["skill_state"])
    return out


def deployed() -> dict:
    out: dict = {}
    for body, tag in DEPLOYED.items():
        for arm in ARMS:
            for command in ("forward", "fast", "turning"):
                name = f"zs_{tag}_{arm}_own_auto_kin_{command}_encoder.npz"
                path = REPO_ROOT / body / "rollouts" / name
                if not path.exists():
                    continue
                data = np.load(path)
                out.setdefault(body, {}).setdefault(arm, {})[command] = {
                    "asked": posture(data["contract_due"]),
                    "body": posture(data["skill_state"]),
                }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "radius": RADIUS,
        "span": SPAN,
        "coverage": coverage(),
        "postures": postures(),
        "deployed": deployed(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(report, indent=2) + "\n").encode("utf-8"))
    for name, entry in report["coverage"].items():
        for label in ("given", "relabelled"):
            e = entry[label]
            shares = " ".join(f"{k} {v:.2f}" for k, v in e["share"].items() if v >= 0.01)
            print(
                f"{name:19s} {label:10s} steps {e['steps']:6d}  {shares}"
                + (f"  height {e['height']:.3f} loaded {e['loaded']:.2f}" if e["steps"] else "")
            )
    for name, p in report["postures"].items():
        print(f"posture {name:13s} height {p['height']:.3f} loaded {p['loaded']:.3f}")
    for body, arms in report["deployed"].items():
        for arm, commands in arms.items():
            for command, e in commands.items():
                print(
                    f"deployed {body:3s} {arm:9s} {command:8s} asked height "
                    f"{e['asked']['height']:.3f} loaded {e['asked']['loaded']:.2f}; body height "
                    f"{e['body']['height']:.3f} loaded {e['body']['loaded']:.2f}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
