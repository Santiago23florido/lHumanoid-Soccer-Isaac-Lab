"""Check the analytic z* fields (``pact/anchors.py``) against the true z* of captures.

For each body, every step of its expert captures (``logs/pact/rollouts/<body>/flat_*.npz``,
recorded with ``--joints``) is passed through ``KinematicAnchors`` from the joint angles
and speeds, the projected gravity and the base's angular velocity alone, and compared
with the ``z*`` the simulator's link states give. Errors are reported per field as an
RMSE in units of the field's own standard deviation in the capture, and as a fraction
of variance explained (R^2). The foot velocity is compared with the true CoM velocity
added back; the height only on steps with a loaded foot.

Usage::

    python transfer/scripts/pact_check_anchors.py --bodies g1,nao,h1,cassie
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "source" / "humanoid_transfer"))

import torch  # noqa: E402
from humanoid_transfer.common.skill_state import SKILL_LAYOUT  # noqa: E402
from humanoid_transfer.transfer.latent.pact.anchors import KinematicAnchors  # noqa: E402

FIELDS = (
    "height",
    "gravity",
    "base_angular_velocity",
    "foot_position",
    "foot_velocity",
    "angular_momentum",
)


def sole_override(body: str):
    """The sole points the captures used, when they are not the card's (the NAO: URDF soles)."""
    if body != "nao":
        return None
    from humanoid_transfer.nao.assets import nao_kinematics as nk

    return [list(map(float, s)) for s in nk.sole_points_in_foot(0.0)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bodies", default="g1,nao,h1,cassie")
    parser.add_argument("--captures", default="logs/pact/rollouts/{body}/flat_*.npz")
    parser.add_argument("--max-steps", type=int, default=300, help="Steps per capture (all envs).")
    parser.add_argument(
        "--output", type=Path, default=REPO / "transfer/results/icml_anchors_check.json"
    )
    args = parser.parse_args()
    registry = json.loads((REPO / "transfer/bodies/bodies.json").read_text("utf-8"))
    report = {"captures": args.captures, "max_steps": args.max_steps, "bodies": {}}
    for body in [b for b in args.bodies.split(",") if b]:
        entry = registry[body]
        card = json.loads((REPO / entry["card"]).read_text("utf-8"))
        anchors = KinematicAnchors(card, entry.get("leg_length_unit"), sole_override(body))
        truth, predicted = {f: [] for f in FIELDS}, {f: [] for f in FIELDS}
        for path in sorted(REPO.glob(args.captures.format(body=body))):
            d = np.load(path)
            if tuple(str(n) for n in d["joint_names"]) != tuple(card["joints"]["names"]):
                raise SystemExit(f"{path}: joint order differs from the card")
            steps = min(args.max_steps, d["joint_pos"].shape[0])

            def flat(key, data=d, n=steps):
                return torch.as_tensor(data[key][:n].reshape(-1, data[key].shape[-1]))

            with torch.no_grad():
                out = anchors(
                    flat("joint_pos"),
                    flat("joint_vel"),
                    flat("projected_gravity"),
                    flat("base_ang_vel"),
                )
            z = SKILL_LAYOUT.unpack(d["skill_state"][:steps].reshape(-1, SKILL_LAYOUT.size))
            loaded = (z["foot_load"] > 0.05).any(axis=-1)
            for field in FIELDS:
                true = np.asarray(z[field], dtype=np.float64).reshape(len(loaded), -1)
                if field == "foot_velocity":
                    pred = out["foot_velocity_relative"].numpy().reshape(len(loaded), 2, 3)
                    pred = (pred + z["com_velocity"][:, None, :]).reshape(len(loaded), -1)
                else:
                    pred = out[field].numpy().reshape(len(loaded), -1)
                if field == "height":
                    true, pred = true[loaded], pred[loaded]
                truth[field].append(true)
                predicted[field].append(pred)
        rows = {}
        for field in FIELDS:
            t, p = np.concatenate(truth[field]), np.concatenate(predicted[field])
            std = np.maximum(t.std(axis=0), 1e-9)
            rmse = np.sqrt(((p - t) ** 2).mean(axis=0))
            r2 = 1 - ((p - t) ** 2).sum(axis=0) / np.maximum(
                ((t - t.mean(0)) ** 2).sum(axis=0), 1e-12
            )
            rows[field] = {
                "rmse_over_std": float(np.mean(rmse / std)),
                "r2_min": float(r2.min()),
                "max_abs": float(np.abs(p - t).max()),
            }
        report["bodies"][body] = rows
        print(body, {f: round(v["rmse_over_std"], 4) for f, v in rows.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"written {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
