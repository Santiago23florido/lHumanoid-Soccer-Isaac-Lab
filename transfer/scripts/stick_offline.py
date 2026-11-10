"""W9's offline pre-check: contact-consistent plans drawn from the teachers' own histories.

For ``flat_dagger2`` and ``rough_b4`` (each on its teacher's three play captures), one
window every ``--stride`` steps of each episode, read with the contract's history noise as
``sampler_sharpness.py`` reads it. Eight plans are drawn per window (the generator's
deployed integrator, on its own grid), and four plans are compared:

* ``as_drawn`` -- the first draw;
* ``ground`` -- W8's ``ground_loaded_feet``;
* ``stick`` -- ``stick_loaded_feet`` (grounded, and a loaded foot's velocity zero);
* ``best_of_8`` -- the draw whose loaded feet are most consistent: the smallest mean of
  ``(h / h90)^2 + (|v| / v90)^2`` over its loaded samples, ``h90`` and ``v90`` being the
  teacher's own 90th percentiles of a loaded foot's sole height and horizontal speed.

For each: the loaded feet's horizontal speed and sole height; the duty, light share and
contact changes per plan (``sampler_sharpness.load_stats``); the error against the future
the teacher walked from that window, per group of fields, in the generator's normalised
units; and the stance drift -- how far a loaded foot moves in the world over one stance in
the plan (CoM displacement from the plan's own velocity, plus the change of the foot's
position from the CoM; yaw ignored).

    python transfer/scripts/stick_offline.py --output transfer/results/stick_offline.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(REPO_ROOT / "source" / "humanoid_transfer"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from humanoid_transfer.common.contact import foot_height  # noqa: E402
from humanoid_transfer.common.similarity import UnitSystem  # noqa: E402
from humanoid_transfer.common.skill_state import (  # noqa: E402
    DEFAULT_CONTACT_THRESHOLD,
    SKILL_LAYOUT,
)
from humanoid_transfer.transfer.latent.contract import load_contract  # noqa: E402
from humanoid_transfer.transfer.latent.plan_contact import (  # noqa: E402
    ground_loaded_feet,
    stick_loaded_feet,
)
from plan_contact import TEACHERS  # noqa: E402
from sampler_sharpness import load_stats  # noqa: E402

CONTRACTS = {
    "flat": "logs/contract/flat_dagger2_zs_g1_warm_r3_logload_sym/generator.pt",
    "rough": "logs/contract/rough_b4_zs_g1_warm_r3_logload_sym/generator.pt",
}
DRAWS = 8
LOADS = SKILL_LAYOUT.slice_of("foot_load")
VELOCITY = SKILL_LAYOUT.slice_of("foot_velocity")
POSITION = SKILL_LAYOUT.slice_of("foot_position")
COM_VELOCITY = SKILL_LAYOUT.slice_of("com_velocity")
FEET_Z = [POSITION.start + 2, POSITION.start + 5]


def loaded_feet(z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mask, horizontal speed and sole height of loaded feet, ``(..., 2)`` each."""
    fields = SKILL_LAYOUT.unpack(z)
    loaded = fields["foot_load"] > DEFAULT_CONTACT_THRESHOLD
    speed = np.linalg.norm(fields["foot_velocity"][..., :2], axis=-1)
    return loaded, speed, foot_height(z)


def residual(z: np.ndarray, h90: float, v90: float) -> np.ndarray:
    """``(B,)`` mean over each plan's loaded samples of ``(h/h90)^2 + (v/v90)^2``."""
    loaded, speed, height = loaded_feet(z)
    cost = (height / h90) ** 2 + (speed / v90) ** 2
    count = loaded.sum(axis=(-2, -1))
    return np.where(count > 0, (cost * loaded).sum(axis=(-2, -1)) / np.maximum(count, 1), 0.0)


def stance_drift(z: np.ndarray, step: float) -> list[float]:
    """World displacement of a loaded foot over each complete stance in ``z`` ``(B, K, 27)``."""
    fields = SKILL_LAYOUT.unpack(z)
    com = np.cumsum(fields["com_velocity"][..., :2] * step, axis=-2)
    world = com[..., None, :] + fields["foot_position"][..., :2]
    loaded = fields["foot_load"] > DEFAULT_CONTACT_THRESHOLD
    drifts = []
    for b in range(z.shape[0]):
        for side in range(2):
            run = loaded[b, :, side]
            start = None
            for k in range(len(run) + 1):
                on = k < len(run) and run[k]
                if on and start is None:
                    start = k
                elif not on and start is not None:
                    if start > 0 and k < len(run) and k - start >= 2:  # bounded on both sides
                        delta = world[b, k - 1, side] - world[b, start, side]
                        drifts.append(float(np.linalg.norm(delta)))
                    start = None
    return drifts


def describe(z: np.ndarray, future: np.ndarray, norm, step: float) -> dict:
    loaded, speed, height = loaded_feet(z)
    zn = norm(torch.as_tensor(z, dtype=torch.float32)).numpy()
    fn = norm(torch.as_tensor(future, dtype=torch.float32)).numpy()
    error = (zn - fn) ** 2
    other = np.ones(SKILL_LAYOUT.size, dtype=bool)
    other[VELOCITY] = False
    other[FEET_Z] = False
    drift = stance_drift(z, step)
    return {
        "loaded_speed_median": float(np.median(speed[loaded])),
        "loaded_height_p90": float(np.quantile(height[loaded], 0.9)),
        **load_stats(z[..., LOADS]),
        "rms_vs_teacher": {
            "foot_velocity": float(np.sqrt(error[..., VELOCITY].mean())),
            "foot_z": float(np.sqrt(error[..., FEET_Z].mean())),
            "other_fields": float(np.sqrt(error[..., other].mean())),
            "com_velocity": float(np.sqrt(error[..., COM_VELOCITY].mean())),
            "foot_load": float(np.sqrt(error[..., LOADS].mean())),
        },
        "stance_drift_median": float(np.median(drift)) if drift else None,
        "stances": len(drift),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--stride", type=int, default=14, help="steps between windows")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    report: dict = {"draws": DRAWS, "stride": args.stride}
    for skill, files in TEACHERS.items():
        torch.manual_seed(args.seed)
        contract = load_contract(REPO_ROOT / CONTRACTS[skill], device)
        generator = contract.generator
        norm = generator.skill_norm
        history, horizon = generator.history, generator.horizon
        windows, futures, commands = [], [], []
        for name in files:
            data = np.load(REPO_ROOT / name)
            units = UnitSystem(float(data["leg_length"]), float(data["total_mass"]))
            scale = np.array([1.0 / units.velocity, 1.0 / units.velocity, units.time])
            z = data["skill_state"].astype(np.float32)
            done = data["done"].astype(bool)
            for env in range(z.shape[1]):
                for t in range(100 + history, z.shape[0] - horizon, args.stride):
                    if done[t - history : t + horizon, env].any():
                        continue
                    windows.append(z[t - history + 1 : t + 1, env])
                    futures.append(z[t + 1 : t + 1 + horizon, env])
                    commands.append(data["command"][t, env] * scale)
        future = np.stack(futures).astype(float)
        teacher_loaded, teacher_speed, teacher_height = loaded_feet(future)
        h90 = float(np.quantile(teacher_height[teacher_loaded], 0.9))
        v90 = float(np.quantile(teacher_speed[teacher_loaded], 0.9))
        window = norm(torch.as_tensor(np.stack(windows), device=device))
        window = window + contract.sigma * torch.randn_like(window)
        command = torch.as_tensor(np.stack(commands), dtype=torch.float32, device=device)
        draws = []
        with torch.no_grad():
            for d in range(DRAWS):
                noise = torch.randn(
                    len(window), generator.noise_size, device=device,
                    generator=torch.Generator(device).manual_seed(args.seed + 1 + d),
                )  # fmt: skip
                draws.append(generator.sample(window, command, noise).float())
        first = draws[0]
        plain = [norm.inverse(p).double().cpu().numpy() for p in draws]
        scores = np.stack([residual(p, h90, v90) for p in plain])  # (DRAWS, B)
        best = np.argmin(scores, axis=0)
        chosen = np.stack(plain)[best, np.arange(len(best))]
        norm_cpu = norm.cpu()
        first_cpu = first.cpu()
        variants = {
            "as_drawn": plain[0],
            "ground": norm_cpu.inverse(ground_loaded_feet(first_cpu, norm_cpu)).double().numpy(),
            "stick": norm_cpu.inverse(stick_loaded_feet(first_cpu, norm_cpu)).double().numpy(),
            "best_of_8": chosen,
        }
        step = float(generator.grid_step)
        entry = {
            "contract": CONTRACTS[skill],
            "windows": len(windows),
            "teacher_h90": h90,
            "teacher_v90": v90,
            "teacher_future": describe(future, future, norm_cpu, step),
            "residual": {
                "as_drawn_median": float(np.median(scores[0])),
                "best_of_8_median": float(np.median(scores.min(axis=0))),
            },
        }
        for name, z in variants.items():
            entry[name] = describe(z, future, norm_cpu, step)
        report[skill] = entry
        norm.to(device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for skill in TEACHERS:
        e = report[skill]
        print(f"{skill}: {e['windows']} windows, teacher h90 {e['teacher_h90']:.4f}", end="")
        print(f" v90 {e['teacher_v90']:.4f}")
        for name in ("teacher_future", "as_drawn", "ground", "stick", "best_of_8"):
            d = e[name]
            r = d["rms_vs_teacher"]
            print(
                f"  {name:15s} speed {d['loaded_speed_median']:.4f}"
                f"  h90 {d['loaded_height_p90']:.4f}"
                f"  duty {np.mean(d['duty']):.3f}  light {np.mean(d['light_share']):.3f}"
                f"  rms vel {r['foot_velocity']:.3f} z {r['foot_z']:.3f}"
                f" other {r['other_fields']:.3f}  drift {d['stance_drift_median']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
