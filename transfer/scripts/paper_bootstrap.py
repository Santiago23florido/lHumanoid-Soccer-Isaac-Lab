"""Bootstrap standard errors for the methods in the paper's tables (``SE``).

On the NAO every method, on the G1 the zero-shot model and B1 (the other baselines need the
NAO's own task). Every cell of tables B1, B2, B3 and B is one capture per L4 command, 32
environments each, held to the teacher's capture at the command. The standard errors here
resample the environments of both, with replacement, and recompute what the tables read: the
speed ratio and each gate C feature's |relative deviation|, by command and averaged over the
commands, and each command's largest deviation averaged over the commands, as
``paper_figures.py`` reads it. A draw resamples the teacher's capture once and holds every
method to it, so a difference between two methods is paired on the teacher.

What that measures is the sampling error of 32 environments, and it is small: the play
captures are deterministic and their environments nearly alike. Between training seeds a
method varies far more (the zero-shot model's own seeds, ``S``, span 0.115–0.169 in the
largest deviation on rough, against an environment SE of 0.002). So the comparison the
paper can make is between seeds: for each method trained with two seeds or more, the
mean over its seeds and its standard error, the seeds' standard deviation over the root
of their number (each seed's estimate carries its environment noise with it), and its
difference from the zero-shot model's seeds with the two errors combined. With two or
three seeds that error is itself rough.

The full sample must reproduce the committed reports (``evaluate_contract.py``'s
verdicts) to a relative 1e-9, or the script stops: what is resampled is what was scored.

Writes ``transfer/paper/data/bootstrap.json``.

Usage::

    python transfer/scripts/paper_bootstrap.py [--draws 1000] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_latent import _settled  # noqa: E402
from humanoid_transfer.common.gait import gait_signature  # noqa: E402
from humanoid_transfer.common.similarity import UnitSystem  # noqa: E402
from humanoid_transfer.common.skill_state import SKILL_LAYOUT  # noqa: E402
from paper_tables import (  # noqa: E402
    COMMANDS,
    GAIT,
    GENERATOR,
    MODEL,
    ORACLE_SEEDS,
    PAPER,
    RETARGET_SEEDS,
    ROOT,
    SEEDS,
    TASK_SEEDS,
    finite,
    nao_method_files,
    read,
    task_tracker_files,
    zero_shot_pattern,
)

BASELINE = "zero-shot"
"""The method every other is differenced against."""


def groups(body: str) -> dict[str, list[str]]:
    """Each method's runs on ``body`` by training seed. The zero-shot model's are ``S``'s, its
    last stage repeated from the same start: on the NAO the adopted model is not among them,
    on the G1 it is the first."""
    return {
        "zero-shot": [f"zero-shot, seed {s}" for s, _ in SEEDS[body]],
        "B1, task tracker": [
            "B1, task tracker",
            *(f"B1, task tracker, seed {s}" for s in TASK_SEEDS[body][1:]),
        ],
        "B2, oracle": [f"B2, oracle, seed {s}" for s in ORACLE_SEEDS],
        "B2b, bounded": [f"B2b, bounded, seed {s}" for s in ORACLE_SEEDS],
        "B3, copy": ["B3, copy", *(f"B3, copy, seed {s}" for s in RETARGET_SEEDS[1:])],
    }


def method_files(body: str, skill: str) -> dict[str, tuple[str, str]]:
    """The methods compared on ``body`` and ``skill``, and the zero-shot model's seeds."""
    generator = GENERATOR[skill]
    if body == "nao":
        files = nao_method_files(skill)
    else:
        files = {"zero-shot": (zero_shot_pattern(body, MODEL[body], generator), "encoder")}
        files |= task_tracker_files(body, skill)
    seeds = {
        f"zero-shot, seed {s}": (zero_shot_pattern(body, tag, generator), "encoder")
        for s, tag in SEEDS[body]
    }
    return files | seeds


class Capture:
    """One capture's per-environment series, enough to score any resample of its columns."""

    def __init__(self, path: str) -> None:
        data = np.load(ROOT / path)
        units = UnitSystem(float(data["leg_length"]), float(data["total_mass"]))
        self.z = data["skill_state"].astype(float)
        self.done = data["done"]
        running = int(data["settle_steps"]) if "settle_steps" in data.files else 0
        self.settled = _settled(self.done, already_running=running)
        # In V = sqrt(g L), the unit the verdicts compare speeds in.
        self.forward = SKILL_LAYOUT.unpack(self.z)["com_velocity"][..., 0]
        self.dt_star = float(data["control_dt"]) / units.time
        self.envs = self.done.shape[1]

    def measure(self, envs: np.ndarray) -> dict:
        settled = self.settled[:, envs]
        signature = gait_signature(self.z[:, envs], self.done[:, envs], self.dt_star)
        return {
            "speed": float(self.forward[:, envs][settled].mean()) if settled.any() else None,
            "stride_length": signature.stride_length,
            "stride_period": signature.stride_period,
            "duty_factor": list(signature.duty_factor),
        }


def deviation(run: dict, teacher: dict) -> dict:
    """``evaluate_latent.verdict``'s speed ratio and |relative deviations|."""
    out = {"speed_ratio": 0.0 if run["speed"] is None else run["speed"] / teacher["speed"]}
    for key in ("stride_length", "stride_period"):
        out[key] = abs(run[key] - teacher[key]) / teacher[key]
    for side, (value, reference) in enumerate(
        zip(run["duty_factor"], teacher["duty_factor"], strict=True)
    ):
        out[f"duty_factor_{side}"] = abs(value - reference) / reference
    return out


def readings(by_command: list[dict]) -> dict:
    """What the tables read from a method's three commands."""
    out = {f"speed_ratio_{c}": d["speed_ratio"] for c, d in zip(COMMANDS, by_command, strict=True)}
    for f in GAIT:
        out[f] = float(np.mean([d[f] for d in by_command]))
    largest = [np.nanmax([d[f] for f in GAIT]) for d in by_command]
    out["largest"] = float(np.mean(largest))
    return out


def score(
    methods: dict[str, list[Capture]],
    teachers: list[Capture],
    picks: dict[str, list[np.ndarray]],
    teacher_picks: list[np.ndarray],
) -> dict:
    """Every method's readings on the environments ``picks``, held to the teacher's on
    ``teacher_picks``, command by command."""
    held = [t.measure(p) for t, p in zip(teachers, teacher_picks, strict=True)]
    return {
        name: readings([
            deviation(c.measure(p), h) for c, p, h in zip(captures, picks[name], held, strict=True)
        ])
        for name, captures in methods.items()
    }  # fmt: skip


def spread(values: np.ndarray) -> float:
    """The standard deviation of the defined draws; NaN with fewer than two."""
    defined = values[np.isfinite(values)]
    return float(defined.std()) if defined.size > 1 else float("nan")


def close(a: float, b: float) -> bool:
    if not (math.isfinite(a) and math.isfinite(b)):
        return math.isfinite(a) == math.isfinite(b)
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=PAPER / "data" / "bootstrap.json")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    report = {"draws": args.draws, "seed": args.seed, "baseline": BASELINE, "bodies": {}}
    for body, skill in [(body, skill) for body in ("nao", "g1") for skill in GENERATOR]:
        # Each method's capture per command, and the teacher's, which every report shares.
        methods: dict[str, list[Capture]] = {}
        teachers: list[Capture] = []
        for name, (pattern, run) in method_files(body, skill).items():
            reports = [read(pattern.format(c)) for c in COMMANDS]
            if any(r is None for r in reports):
                continue
            held_to = [r["runs"]["teacher"]["file"] for r in reports]
            if not teachers:
                teachers = [Capture(f) for f in held_to]
                teacher_files = held_to
            assert held_to == teacher_files, f"{name} on {body}/{skill} is held to another teacher"
            methods[name] = [Capture(r["runs"][run]["file"]) for r in reports]
            # The full sample is what was scored.
            for capture, teacher, r in zip(methods[name], teachers, reports, strict=True):
                every = np.arange(capture.envs)
                d = deviation(capture.measure(every), teacher.measure(np.arange(teacher.envs)))
                verdict = r["runs"][run]["verdict"]
                expected = {"speed_ratio": verdict["speed_ratio"], **verdict["signature_deviation"]}
                for key, value in expected.items():
                    assert close(d[key], value), (body, skill, name, key, d[key], value)
            print(f"{body}/{skill}: {name} reproduces its reports")

        full = score(
            methods,
            teachers,
            {n: [np.arange(c.envs) for c in cs] for n, cs in methods.items()},
            [np.arange(t.envs) for t in teachers],
        )
        draws = {name: {key: [] for key in full[name]} for name in methods}
        for k in range(args.draws):
            picks = {n: [rng.integers(0, c.envs, c.envs) for c in cs] for n, cs in methods.items()}
            teacher_picks = [rng.integers(0, t.envs, t.envs) for t in teachers]
            drawn = score(methods, teachers, picks, teacher_picks)
            for name, values in drawn.items():
                for key, value in values.items():
                    draws[name][key].append(value)
            if (k + 1) % 100 == 0:
                print(f"{body}/{skill}: {k + 1}/{args.draws} draws")

        entries = {}
        for name in methods:
            entry = {"point": full[name], "se": {}, "undefined_draws": {}}
            for key, values in draws[name].items():
                values = np.asarray(values, dtype=float)
                entry["se"][key] = spread(values)
                entry["undefined_draws"][key] = int((~np.isfinite(values)).sum())
            if name != BASELINE and BASELINE in methods:
                entry["minus_baseline"] = {}
                for key, values in draws[name].items():
                    diff = np.asarray(values, dtype=float) - np.asarray(draws[BASELINE][key])
                    entry["minus_baseline"][key] = {
                        "difference": full[name][key] - full[BASELINE][key],
                        "se": spread(diff),
                    }
            entries[name] = entry

        seeds = {}
        for group, names in groups(body).items():
            present = [n for n in names if n in methods]
            if len(present) < 2:
                continue
            seeds[group] = {"runs": present, "mean": {}, "se": {}}
            for key in full[present[0]]:
                values = np.array([full[n][key] for n in present], dtype=float)
                seeds[group]["mean"][key] = float(values.mean())
                seeds[group]["se"][key] = float(values.std(ddof=1) / np.sqrt(len(values)))
        for group, entry in seeds.items():
            if group == BASELINE or BASELINE not in seeds:
                continue
            base = seeds[BASELINE]
            entry["minus_baseline"] = {
                key: {
                    "difference": entry["mean"][key] - base["mean"][key],
                    "se": float(np.hypot(entry["se"][key], base["se"][key])),
                }
                for key in entry["mean"]
            }
        report["bodies"].setdefault(body, {})[skill] = {"runs": entries, "seeds": seeds}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(finite(report), indent=2) + "\n", encoding="utf-8")
    cases = [(f"{b}/{s}", by) for b, by_skill in report["bodies"].items() for s, by in by_skill.items()]  # fmt: skip  # noqa: E501
    for case, by in cases:
        for name, e in by["runs"].items():
            line = f"{case:<10} {name:<26} largest {e['point']['largest']:.3f} ± {e['se']['largest']:.3f}"  # noqa: E501
            if "minus_baseline" in e:
                d = e["minus_baseline"]["largest"]
                line += f"   minus zero-shot {d['difference']:+.3f} ± {d['se']:.3f}"
            print(line)
        for group, e in by["seeds"].items():
            line = (
                f"{case:<10} {group + ' (' + str(len(e['runs'])) + ' seeds)':<26} largest "
                f"{e['mean']['largest']:.3f} ± {e['se']['largest']:.3f}"
            )
            if "minus_baseline" in e:
                d = e["minus_baseline"]["largest"]
                line += f"   minus zero-shot {d['difference']:+.3f} ± {d['se']:.3f}"
            print(line)
    print(f"written {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
