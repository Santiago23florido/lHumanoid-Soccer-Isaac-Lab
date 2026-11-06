"""Contact-transition fidelity of closed-loop captures, read from their L4 result files.

For every result file given (``zero_shot_*_l4_*.json``, ``retarget_*``, ``oracle_*``),
each run's capture is scored with :func:`~humanoid_transfer.common.contact.capture_fidelity`:
against its plans where the capture recorded them, on its own otherwise. The teacher's
capture is scored too, and its duty factors are recomputed at each other run's control
rate three ways (loads interpolated linearly, as ``evaluate_contract.signature_at_rate``
does; in log units; by nearest sample), which shows the resampling's bias directly.

Usage::

    python transfer/scripts/contact_fidelity.py \
        --result transfer/results/<an L4 result>.json \
        --output transfer/results/contact_fidelity_nao_r1_3498_sym_v3.json

``--glob`` takes a pattern under ``transfer/results`` instead of explicit files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(REPO_ROOT / "source" / "humanoid_transfer"))

from humanoid_transfer.common.contact import (  # noqa: E402
    LOAD_MODES,
    capture_fidelity,
    resample,
)
from humanoid_transfer.common.similarity import UnitSystem  # noqa: E402
from humanoid_transfer.common.skill_state import (  # noqa: E402
    DEFAULT_CONTACT_THRESHOLD,
    SKILL_LAYOUT,
)

_LOADS = SKILL_LAYOUT.slice_of("foot_load")


def dt_star(data) -> float:
    units = UnitSystem(float(data["leg_length"]), float(data["total_mass"]))
    return float(data["control_dt"]) / units.time


def duty_at_rate(data, rate: float) -> dict[str, list[float]]:
    """The capture's duty factors, resampled to a control step of ``rate`` body times."""
    z, done = data["skill_state"].astype(float), data["done"].astype(bool)
    out = {}
    for mode in LOAD_MODES:
        resampled, _ = resample(z, done, dt_star(data), rate, loads=mode)
        loads = resampled[..., _LOADS]
        out[mode] = [float(v) for v in (loads > DEFAULT_CONTACT_THRESHOLD).mean(axis=(0, 1))]
    return out


def score(result_path: Path) -> dict:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    runs = result["runs"]
    teacher = np.load(REPO_ROOT / runs["teacher"]["file"])
    teacher_rate = dt_star(teacher)
    report = {
        "result": result_path.relative_to(REPO_ROOT).as_posix(),
        "teacher": {
            "file": runs["teacher"]["file"],
            "dt_star": teacher_rate,
            **capture_fidelity(teacher, teacher_rate).as_dict(),
        },
        "runs": {},
    }
    rates = {}
    for name, run in runs.items():
        if name == "teacher":
            continue
        path = REPO_ROOT / run["file"]
        if not path.exists():
            report["runs"][name] = {"file": run["file"], "missing": True}
            continue
        data = np.load(path)
        rate = dt_star(data)
        rates[round(rate, 6)] = rate
        fidelity = capture_fidelity(data, rate).as_dict()
        teacher_duty = report["teacher"]["duty_body"]
        fidelity["duty_deviation"] = [
            abs(b - t) / t for b, t in zip(fidelity["duty_body"], teacher_duty, strict=True)
        ]
        report["runs"][name] = {"file": run["file"], "dt_star": rate, **fidelity}
    report["teacher"]["duty_at_rate"] = {
        f"{rate:.4f}": duty_at_rate(teacher, rate)
        for rate in rates.values()
        if abs(rate / teacher_rate - 1.0) > 0.01
    }
    return report


def _fmt(pair) -> str:
    if pair is None:
        return "—"
    return "/".join(f"{v:.3f}" if np.isfinite(v) else "nan" for v in pair)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--result", action="append", type=Path, default=[])
    parser.add_argument("--glob", action="append", default=[], help="pattern in transfer/results")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = [p if p.is_absolute() else REPO_ROOT / p for p in args.result]
    for pattern in args.glob:
        paths += sorted((REPO_ROOT / "transfer" / "results").glob(pattern))
    if not paths:
        print("no result files")
        return 2
    reports = [score(p) for p in paths]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"reports": reports}, indent=2) + "\n", encoding="utf-8")

    for report in reports:
        t = report["teacher"]
        print(f"{report['result']}")
        print(
            f"  teacher  duty {_fmt(t['duty_body'])}  light {_fmt(t['light_share_body'])}  "
            f"clearance {_fmt(t['clearance_body'])}"
        )
        for rate, modes in t["duty_at_rate"].items():
            line = "  ".join(f"{m} {_fmt(v)}" for m, v in modes.items())
            print(f"  teacher at dt* {rate}: {line}")
        for name, r in report["runs"].items():
            if r.get("missing"):
                print(f"  {name:<8} missing {r['file']}")
                continue
            print(
                f"  {name:<8} duty {_fmt(r['duty_body'])} (asked {_fmt(r['duty_asked'])})  "
                f"light {_fmt(r['light_share_body'])} (asked {_fmt(r['light_share_asked'])})  "
                f"airborne|light {_fmt(r['airborne_when_light'])}  "
                f"graze {_fmt(r['graze_in_swing'])}  "
                f"lift lag {_fmt(r['liftoff_lag_mean'])}  "
                f"land lag {_fmt(r['touchdown_lag_mean'])}  "
                f"clear {_fmt(r['clearance_body'])} (asked {_fmt(r['clearance_asked'])})"
            )
    print(f"written {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
