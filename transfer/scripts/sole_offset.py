"""Where a body's sole point sits when its foot is flat on the ground (research_report.md §2.2).

``z*`` places each foot at its *sole point*. On the G1 that is the underside of the foot's
collision box, measured in the simulator (``g1/scripts/measure_body.py``), so a flat, loaded
foot reads height 0. On the NAO it is the URDF's ``l_sole``/``r_sole`` frame
(``nao_track_env_cfg.sole_points``), which the simulated foot does not rest on. Every plan
puts a stance sole at height 0, so the NAO carries a constant error there: in the tracking
reward, in the clearance it is asked for, and in the kinematic estimator's ground (the
lowest sole corner, which lies at the sole point's height).

Per capture:

* the sole height of firm stance (a foot carrying more than 0.3 of the weight): its 10th
  percentile is the flat foot, higher values a pitched or rolled one;
* the kinematic estimate's trunk height against the truth;
* per planned swing, the clearance shortfall split into the absolute peak's shortfall and
  the stance level's offset (shortfall = asked - reached clearance, each above its own
  stance level).

With ``--urdf``, the NAO's foot collision meshes are read too: their lowest vertex
against the sole frame, in the ankle link.

    python transfer/scripts/sole_offset.py \
        --glob "nao/rollouts/zs_r1_3498_sym_v3_*.npz" \
        --glob "nao/rollouts/zs_r1_3498_sym_f5_*.npz" \
        --glob "nao/rollouts/zs_warm_nao_r1_s3[123]_sym_*_auto_kin_forward_encoder.npz" \
        --glob "nao/rollouts/zs_nao_task_*_sym_*_auto_kin_forward_encoder.npz" \
        --glob "nao/rollouts/oracle_flat_forward.npz" \
        --glob "nao/rollouts/imitate_flat_copy_forward.npz" \
        --glob "g1/rollouts/zs_warm_r3_logload_sym_*.npz" \
        --urdf nao/assets/generated/nao/nao_isaac.urdf --output transfer/results/sole_offset.json
"""

from __future__ import annotations

import argparse
import glob
import json
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(REPO_ROOT / "source" / "humanoid_transfer"))

from humanoid_transfer.common.contact import foot_height  # noqa: E402
from humanoid_transfer.common.skill_state import (  # noqa: E402
    DEFAULT_CONTACT_THRESHOLD,
    SKILL_LAYOUT,
)

FIRM = 0.3
"""Load, in body weights, above which a foot is taken as flat and loaded."""


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True))


def swing_split(z: np.ndarray, due: np.ndarray, valid: np.ndarray) -> dict | None:
    """Median over planned swings of the clearance asked, reached, and the two parts of the gap."""
    fields_b, fields_a = SKILL_LAYOUT.unpack(z), SKILL_LAYOUT.unpack(due)
    hb, ha = foot_height(z), foot_height(due)
    cb = fields_b["foot_load"] > DEFAULT_CONTACT_THRESHOLD
    ca = fields_a["foot_load"] > DEFAULT_CONTACT_THRESHOLD
    rows = []
    steps, envs = valid.shape
    for e in range(envs):
        v = valid[:, e]
        for s in range(2):
            asked_stance, body_stance = ca[:, e, s] & v, cb[:, e, s] & v
            if not asked_stance.any() or not body_stance.any():
                continue
            level_a = float(np.median(ha[:, e, s][asked_stance]))
            level_b = float(np.median(hb[:, e, s][body_stance]))
            for a, b in _runs(~ca[:, e, s] & v):
                if b - a < 2:
                    continue
                window = slice(max(a - 2, 0), min(b + 2, steps))
                rows.append((ha[a:b, e, s].max(), hb[window, e, s].max(), level_a, level_b))
    if not rows:
        return None
    r = np.array(rows)
    asked, reached = r[:, 0] - r[:, 2], r[:, 1] - r[:, 3]
    return {
        "swings": len(r),
        "asked": float(np.median(asked)),
        "reached": float(np.median(reached)),
        "shortfall": float(np.median(asked - reached)),
        "peak_shortfall": float(np.median(r[:, 0] - r[:, 1])),
        "stance_offset": float(np.median(r[:, 3] - r[:, 2])),
    }


def capture_report(path: Path) -> dict:
    data = np.load(path)
    leg = float(data["leg_length"])
    z = data["skill_state"][1:].astype(float)
    done = data["done"].astype(bool)
    valid = ~(done[:-1] | done[1:])
    valid[:20] = False
    fields = SKILL_LAYOUT.unpack(z)
    firm = (fields["foot_load"] > FIRM) & valid[..., None]
    sole = foot_height(z)[firm]
    report = {
        "leg_length": leg,
        "firm_samples": int(firm.sum()),
        "firm_sole_p10": float(np.percentile(sole, 10)),
        "firm_sole_median": float(np.median(sole)),
        "firm_sole_p10_mm": float(np.percentile(sole, 10) * leg * 1000.0),
    }
    if "estimate" in data.files:
        est = SKILL_LAYOUT.unpack(data["estimate"][1:].astype(float))
        report["estimate_minus_truth_height"] = float(
            np.median((est["height"] - fields["height"])[valid])
        )
    if "contract_due" in data.files:
        split = swing_split(z, data["contract_due"][:-1].astype(float), valid)
        if split is not None:
            report["clearance"] = split
    return report


def _stl_vertices(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if raw[:5] == b"solid" and b"facet" in raw[:512]:
        lines = raw.decode(errors="ignore").splitlines()
        return np.array(
            [list(map(float, ln.split()[1:4])) for ln in lines if ln.strip().startswith("vertex")]
        )
    count = struct.unpack("<I", raw[80:84])[0]
    record = np.dtype([("normal", "<3f4"), ("v", "<9f4"), ("attr", "<u2")])
    return np.frombuffer(raw[84 : 84 + count * 50], dtype=record)["v"].reshape(-1, 3).astype(float)


def mesh_report(urdf: Path) -> dict:
    """Each sole frame below its ankle, against the ankle collision mesh's lowest point."""
    root = ET.parse(urdf).getroot()
    out = {}
    for ankle, sole in (("l_ankle", "l_sole"), ("r_ankle", "r_sole")):
        joint = next(j for j in root.iter("joint") if j.find("child").get("link") == sole)
        sole_z = float(joint.find("origin").get("xyz").split()[2])
        link = next(k for k in root.iter("link") if k.get("name") == ankle)
        mesh = link.find("collision/geometry/mesh")
        scale = float(mesh.get("scale", "1 1 1").split()[2])
        low = float(_stl_vertices(urdf.parent / mesh.get("filename"))[:, 2].min() * scale)
        out[ankle] = {
            "sole_frame_z": sole_z,
            "mesh_lowest_z": low,
            "mesh_below_sole_mm": (sole_z - low) * 1000.0,
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--capture", type=Path, action="append", default=[])
    parser.add_argument("--glob", action="append", default=[])
    parser.add_argument("--urdf", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = list(args.capture)
    for pattern in args.glob:
        paths += [Path(p) for p in sorted(glob.glob(pattern))]
    captures = {}
    for path in paths:
        data = np.load(path)
        if "skill_state" not in data.files:
            continue
        captures[path.as_posix()] = rep = capture_report(path)
        line = (
            f"{path.name:58s} firm sole p10 {rep['firm_sole_p10']:+.4f} L "
            f"({rep['firm_sole_p10_mm']:+.2f} mm), median {rep['firm_sole_median']:+.4f}"
        )
        if "estimate_minus_truth_height" in rep:
            line += f", estimated height {rep['estimate_minus_truth_height']:+.4f}"
        if "clearance" in rep:
            c = rep["clearance"]
            line += (
                f"; clearance {c['reached']:.3f}/{c['asked']:.3f} = peak {c['peak_shortfall']:+.4f}"
                f" + stance {c['stance_offset']:+.4f}"
            )
        print(line)
    report = {"firm_load": FIRM, "captures": captures}
    by_body: dict[str, list[float]] = {}
    for path, rep in captures.items():
        by_body.setdefault(Path(path).parts[0], []).append(rep["firm_sole_p10"])
    report["flat_foot"] = {
        body: {
            "captures": len(v),
            "p10_median": float(np.median(v)),
            "p10_range": [float(min(v)), float(max(v))],
        }
        for body, v in by_body.items()
    }
    for body, s in report["flat_foot"].items():
        low, high = s["p10_range"]
        print(
            f"{body}: flat-foot sole height {s['p10_median']:+.5f} L "
            f"(range {low:+.5f} to {high:+.5f}, {s['captures']} captures)"
        )
    if args.urdf is not None:
        report["mesh"] = mesh_report(args.urdf)
        for ankle, m in report["mesh"].items():
            print(f"{ankle}: collision mesh {m['mesh_below_sole_mm']:.2f} mm below the sole frame")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"written {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
