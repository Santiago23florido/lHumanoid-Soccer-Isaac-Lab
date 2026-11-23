"""The ICML paper's figures, rebuilt from ``transfer/results`` into PDFs beside the tables.

Each figure reads the result files the manifest's jobs write (``icml_runs.py``). A panel whose
files do not exist yet says "pending", so the paper compiles from the first day.

* ``fig_scaling.pdf`` (H4): the held-out body's offline R² (commanded torque) against the number
  of training bodies, PACT, one panel per registered held-out body (NAO, H1). Dots are subsets
  and seeds; the line is their mean ± SE.
* ``fig_adapt.pdf`` (G-N3): adding a body offline at k of its probes, one panel per body. The
  three arms are the registered comparison's: PACT ``bitfit``, HPT ``stem``, and URMA
  ``bitfit``. Zero-shot (k = 0) is shown where the arm has it. Mean ± SE over seeds.

Colors are the first three categorical slots, validated all-pairs for light surfaces
(dataviz ``validate_palette.js``). Each arm also has its own marker shape and a legend entry,
and its numbers are in the tables, so identity never rests on color alone.

    python transfer/scripts/icml_figures.py [--out docs/icml/generated]
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "transfer" / "scripts"))

from body_conditions import TEACHERS, registry  # noqa: E402
from icml_runs import SCALING_HELD_OUT, SEEDS  # noqa: E402

RESULTS = REPO / "transfer/results"
SERIES = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3de"
BODY = {"g1": "G1", "nao": "NAO", "h1": "H1", "cassie": "Cassie", "digit": "Digit"}
ADAPT_ARMS = (
    ("pact_d", "bitfit", "PACT bitfit", SERIES["blue"], "o"),
    ("hpt_d", "stem", "HPT stem", SERIES["orange"], "s"),
    ("urma_d", "bitfit", "URMA bitfit", SERIES["aqua"], "^"),
)
BUDGETS = (0, 1, 4, 16, "all")  # 0: zero-shot


def style() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6,
        "ytick.labelsize": 6, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.5, "lines.markersize": 4, "pdf.fonttype": 42,
    })  # fmt: skip


def r2(name: str, body: str, arm: str) -> float | None:
    path = RESULTS / name
    if not path.exists():
        return None
    entry = json.loads(path.read_text("utf-8"))["bodies"].get(body, {}).get(arm)
    return None if entry is None else float(entry["r2"])


def mean_se(values: list[float]) -> tuple[float, float] | None:
    if not values:
        return None
    se = float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    return float(np.mean(values)), se


def pending(ax, title: str) -> None:
    ax.set_title(title)
    ax.text(0.5, 0.5, "pending", ha="center", va="center", color=MUTED, transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def fig_scaling(out: Path) -> int:
    bodies = list(registry(TEACHERS))
    fig, axes = plt.subplots(1, len(SCALING_HELD_OUT), figsize=(3.25, 1.6), sharey=True)
    filled = 0
    for ax, held in zip(np.atleast_1d(axes), SCALING_HELD_OUT, strict=True):
        others = [b for b in bodies if b != held]
        points: dict[int, list[float]] = {}
        for size in range(2, len(others) + 1):
            for subset in itertools.combinations(others, size):
                for seed in SEEDS:
                    if size == len(others):
                        name = f"icml_loo_pact_d_{held}_s{seed}.json"
                    else:
                        name = f"icml_scaling_pact_d_{held}_{'+'.join(subset)}_s{seed}.json"
                    value = r2(name, held, "pact_d")
                    if value is not None:
                        points.setdefault(size, []).append(value)
        title = f"{BODY[held]} held out"
        if not points:
            pending(ax, title)
            continue
        filled += 1
        sizes = sorted(points)
        for size in sizes:
            jitter = np.linspace(-0.08, 0.08, len(points[size]))
            ax.scatter(size + jitter, points[size], s=6, color=GRID, edgecolors=MUTED, linewidths=0.3, zorder=2)
        stats = [mean_se(points[s]) for s in sizes]
        ax.errorbar(
            sizes, [m for m, _ in stats], yerr=[e for _, e in stats], color=SERIES["blue"],
            marker="o", capsize=1.5, zorder=3,
        )  # fmt: skip
        ax.set_title(title)
        ax.set_xticks(range(2, len(others) + 1))
        ax.set_xlabel("training bodies")
    np.atleast_1d(axes)[0].set_ylabel("held-out $R^2$ (torque)")
    fig.tight_layout()
    fig.savefig(out / "fig_scaling.pdf")
    plt.close(fig)
    return filled


def fig_adapt(out: Path) -> int:
    bodies = list(registry(TEACHERS))
    fig, axes = plt.subplots(1, len(bodies), figsize=(6.75, 1.6), sharey=True)
    filled = 0
    for ax, body in zip(np.atleast_1d(axes), bodies, strict=True):
        drew = False
        for arm, mode, label, color, marker in ADAPT_ARMS:
            xs, ys, es = [], [], []
            for x, k in enumerate(BUDGETS):
                if k == 0:
                    if mode == "stem":
                        continue  # HPT has no stem for an unseen body
                    names = [f"icml_loo_{arm}_{body}_s{s}.json" for s in SEEDS]
                else:
                    names = [f"icml_adapt_{arm}_{body}_{mode}_k{k}_s{s}.json" for s in SEEDS]
                got = [v for v in (r2(n, body, arm) for n in names) if v is not None]
                stat = mean_se(got)
                if stat is not None:
                    xs.append(x)
                    ys.append(stat[0])
                    es.append(stat[1])
            if xs:
                drew = True
                ax.errorbar(xs, ys, yerr=es, color=color, marker=marker, capsize=1.5, label=label)
        if not drew:
            pending(ax, f"{BODY[body]} added")
            continue
        filled += 1
        ax.set_title(f"{BODY[body]} added")
        ax.set_xticks(range(len(BUDGETS)))
        ax.set_xticklabels(["0", "1", "4", "16", "all"])
        ax.set_xlabel("probes of the new body")
    np.atleast_1d(axes)[0].set_ylabel("$R^2$ (torque)")
    handles = [plt.Line2D([], [], color=c, marker=m, label=lab) for _, _, lab, c, m in ADAPT_ARMS]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    fig.savefig(out / "fig_adapt.pdf", bbox_inches="tight")
    plt.close(fig)
    return filled


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=REPO / "docs/icml/generated")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    style()
    scaling = fig_scaling(args.out)
    adapt = fig_adapt(args.out)
    print(f"written {args.out}: fig_scaling ({scaling} panels with data), fig_adapt ({adapt})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
