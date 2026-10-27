"""The paper's figures, drawn from the same result files as its tables.

Written to ``transfer/paper/figures/`` (gitignored: rerun this to rebuild them) as PDF
and PNG. A figure whose results do not exist yet is skipped, as a table row is.

* ``signature``: the adopted model's gait against the teacher's, per body and skill:
  each feature's relative deviation at each L4 command, with gate C's 10 % line.
* ``decomposition``: the deviation split into the generator's, the tracker's and the
  estimate's parts (``paper_decomposition.py``).
* ``seeds``: the log-space loads against linear, three seeds each, probes passed.
* ``baselines``: falls, speed and the gait's largest deviation of the zero-shot model
  against B1, B2 and B3, per skill.

Usage::

    python transfer/scripts/paper_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_tables import (  # noqa: E402
    COMMANDS,
    GENERATOR,
    MODEL,
    PAPER,
    nao_methods,
    read,
)

OUT = PAPER / "figures"
# The reference palette's first three categorical slots (validated all-pairs, light), a
# neutral for totals, and the text inks; hatching repeats identity for print.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
NEUTRAL = "#8a8985"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
HATCH = ("", "//", "..")
MARKERS = ("o", "s", "^")
FEATURES = {
    "stride_length": "stride\nlength",
    "stride_period": "stride\nperiod",
    "duty_factor_0": "duty\nleft",
    "duty_factor_1": "duty\nright",
}
DECOMPOSED = {
    "speed": "speed",
    "stride_length": "stride\nlength",
    "stride_period": "stride\nperiod",
    "duty_left": "duty\nleft",
    "duty_right": "duty\nright",
}
"""``paper_decomposition.py``'s features."""
TERMS = {"estimator": "deployed − true path"}
TOLERANCE = 0.10

plt.rcParams.update({
    "font.size": 8, "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2,
    "ytick.color": INK_2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "pdf.fonttype": 42,
})  # fmt: skip


def save(figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        figure.savefig(OUT / f"{name}.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"written {OUT / name}.pdf")


def panels(count: int):
    """Two panels to a row (bodies by row, skills by column), one y scale for all."""
    rows = (count + 1) // 2
    figure, axes = plt.subplots(rows, 2, figsize=(7.2, 2.2 * rows), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[count:]:
        ax.set_visible(False)
    for ax in axes[::2]:
        ax.set_ylabel("")
    figure.subplots_adjust(hspace=0.45, wspace=0.12)
    return figure, axes[:count]


def legend_above(figure, axes, title: str | None = None) -> None:
    """One legend row above the panels, clear of every bar."""
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, title=title, loc="lower center", ncol=len(labels), fontsize=7,
        bbox_to_anchor=(0.5, 1.0), title_fontsize=7,
    )  # fmt: skip


def bars(ax, groups: list[str], series: dict[str, list[float]], colors, hatches) -> None:
    """Grouped bars with a 2-point gap between neighbours."""
    width = 0.8 / len(series)
    x = np.arange(len(groups))
    for k, (label, values) in enumerate(series.items()):
        ax.bar(
            x + (k - (len(series) - 1) / 2) * width, values, width * 0.92, label=label,
            color=colors[k], hatch=hatches[k], edgecolor="white", linewidth=0.6,
        )  # fmt: skip
    ax.set_xticks(x, groups)


def signature() -> None:
    cases = [(body, skill) for body in MODEL for skill in GENERATOR]
    reports = {}
    for body, skill in cases:
        pattern = f"zero_shot_{body}_{MODEL[body]}_{GENERATOR[skill]}_auto_kin_l4_{{}}.json"
        runs = [read(pattern.format(c)) for c in COMMANDS]
        if all(r is not None for r in runs):
            reports[(body, skill)] = [r["runs"]["encoder"]["verdict"] for r in runs]
    if not reports:
        return
    figure, axes = panels(len(reports))
    for ax, ((body, skill), verdicts) in zip(np.atleast_1d(axes), reports.items(), strict=True):
        series = {
            c: [v["signature_deviation"][f] for f in FEATURES]
            for c, v in zip(COMMANDS, verdicts, strict=True)
        }
        bars(ax, list(FEATURES.values()), series, SERIES, HATCH)
        ax.axhline(TOLERANCE, color=INK_2, linewidth=0.8, linestyle="--")
        ax.set_title(f"{body.upper()}, {skill}", color=INK, fontsize=8)
    for ax in axes[::2]:
        ax.set_ylabel("|deviation| / teacher")
    np.atleast_1d(axes)[0].annotate(
        "gate C, 10 %", (-0.45, TOLERANCE), textcoords="offset points",
        xytext=(0, 2), ha="left", va="bottom", color=INK_2, fontsize=7,
    )  # fmt: skip
    legend_above(figure, axes, "command")
    save(figure, "signature")


def decomposition() -> None:
    path = PAPER / "data" / "decomposition.json"
    if not path.exists():
        return
    report = json.loads(path.read_text(encoding="utf-8"))
    summary = report["summary"]
    names = [DECOMPOSED[n] for n in report["features"]]
    figure, axes = panels(len(summary))
    for ax, (key, terms) in zip(np.atleast_1d(axes), summary.items(), strict=True):
        # The last term is the deployed path against the true-state one, not the estimator's
        # error: on flat the true-state path is the worse of the two.
        series = {TERMS.get(t, t): [terms[t][f] for f in report["features"]] for t in terms}
        bars(ax, names, series, (NEUTRAL, *SERIES), ("", *HATCH))
        body, skill = key.split("/")
        ax.set_title(f"{body.upper()}, {skill}", color=INK, fontsize=8)
    for ax in axes[::2]:
        ax.set_ylabel("mean |deviation| / teacher")
    legend_above(figure, axes)
    save(figure, "decomposition")


def seeds() -> None:
    report = read("compare_seeds_logload_grid_and_corners.json")
    if report is None:
        return
    per_seed = report["passed_per_seed"]
    figure, ax = plt.subplots(figsize=(2.6, 2.2))
    for k, arm in enumerate(("reference", "candidate")):
        values = np.asarray(per_seed[arm], dtype=float)
        ax.scatter(np.full(len(values), k), values, s=28, color=SERIES[k], zorder=3,
                   edgecolor="white", linewidth=1.5)  # fmt: skip
        # The mean beside the seeds, not over them.
        ax.hlines(values.mean(), k + 0.08, k + 0.28, color=INK, linewidth=1.2)
        ax.annotate(f"mean {values.mean():.1f}", (k + 0.3, values.mean()), va="center",
                    color=INK, fontsize=7)  # fmt: skip
    ax.set_xticks([0, 1], ["linear", "log space"])
    ax.set_xlim(-0.5, 1.6)
    ax.set_ylabel("probes passed (of 97)")
    ax.set_title(
        f"{report['passed_difference']:+.2f} ± {report['passed_se']:.2f} (bootstrap SE)",
        color=INK_2, fontsize=7,
    )  # fmt: skip
    save(figure, "seeds")


def baselines() -> None:
    by_skill = {skill: nao_methods(skill) for skill in GENERATOR}
    names = [n for n in by_skill["flat"] if any(by_skill[s][n] is not None for s in GENERATOR)]
    if len(names) < 2:
        return
    figure, (falls, speed, gait) = plt.subplots(1, 3, figsize=(7.2, 2.4))
    handles = []
    for k, skill in enumerate(GENERATOR):
        results = [by_skill[skill][n] for n in names]
        rate = [np.nan if r is None else r["falls"] / r["episodes"] for r in results]
        ratio = [np.nan if r is None else np.mean(r["speed_ratio"]) for r in results]
        # Gate C's reading: each command's largest feature deviation, then their mean. A
        # command without a stride has no stride features: its largest is among the rest,
        # and the marker is left open.
        worst = [
            np.nan if r is None else np.mean(np.nanmax(list(r["deviation"].values()), axis=0))
            for r in results
        ]
        strideless = [
            r is not None and not np.isfinite(list(r["deviation"].values())).all() for r in results
        ]
        x = np.arange(len(names)) + (k - 0.5) * 0.38
        for ax, values in ((falls, rate), (speed, ratio)):
            bar = ax.bar(x, values, 0.35, color=SERIES[k], hatch=HATCH[k], edgecolor="white",
                         linewidth=0.6)  # fmt: skip
        # A log axis has no baseline for a bar to start from: markers, shaped per skill.
        (dot,) = gait.plot(x, worst, MARKERS[k], color=SERIES[k], markersize=6,
                           markeredgecolor="white", linestyle="none")  # fmt: skip
        for xi, yi, open_ in zip(x, worst, strideless, strict=True):
            if open_:
                gait.plot(xi, yi, MARKERS[k], markerfacecolor="white", markeredgecolor=SERIES[k],
                          markersize=6, markeredgewidth=1.2)  # fmt: skip
                gait.annotate("no stride on\na command", (xi, yi), textcoords="offset points",
                              xytext=(5, 0), va="center", color=INK_2, fontsize=6)  # fmt: skip
        handles.append((bar, dot))
    falls.set_ylabel("falls / episode (NAO, L4)")
    falls.set_ylim(bottom=0, top=max(0.05, falls.get_ylim()[1]))
    speed.set_ylabel("speed / teacher's")
    speed.axhline(0.9, color=INK_2, linewidth=0.8, linestyle="--")
    gait.set_ylabel("largest gait |deviation| / teacher")
    gait.set_yscale("log")
    gait.axhline(TOLERANCE, color=INK_2, linewidth=0.8, linestyle="--")
    gait.set_xlim(-0.6, len(names) - 0.4)
    for ax in (falls, speed, gait):
        ax.set_xticks(np.arange(len(names)), names, rotation=35, ha="right")
    figure.legend(
        handles, list(GENERATOR), title="skill", loc="lower center", ncol=len(handles),
        fontsize=7, title_fontsize=7, bbox_to_anchor=(0.5, 1.0),
        handler_map={tuple: HandlerTuple(ndivide=None)},
    )  # fmt: skip
    figure.subplots_adjust(wspace=0.45)
    save(figure, "baselines")


def main() -> int:
    signature()
    decomposition()
    seeds()
    baselines()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
