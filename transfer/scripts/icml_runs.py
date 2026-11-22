"""Every run the ICML paper's tables and figures need, as one manifest: status, order, job files.

Each job has a kind (``isaac``: needs Isaac Sim, one process per GPU; ``offline``: PyTorch
only, any CUDA GPU or the CPU), the files that mark it done, the jobs it needs, the command,
and its cost on the development laptop (RTX 8 GB), measured from earlier runs of the same
script (``hours``). A job is runnable when every job it needs is done.

    python transfer/scripts/icml_runs.py status [--group main]   # done / runnable / blocked
    python transfer/scripts/icml_runs.py next --kind isaac         # runnable commands, in order
    python transfer/scripts/icml_runs.py slurm --kind offline --out logs/icml/slurm \\
        [--partition gpu] [--hours 4]                              # one sbatch array per group

Groups (the paper's tables and figures):
* ``data``: the final teachers' co-executions (replay chains); the map stage links each
  teacher's audit probe capture.
* ``main``: every shared arm on three seeds through ``icml_recipe.sh``. It gives one policy for
  five bodies (G-D1, Table 1) and the held-out task on every body (Table 2).
* ``h2``: plans projected onto still loaded feet, closed loop, against the plans as drawn.
* ``loo``: each body held out of the shared decoder in turn (offline R^2, Fig. held-out body).
* ``scaling``: H4, the held-out body's R^2 against the number of training bodies.
* ``adapt``: G-N3, adding a body offline at k of its probes, every arm.
* ``r4``: the registered unseen-body criterion (encoder and translator gap closure).
* ``cross``: G-X and G-A on the final co-executions (translator, encoder, in distribution).
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "transfer" / "scripts"))

from body_conditions import TEACHERS, registry  # noqa: E402

ARMS = ("pact_d", "pact_na", "pact_nb", "urma_d", "hpt_d")
"""PACT, without the anchor, without E_b, URMA (end-to-end), HPT (per-body stems)."""
SEEDS = (0, 1, 2)
HELD_OUT_ARMS = ("pact_d", "pact_na", "pact_nb", "urma_d", "hpt_d")
"""Every arm is trained without each body; HPT's has no stem for it (G-N3's ``stem`` adds one)."""
SCALING_HELD_OUT = ("nao", "h1")
"""H4's held-out bodies, as registered (the NAO first, the H1 second)."""
ADAPT = {"pact_d": ("zero", "etune", "bitfit", "full"), "urma_d": ("zero", "bitfit", "full"), "hpt_d": ("stem", "full")}
BUDGETS = (1, 4, 16, 0)  # 0: every training probe
BASE_MAP = "logs/pact/teachers/{body}_{variant}_probes.npz"
BASE_TRAIN = "logs/pact/teachers/{body}_{variant}_train.npz"
COEXEC = "logs/pact/final/coexec"
PY = '"${PYTHON:-python}"'  # the environment's interpreter (PYTHON on this laptop)


@dataclass
class Job:
    id: str
    group: str
    kind: str
    command: str
    outputs: list[str]
    hours: float
    needs: list[str] = field(default_factory=list)

    def done(self) -> bool:
        return all((REPO / o).exists() for o in self.outputs)


def bodies() -> list[str]:
    return list(registry(TEACHERS))


def manifest() -> list[Job]:
    jobs: list[Job] = []
    all_bodies = bodies()
    csv = ",".join(all_bodies)
    jobs.append(Job(
        "data/replay", "data", "isaac",
        f"PACT_COEXEC_DIR={COEXEC} bash transfer/scripts/pact_coexecute.sh",
        [f"{COEXEC}/{b}_replay.npz" for b in all_bodies], 0.4,
    ))  # fmt: skip
    for arm, seed in itertools.product(ARMS, SEEDS):
        name = f"{arm}_s{seed}"
        jobs.append(Job(
            f"main/{name}", "main", "isaac", f"bash transfer/scripts/icml_recipe.sh {arm} {seed}",
            [f"transfer/results/icml_final_{name}_d1.json", f"transfer/results/icml_held_out_icml_{name}.json"],
            7.0,
        ))  # fmt: skip
    for seed in SEEDS:
        name = f"pact_d_s{seed}"
        jobs.append(Job(
            f"h2/{name}", "h2", "isaac",
            f"PACT_HELD_OUT_H2=1 bash transfer/scripts/pact_held_out.sh logs/icml/models/{name}.pt icml_{name}"
            f" && {PY} transfer/scripts/icml_held_out_read.py --h2 --tag icml_{name}",
            [f"transfer/results/icml_h2_icml_{name}.json"], 2.5, [f"main/{name}"],
        ))  # fmt: skip
    for arm, body, seed in itertools.product(HELD_OUT_ARMS, all_bodies, SEEDS):
        name = f"loo_{arm}_{body}_s{seed}"
        jobs.append(Job(
            f"loo/{arm}_{body}_s{seed}", "loo", "offline",
            f"{PY} transfer/scripts/pact_gate_d.py --bodies {csv} --hold-out {body} --captures '{BASE_MAP}'"
            f" --dagger '{BASE_TRAIN}' --min-duty 0.55 --steps 24000 --arms hold,{arm} --seed {seed}"
            f" --save logs/icml/models/{name}.pt --output transfer/results/icml_{name}.json",
            [f"transfer/results/icml_{name}.json", f"logs/icml/models/{name}.pt"], 1.0,
        ))  # fmt: skip
    for held, seed in itertools.product(SCALING_HELD_OUT, SEEDS):
        others = [b for b in all_bodies if b != held]
        for size in range(2, len(others)):  # the full set is loo/
            for subset in itertools.combinations(others, size):
                key = "+".join(subset)
                name = f"scaling_pact_d_{held}_{key}_s{seed}"
                jobs.append(Job(
                    f"scaling/{held}_{key}_s{seed}", "scaling", "offline",
                    f"{PY} transfer/scripts/pact_gate_d.py --bodies {','.join((*subset, held))} --hold-out {held}"
                    f" --captures '{BASE_MAP}' --dagger '{BASE_TRAIN}' --min-duty 0.55 --steps 24000"
                    f" --arms hold,pact_d --seed {seed} --save logs/icml/models/{name}.pt"
                    f" --output transfer/results/icml_{name}.json",
                    [f"transfer/results/icml_{name}.json"], 0.8,
                ))  # fmt: skip
    for arm, body, seed in itertools.product(ADAPT, all_bodies, SEEDS):
        base = f"loo_{arm}_{body}_s{seed}"
        for mode, k in itertools.product(ADAPT[arm], BUDGETS):
            if mode == "zero" and k:
                continue  # zero-shot reads no probe
            name = f"adapt_{arm}_{body}_{mode}_k{k or 'all'}_s{seed}"
            jobs.append(Job(
                f"adapt/{arm}_{body}_{mode}_k{k or 'all'}_s{seed}", "adapt", "offline",
                f"{PY} transfer/scripts/pact_gate_d.py --bodies {csv} --hold-out {body} --captures '{BASE_MAP}'"
                f" --dagger '{BASE_TRAIN}' --min-duty 0.55 --init logs/icml/models/{base}.pt --adapt {mode}"
                f" --probes {k} --steps 2000 --arms hold,{arm} --seed {seed}"
                f" --save logs/icml/models/{name}.pt --output transfer/results/icml_{name}.json",
                [f"transfer/results/icml_{name}.json"], 0.1, [f"loo/{arm}_{body}_s{seed}"],
            ))  # fmt: skip
    for part, body, seed in itertools.product(("encoder", "translator"), all_bodies, SEEDS):
        name = f"r4_{part}_{body}_s{seed}"
        jobs.append(Job(
            f"r4/{part}_{body}_s{seed}", "r4", "offline",
            f"{PY} transfer/scripts/pact_gate_n1.py --part {part} --bodies {csv} --hold-out {body}"
            f" --budgets 1 --arms zero --seed {seed} --encoder-captures '{COEXEC}/{{body}}_map.npz'"
            f" --translator-captures '{COEXEC}/{{body}}_replay.npz' --output transfer/results/icml_{name}.json",
            [f"transfer/results/icml_{name}.json"], 0.7, ["data/replay"],
        ))  # fmt: skip
    for seed in SEEDS:
        jobs.append(Job(
            f"cross/x_s{seed}", "cross", "offline",
            f"{PY} transfer/scripts/pact_gate_x.py --bodies {csv} --hold-out '' --seed {seed}"
            f" --captures '{COEXEC}/{{body}}_replay.npz' --output transfer/results/icml_final_gate_x_s{seed}.json",
            [f"transfer/results/icml_final_gate_x_s{seed}.json"], 0.7, ["data/replay"],
        ))  # fmt: skip
        jobs.append(Job(
            f"cross/a_s{seed}", "cross", "offline",
            f"{PY} transfer/scripts/pact_gate_a3.py --bodies {csv} --hold-out '' --seed {seed}"
            f" --captures '{COEXEC}/{{body}}_map.npz' --output transfer/results/icml_final_gate_a3_s{seed}.json",
            [f"transfer/results/icml_final_gate_a3_s{seed}.json"], 0.7, ["data/replay"],
        ))  # fmt: skip
    return jobs


def runnable(jobs: list[Job]) -> list[Job]:
    done = {j.id for j in jobs if j.done()}
    return [j for j in jobs if j.id not in done and all(n in done for n in j.needs)]


def status(jobs: list[Job], group: str | None) -> None:
    ready = {j.id for j in runnable(jobs)}
    groups = sorted({j.group for j in jobs}, key=[j.group for j in jobs].index)
    for g in groups:
        if group and g != group:
            continue
        members = [j for j in jobs if j.group == g]
        done = [j for j in members if j.done()]
        todo = [j for j in members if not j.done()]
        hours = {k: sum(j.hours for j in todo if j.kind == k) for k in ("isaac", "offline")}
        print(
            f"{g:8s} {len(done):4d}/{len(members):<4d} done, {sum(j.id in ready for j in todo):4d} runnable;"
            f" left: isaac {hours['isaac']:6.1f} h, offline {hours['offline']:6.1f} h (laptop hours)"
        )
    todo = [j for j in jobs if not j.done()]
    print(
        f"total    left: isaac {sum(j.hours for j in todo if j.kind == 'isaac'):.1f} h, "
        f"offline {sum(j.hours for j in todo if j.kind == 'offline'):.1f} h"
    )


SBATCH = """#!/usr/bin/env bash
#SBATCH --job-name=icml_{group}
#SBATCH --array=0-{last}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time={hours}:00:00
#SBATCH --output={out}/{group}_%a.log
{partition}
# Generated by transfer/scripts/icml_runs.py from the manifest; one job per array index.
set -euo pipefail
cd "${{ICML_REPO:?set ICML_REPO to the repository checkout}}"
mapfile -t commands < {out}/{group}.commands
eval "${{commands[$SLURM_ARRAY_TASK_ID]}}"
"""


def slurm(jobs: list[Job], kind: str, out: Path, partition: str, hours: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    ready = [j for j in runnable(jobs) if j.kind == kind]
    for group in sorted({j.group for j in ready}):
        members = [j for j in ready if j.group == group]
        (out / f"{group}.commands").write_text("\n".join(j.command for j in members) + "\n", encoding="utf-8")
        script = SBATCH.format(
            group=group, last=len(members) - 1, hours=hours, out=out.as_posix(),
            partition=f"#SBATCH --partition={partition}" if partition else "",
        )  # fmt: skip
        (out / f"{group}.sbatch").write_text(script, encoding="utf-8", newline="\n")
        print(f"{out / (group + '.sbatch')}: {len(members)} jobs")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="action", required=True)
    s = sub.add_parser("status")
    s.add_argument("--group", default=None)
    n = sub.add_parser("next")
    n.add_argument("--kind", choices=("isaac", "offline"), default=None)
    n.add_argument("--group", default=None)
    b = sub.add_parser("slurm")
    b.add_argument("--kind", choices=("isaac", "offline"), required=True)
    b.add_argument("--out", type=Path, default=REPO / "logs/icml/slurm")
    b.add_argument("--partition", default="")
    b.add_argument("--hours", type=int, default=4)
    sub.add_parser("json")
    args = parser.parse_args()
    jobs = manifest()
    if args.action == "status":
        status(jobs, args.group)
    elif args.action == "next":
        for j in runnable(jobs):
            if (args.kind is None or j.kind == args.kind) and (args.group is None or j.group == args.group):
                print(j.command)
    elif args.action == "slurm":
        slurm(jobs, args.kind, args.out, args.partition, args.hours)
    else:
        print(json.dumps([j.__dict__ | {"done": j.done()} for j in jobs], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
