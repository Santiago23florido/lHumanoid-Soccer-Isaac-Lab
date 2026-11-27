"""Build and update the traceable Local + Mesogip workload dispatch ledger.

Produces ``transfer/results/cluster_dispatch_ledger.json`` combining:
  - the single-source-of-truth job manifest (``transfer/scripts/icml_runs.py``),
  - verified local executions (RTX laptop),
  - verified Mesogip executions (``logs/icml/mesogip_provenance/*.json``),
  - cluster hardware/node probe records, and
  - aggregate per-group and per-site compute summaries suitable for a software/engineering report.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "transfer" / "scripts"))
import icml_runs  # noqa: E402

PROV_DIR = REPO / "logs" / "icml" / "mesogip_provenance"
LEDGER_PATH = REPO / "transfer" / "results" / "cluster_dispatch_ledger.json"


def git_info() -> dict[str, str]:
    def run(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()

    return {
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": run("rev-parse", "HEAD"),
    }


def load_mesogip_provenance() -> tuple[dict[str, dict], dict | None]:
    records: dict[str, dict] = {}
    node_probe: dict | None = None
    if not PROV_DIR.exists():
        return records, node_probe
    for p in sorted(PROV_DIR.glob("*.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        if p.name == "node_probe.json":
            node_probe = data
        elif "job_id" in data:
            records[data["job_id"]] = data
    return records, node_probe


def assign_target_site(job: icml_runs.Job) -> tuple[str, str]:
    """Deterministic placement policy for every manifest job."""
    if job.kind == "isaac":
        return "local-rtx", "Isaac Sim 5.1 closed-loop (or Mesogip ENSTA-l40s if Apptainer validated)"
    return "mesogip-h100", "Pure PyTorch offline job on Mesogip ENSTA-h100 (<=2 GPUs concurrent)"


def build_ledger() -> dict:
    jobs = icml_runs.manifest()
    ready_ids = {j.id for j in icml_runs.runnable(jobs)}
    prov_map, node_probe = load_mesogip_provenance()

    entries = []
    by_group: dict[str, Counter] = {}
    by_site: Counter = Counter()
    mesogip_wall_s = 0.0

    for j in jobs:
        done = j.done()
        prov = prov_map.get(j.id)
        planned_site, rationale = assign_target_site(j)
        if prov and prov.get("status") == "done":
            actual_site = f"mesogip/{prov.get('partition', 'ENSTA-h100')}"
            status = "done"
            mesogip_wall_s += float(prov.get("wall_seconds", 0.0))
        elif done:
            actual_site = "local/windows-rtx"
            status = "done"
        elif prov and prov.get("status") == "failed":
            actual_site = f"mesogip/{prov.get('partition', 'ENSTA-h100')}"
            status = "failed"
        elif j.id in ready_ids:
            actual_site = None
            status = "runnable"
        else:
            actual_site = None
            status = "blocked"

        g_cnt = by_group.setdefault(j.group, Counter())
        g_cnt[status] += 1
        g_cnt["total"] += 1
        if actual_site:
            by_site[actual_site] += 1

        entry = {
            "job_id": j.id,
            "group": j.group,
            "kind": j.kind,
            "planned_site": planned_site,
            "actual_site": actual_site,
            "status": status,
            "laptop_est_hours": j.hours,
            "outputs": j.outputs,
        }
        if prov:
            entry["mesogip"] = {
                "host": prov.get("host"),
                "partition": prov.get("partition"),
                "slurm_job_id": prov.get("slurm_job_id"),
                "gpu_model": prov.get("gpu_model"),
                "git_commit": prov.get("git_commit"),
                "wall_seconds": prov.get("wall_seconds"),
                "started_utc": prov.get("started_utc"),
                "ended_utc": prov.get("ended_utc"),
            }
        entries.append(entry)

    ledger = {
        "schema_version": "1.0",
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": git_info(),
        "policy": {
            "local_site": "Windows workstation (RTX 8 GB) — Isaac Sim 5.1 closed-loop jobs (main, h2)",
            "cluster_site": "Mesogip ENSTA (mesogip.r2.enst.fr) — ENSTA-h100 for pure PyTorch offline waves (cross, r4, loo, scaling, adapt)",
            "cluster_guards": {
                "max_walltime": "03:50:00",
                "signal": "B:USR1@120",
                "qos_mode": "packed single-job sequential wave (respects QOSMaxJobsPerUserLimit=1)",
                "login_node_compute": False,
                "nfs_data_bundle_gb": 6.3,
                "nfs_symlink_dedup_saved_gb": 2.1,
                "node_local_tmp_cache": True,
                "venv_path": "~/venvs/pact-icml",
                "coexistence_note": "Leaves ENSTA-l40s uncontested for active Musculoskeletal-Hand-Simulation jobs",
            },
        },
        "mesogip_node_probe": node_probe,
        "summary": {
            "total_jobs": len(entries),
            "done_total": sum(1 for e in entries if e["status"] == "done"),
            "runnable_total": sum(1 for e in entries if e["status"] == "runnable"),
            "blocked_total": sum(1 for e in entries if e["status"] == "blocked"),
            "failed_total": sum(1 for e in entries if e["status"] == "failed"),
            "completed_by_site": dict(by_site),
            "mesogip_total_gpu_hours": round(mesogip_wall_s / 3600.0, 3),
            "by_group": {g: dict(c) for g, c in by_group.items()},
        },
        "jobs": entries,
    }
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    LEDGER_PATH.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    print(
        f"Updated {LEDGER_PATH}: {ledger['summary']['done_total']}/{ledger['summary']['total_jobs']} done "
        f"| sites={ledger['summary']['completed_by_site']} | mesogip_gpu_h={ledger['summary']['mesogip_total_gpu_hours']}"
    )
    return ledger


if __name__ == "__main__":
    build_ledger()
