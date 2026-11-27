# Hybrid Local + Mesogip Workload Distribution & Traceability Plan

This document specifies how the ICML 2027 PACT experiment manifest (`transfer/scripts/icml_runs.py`) is split between the local workstation and the **Mesogip** HPC cluster (`mesogip.r2.enst.fr`, ENSTA account `ensta-etudiant`), while preserving complete per-job traceability for software and reproducibility reporting.

## 1. Site Responsibilities & Coexistence

| Execution Site | Hardware | Assigned Job Groups | Rationale |
| --- | --- | --- | --- |
| **Local Workstation** (Windows) | 1 × RTX 8 GB + Isaac Sim 5.1 | `data`, `main/*`, `h2/*` (`isaac` jobs) | Native Isaac Sim 5.1 + local NAO CC BY-NC-ND asset (never redistributed or pushed). |
| **Mesogip `ENSTA-h100`** | 4 × NVIDIA H100 80 GB (`ensta-h10001`) | `cross/*`, `r4/*`, `loo/*`, `scaling/*`, `adapt/*` (`offline` jobs) | Pure PyTorch/NumPy jobs (`tests/test_offline_imports.py`); leaves `ENSTA-l40s` completely free for active `Musculoskeletal-Hand-Simulation` jobs. |
| **Mesogip `ENSTA-l40s`** | 8 × NVIDIA L40S 48 GB (`ensta-l40s01..02`) | Reserved / optional Stage 2 | Uncontested while `Musculoskeletal-Hand-Simulation` runs; RT-core capable if Isaac Lab Apptainer is later enabled. |

## 2. Mesogip Compliance Requirements (Aligned with `Musculoskeletal-Hand-Simulation`)

Every script under `scripts/mesogip/` enforces the cluster operational requirements:

1. **Zero compute on the login node:** all tests, probes, and trainings execute strictly inside `sbatch` jobs.
2. **Short resumable walltimes:** every job uses `#SBATCH --time <= 03:50:00` with `#SBATCH --signal=B:USR1@120` and skips tasks whose registered outputs already exist.
3. **QOS-aware packed execution (`QOSMaxJobsPerUserLimit`):** Mesogip's `ensta-etudiant` QOS enforces a strict concurrent job limit per user. `04_submit_offline_wave.sh` defaults to **packed mode**, running short offline tasks sequentially inside a single resumable `03:50:00` H100 job (with a 15-minute soft deadline guard before walltime), avoiding `MaxSubmit` waste and respecting active `Musculoskeletal-Hand-Simulation` jobs on `ENSTA-l40s`.
4. **Minimal NFS `$HOME` footprint (storage-safe):**
   - Checkout at `~/proyectos/lHumanoid-Soccer-Isaac-Lab` (branch `feat/icml-token-latent`).
   - Isolated virtual environment at `~/venvs/pact-icml` (`pip install --no-cache-dir`, `~/.cache/pip` purged immediately after install).
   - Data sync (`scripts/mesogip/02_sync_data.ps1`) transfers **only** the 17 unique frozen files (**`6.3 GB`**) and symlinks the 5 SHA256-identical `coexec/{body}_map.npz` files to `teachers/{body}_{variant}_probes.npz` (saving `2.1 GB` on NFS vs raw `8.4 GB`); rollouts, backups, and temporary captures never touch NFS.
   - Temporary caches (`TORCH_HOME`, `pytest` basetemp, container layers) stay strictly in node-local `/tmp` (`970 GB` free) and are removed at job exit.
   - Intermediate Wave 2 `.pt` checkpoints not needed by downstream jobs are pruned after scoring (or archived to Google Drive `lHumanoid-Soccer-Isaac-Lab-Archive`).
5. **Single-branch git traceability:** all verified results and dispatch ledgers are pulled back via `scripts/mesogip/05_pull_results.ps1` and committed on `feat/icml-token-latent` under `Santiago23florido <sanflogom@gmail.com>`.

## 3. Staged Wave Schedule (Zero Throwaway Compute)

- **Wave 1 (100 % frozen, independent of decoder v2.2 vs v2.3 decision — launched immediately on `ENSTA-h100`):**
  - `cross` (`4` remaining jobs: `x_s1, x_s2, a_s1, a_s2`) → completes **6/6** (full 3-seed G-X and G-A3 tables).
  - `r4` (`30` jobs: `encoder` + `translator` × 5 bodies × 3 seeds) → completes **30/30** (full 3-seed R4 unseen-body gap closure).
- **Local concurrent track (already running on Windows RTX):**
  - `main/pact_na_s0` → `main/urma_d_s0` (seed-0 closed-loop R1/R2 contrast against `main/pact_d_s0`).
- **Wave 2 (unlocked after reading seed-0 R3 closed-loop contrast and freezing final decoder version):**
  - `loo` (`69` remaining jobs across 5 arms × 5 bodies × 3 seeds) on `ENSTA-h100`.
  - `scaling` (`60` jobs for H4) on `ENSTA-h100`.
  - `adapt` (`450` lightweight G-N3 jobs, unlocked automatically as `loo` models finish) on `ENSTA-h100`.

## 4. Traceability Ledger (Software Engineering Report Artifact)

- Every Mesogip task writes a machine-readable provenance record to `logs/icml/mesogip_provenance/<job_id>.json` containing `host`, `partition`, `slurm_job_id`, `gpu_model`, `git_commit`, `started_utc`, `ended_utc`, `wall_seconds`, `returncode`, and `outputs`.
- `scripts/mesogip/update_dispatch_ledger.py` aggregates the manifest (`icml_runs.py`), local completed outputs, Mesogip provenance records, and node probe (`node_probe.json`) into **`transfer/results/cluster_dispatch_ledger.json`** (version-controlled on `feat/icml-token-latent`).
