# Running the ICML manifest on a cluster

Every run the paper's tables and figures need is one job in `transfer/scripts/icml_runs.py`.
Jobs come in two kinds:

- **offline:** PyTorch and NumPy only. They run on any CUDA GPU, A100/H100 included.
- **isaac:** these need Isaac Sim 5.1, so a GPU with RT cores (L40S, A40, RTX 6000 Ada or
  similar). Isaac Sim does not support A100/H100.

```bash
python transfer/scripts/icml_runs.py status          # done / runnable / left, per group
python transfer/scripts/icml_runs.py next --kind offline
```

## Offline jobs (any GPU)

1. Clone the repository and check out the paper's branch.
2. Create an environment with Python 3.11, `torch==2.7.0` (the CUDA build of the cluster) and
   `numpy==1.26.0`. `tests/test_offline_imports.py` checks that no Isaac or rsl_rl import is
   needed. Then run the offline tests on the cluster:

   ```bash
   python -m pytest tests/test_offline_imports.py tests/test_drive_ratio.py \
       tests/test_icml_runs.py tests/test_icml_r4_read.py tests/test_pact_windows.py
   ```

3. Data. `python transfer/scripts/icml_cluster_data.py --tar logs/icml/cluster_data.tar` packs
   what offline jobs read. That is each body's final-teacher audit captures, the final
   co-executions and the walk contracts, about 5 GB. Unpack it at the repository root on the
   cluster.
4. Jobs. `python transfer/scripts/icml_runs.py slurm --kind offline --partition <gpu partition>
   --hours 4` writes one SLURM array per group under `logs/icml/slurm/`. Submit them with:

   ```bash
   ICML_REPO=$PWD PYTHON=python sbatch logs/icml/slurm/<group>.sbatch
   ```

   Rerun `slurm` after a group finishes: jobs that need another job's model (G-N3 needs the
   leave-one-out models) become runnable only then.
5. Bring back `transfer/results/icml_*.json` (small). Models stay on the cluster unless a later
   Isaac job drives them (`logs/icml/models/`).

## Isaac jobs

These are `main` (the recipe per arm and seed) and `h2`.

- **On this machine:** about 7 h per arm and seed (`icml_runs.py status` totals them). Run them
  one at a time with `bash transfer/scripts/icml_recipe.sh <arm> <seed>`. Each run removes its
  own captures at the end (stage `clean`), keeping summaries.
- **On a cluster with RT-core GPUs:** use Isaac Lab's container workflow (Docker image to
  Apptainer, `docker/cluster/` in Isaac Lab, SLURM or PBS), headless.
  - The robot USDs (G1, H1, Cassie, Digit) are fetched from the Omniverse asset server at
    launch. Nodes without internet need them cached.
  - The NAO asset is CC BY-NC-ND: copied privately, never committed or redistributed.

## Disk

- A capture is about 0.5 GB; one recipe run writes about 35 before its `clean` stage.
- Leave-one-out, scaling and G-N3 jobs write models of a few MB.
- `transfer/scripts/pact_archive_captures.py` and `prune_checkpoints.py` keep a run's
  summaries and remove the rest.

## Questions for the cluster's administrators

1. Which GPU models are there? RT cores decide whether Isaac Sim runs.
2. What NVIDIA driver version do the nodes have?
3. Is there a container runtime (Apptainer or Singularity)?
4. Which scheduler is used (SLURM or PBS), and what is the longest job time?
5. Do the compute nodes have internet access, for the robot assets?
6. How much storage is there per user (scratch and home)?
