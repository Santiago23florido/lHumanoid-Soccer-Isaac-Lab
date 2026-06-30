#!/usr/bin/env python
"""Train a Physically-Informed Neural Network (DeLaN) on the simulated Franka arm.

This script is the primary deliverable for the PINN-training checkpoint.  It:

  1. Generates (q, q̇, τ, q̈) transitions from the analytic 7-DoF Franka
     simulator (``FrankaAnalytic7DoF`` in ``envs/analytic_systems.py``).
  2. Trains the Deep Lagrangian Network (DeLaN — the PINN) and an unstructured
     MLP baseline on the same data split.
  3. Evaluates both models on a held-out test set and reports one-step
     acceleration RMSE per joint.
  4. Saves three figures and a JSON results file:
       - figures/pinn_loss_curves.png   — train/test loss vs. epoch
       - figures/pinn_accel_scatter.png — true vs. predicted q̈ (test set)
       - figures/pinn_energy.png        — energy conservation under unforced rollout
       - logs/pinn_results.json         — all numbers for the document

Examples
--------
    # Default: 7-DOF Franka-like simulator, 1024 training samples
    python scripts/train_pinn.py

    # Quick smoke test
    python scripts/train_pinn.py --epochs 20 --n-train 128 --quiet

    # Specify output directory
    python scripts/train_pinn.py --out-dir results/pinn_run1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

# ── project imports ───────────────────────────────────────────────────────────
from lagrangian_mbrl.envs.analytic_systems import FrankaAnalytic7DoF, make_system
from lagrangian_mbrl.eval.metrics import acceleration_mse, energy_drift, mass_matrix_min_eigenvalue
from lagrangian_mbrl.models import DeepLagrangianNetwork, DeLaNConfig, MLPDynamics, MLPDynamicsConfig
from lagrangian_mbrl.utils.seeding import seed_everything

# ── default hyper-parameters ──────────────────────────────────────────────────
_DEFAULTS = dict(
    system="franka7",
    n_train=1024,
    n_test=2048,
    epochs=600,
    batch_size=128,
    lr=3e-3,
    weight_decay=0.0,
    seed=42,
    dtype="float64",
    # DeLaN architecture
    delan_hidden=(128, 128),
    # MLP architecture (matched capacity: similar param count)
    mlp_hidden=(256, 256, 256),
    eval_every=20,
    energy_steps=500,
    out_dir="logs/pinn",
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _arch_summary(model: torch.nn.Module, name: str) -> list[str]:
    lines = [f"{name} architecture:"]
    for n, m in model.named_modules():
        if isinstance(m, (torch.nn.Linear,)):
            lines.append(f"  {n or 'root'}: Linear({m.in_features} -> {m.out_features})")
        elif isinstance(m, torch.nn.Softplus):
            lines.append(f"  {n}: Softplus")
        elif isinstance(m, torch.nn.SiLU):
            lines.append(f"  {n}: SiLU")
    lines.append(f"  Total parameters: {_count_params(model):,}")
    return lines


def _iterate_minibatches(data: dict[str, Tensor], batch_size: int, gen: torch.Generator):
    n = data["q"].shape[0]
    perm = torch.randperm(n, generator=gen)
    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        yield {k: v[idx] for k, v in data.items()}


def _fit_model(
    model: torch.nn.Module,
    train: dict[str, Tensor],
    test: dict[str, Tensor],
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    gen: torch.Generator,
    eval_every: int,
    quiet: bool,
) -> dict[str, Any]:
    # Input / output normalization
    if hasattr(model, "fit_normalization"):
        x = torch.cat([train["q"], train["qd"], train["tau"]], dim=-1)
        model.fit_normalization(x, train["qdd"])
    if hasattr(model, "fit_input_normalization"):
        model.fit_input_normalization(train["q"])

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.05)

    history: dict[str, list] = {"epoch": [], "train_loss": [], "test_accel_rmse": []}
    best_rmse = float("inf")
    best_state: dict | None = None

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in _iterate_minibatches(train, batch_size, gen):
            opt.zero_grad()
            loss = model.loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        sched.step()

        if epoch % eval_every == 0 or epoch == epochs - 1:
            model.eval()
            mse = acceleration_mse(model, test)
            rmse = mse ** 0.5
            history["epoch"].append(epoch)
            history["train_loss"].append(epoch_loss / max(n_batches, 1))
            history["test_accel_rmse"].append(rmse)
            if rmse < best_rmse:
                best_rmse = rmse
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if not quiet:
                print(
                    f"  epoch {epoch:4d}/{epochs}  "
                    f"train_loss={epoch_loss / max(n_batches, 1):.4e}  "
                    f"test_rmse={rmse:.4e}"
                )

    if best_state is not None:
        model.load_state_dict(best_state)

    history["best_test_accel_rmse"] = best_rmse
    history["final_test_accel_rmse"] = history["test_accel_rmse"][-1]
    return history


# ── per-joint RMSE ────────────────────────────────────────────────────────────

@torch.no_grad()
def _per_joint_rmse(model: torch.nn.Module, test: dict[str, Tensor]) -> list[float]:
    q, qd, tau, qdd = test["q"], test["qd"], test["tau"], test["qdd"]
    dt = next(model.parameters()).dtype
    q, qd, tau, qdd = q.to(dt), qd.to(dt), tau.to(dt), qdd.to(dt)
    if hasattr(model, "predict_acceleration"):
        pred = model.predict_acceleration(q, qd, tau)
    else:
        pred = model.forward_dynamics(q, qd, tau)
    sq = (pred - qdd) ** 2
    return [float(sq[:, j].mean().sqrt().item()) for j in range(q.shape[-1])]


# ── figure generation ─────────────────────────────────────────────────────────

def _plot_loss_curves(
    delan_hist: dict,
    mlp_hist: dict,
    system_name: str,
    n_train: int,
    path: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.semilogy(delan_hist["epoch"], delan_hist["train_loss"], label="DeLaN (PINN)", color="C0", linewidth=1.5)
    ax.semilogy(mlp_hist["epoch"], mlp_hist["train_loss"], label="MLP (unstructured)", color="C1",
                linewidth=1.5, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss (log scale)")
    ax.set_title(f"Training loss — {system_name}, N={n_train}")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    ax.semilogy(delan_hist["epoch"], delan_hist["test_accel_rmse"], label="DeLaN (PINN)", color="C0",
                linewidth=1.5)
    ax.semilogy(mlp_hist["epoch"], mlp_hist["test_accel_rmse"], label="MLP (unstructured)", color="C1",
                linewidth=1.5, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test acceleration RMSE (rad/s²)")
    ax.set_title(f"Test accuracy — {system_name}, N={n_train}")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("Physically-Informed Neural Network vs Unstructured MLP", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


def _plot_accel_scatter(
    delan_model: torch.nn.Module,
    mlp_model: torch.nn.Module,
    test: dict[str, Tensor],
    path: Path,
    n_samples: int = 800,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    dt = next(delan_model.parameters()).dtype
    q = test["q"][:n_samples].to(dt)
    qd = test["qd"][:n_samples].to(dt)
    tau = test["tau"][:n_samples].to(dt)
    qdd_true = test["qdd"][:n_samples].to(dt)

    with torch.no_grad():
        qdd_delan = delan_model.predict_acceleration(q, qd, tau)
        qdd_mlp = mlp_model.predict_acceleration(q, qd, tau)

    # Plot first 4 joints
    n_joints = min(4, q.shape[-1])
    fig, axes = plt.subplots(2, n_joints, figsize=(4 * n_joints, 8))

    for row, (label, pred) in enumerate([("DeLaN (PINN)", qdd_delan), ("MLP", qdd_mlp)]):
        for j in range(n_joints):
            ax = axes[row, j]
            true_j = qdd_true[:, j].cpu().numpy()
            pred_j = pred[:, j].cpu().numpy()
            lim = max(abs(true_j).max(), abs(pred_j).max()) * 1.05
            ax.scatter(true_j, pred_j, alpha=0.25, s=6, color="C0" if row == 0 else "C1")
            ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=1.0, alpha=0.6)
            rmse = float(((pred_j - true_j) ** 2).mean() ** 0.5)
            ax.set_title(f"{label}\njoint {j + 1}  RMSE={rmse:.3e}", fontsize=9)
            ax.set_xlabel("True q̈ (rad/s²)")
            if j == 0:
                ax.set_ylabel("Predicted q̈ (rad/s²)")
            ax.grid(True, alpha=0.3)

    fig.suptitle("True vs Predicted Accelerations — Test Set", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


def _plot_energy(
    delan_model: torch.nn.Module,
    q0: Tensor,
    qd0: Tensor,
    steps: int,
    dt: float,
    path: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    drift = energy_drift(delan_model, q0.numpy(), qd0.numpy(), steps=steps, dt=dt)
    # drift shape: (steps+1, 1) or (steps+1,)
    if drift.ndim > 1:
        drift = drift[:, 0]

    fig, ax = plt.subplots(figsize=(8, 4))
    t = np.arange(len(drift)) * dt
    ax.plot(t, drift, color="C0", linewidth=1.5)
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("|E(t) − E(0)|  (J)")
    ax.set_title("Energy conservation under unforced rollout — DeLaN (PINN)")
    ax.grid(True, alpha=0.3)
    final_drift = float(drift[-1])
    ax.text(0.97, 0.92, f"Final drift: {final_drift:.3e} J",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="C0", alpha=0.7))
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--system", default=_DEFAULTS["system"],
                   choices=["franka7", "two_link", "pendulum"],
                   help="Analytic rigid-body simulator (default: franka7)")
    p.add_argument("--n-train", type=int, default=_DEFAULTS["n_train"])
    p.add_argument("--n-test", type=int, default=_DEFAULTS["n_test"])
    p.add_argument("--epochs", type=int, default=_DEFAULTS["epochs"])
    p.add_argument("--batch-size", type=int, default=_DEFAULTS["batch_size"])
    p.add_argument("--lr", type=float, default=_DEFAULTS["lr"])
    p.add_argument("--seed", type=int, default=_DEFAULTS["seed"])
    p.add_argument("--out-dir", type=Path, default=_DEFAULTS["out_dir"])
    p.add_argument("--quiet", action="store_true", help="Suppress per-epoch output")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    fig_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    gen = torch.Generator().manual_seed(args.seed)

    # ── 1. Generate data ──────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  Physically-Informed Neural Network — Franka Arm Dynamics")
    print(f"{'='*65}")
    print(f"\n[1/5] Generating data from '{args.system}' simulator …")
    system = make_system(args.system)
    dof = system.dof
    train_data = system.sample_dataset(args.n_train, generator=gen, dtype=dtype)
    test_data = system.sample_dataset(args.n_test, generator=gen, dtype=dtype)
    print(f"      train={args.n_train}, test={args.n_test}, dof={dof}")

    # ── 2. Build models ───────────────────────────────────────────────────────
    print("\n[2/5] Building models …")
    delan = DeepLagrangianNetwork(
        DeLaNConfig(
            dof=dof,
            hidden_sizes=(128, 128),
            activation="softplus",
            loss_type="forward",  # directly optimize acceleration prediction
            epsilon=1e-3,          # stronger PD floor keeps M(q) well-conditioned
        )
    ).to(dtype)
    mlp = MLPDynamics(
        MLPDynamicsConfig(
            dof=dof,
            state_dim=2 * dof,
            action_dim=dof,
            hidden_sizes=(256, 256, 256),
            probabilistic=False,
            target="acceleration",
        )
    ).to(dtype)

    delan_params = _count_params(delan)
    mlp_params = _count_params(mlp)

    for line in _arch_summary(delan, "DeLaN (PINN)"):
        print(f"  {line}")
    print()
    for line in _arch_summary(mlp, "MLP (unstructured baseline)"):
        print(f"  {line}")

    # ── 3. Train models ───────────────────────────────────────────────────────
    print(f"\n[3/5] Training … ({args.epochs} epochs each)")
    common_kw = dict(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=0.0,
        gen=gen,
        eval_every=_DEFAULTS["eval_every"],
        quiet=args.quiet,
    )

    print(f"\n  — DeLaN (PINN) —")
    t0 = time.perf_counter()
    delan_hist = _fit_model(delan, train_data, test_data, **common_kw)
    delan_time = time.perf_counter() - t0

    print(f"\n  — MLP (unstructured) —")
    t0 = time.perf_counter()
    mlp_hist = _fit_model(mlp, train_data, test_data, **common_kw)
    mlp_time = time.perf_counter() - t0

    # ── 4. Evaluate ───────────────────────────────────────────────────────────
    print("\n[4/5] Evaluating on test set …")
    delan.eval()
    mlp.eval()

    delan_test_rmse = delan_hist["best_test_accel_rmse"]
    mlp_test_rmse = mlp_hist["best_test_accel_rmse"]
    delan_joint_rmse = _per_joint_rmse(delan, test_data)
    mlp_joint_rmse = _per_joint_rmse(mlp, test_data)

    # Energy conservation check (unforced rollout from a random state)
    q0 = test_data["q"][:1]
    qd0 = test_data["qd"][:1]
    energy_steps = _DEFAULTS["energy_steps"]
    dt_sim = 1e-3

    # Min eigenvalue of M(q) — must be > 0 for a valid PINN
    eig_min = mass_matrix_min_eigenvalue(delan, test_data["q"][:200].numpy())

    # ── 5. Print results ──────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  RESULTS")
    print(f"{'='*65}")
    print(f"\n  System            : {args.system} (DoF={dof})")
    print(f"  Training samples  : {args.n_train}")
    print(f"  Test samples      : {args.n_test}")
    print(f"\n  {'Model':<28} {'Params':>8}  {'Test RMSE (rad/s²)':>20}  {'Time (s)':>10}")
    print(f"  {'-'*70}")
    print(f"  {'DeLaN (PINN)':<28} {delan_params:>8,}  {delan_test_rmse:>20.6e}  {delan_time:>10.1f}")
    print(f"  {'MLP (unstructured)':<28} {mlp_params:>8,}  {mlp_test_rmse:>20.6e}  {mlp_time:>10.1f}")
    print(f"\n  Improvement (MLP / DeLaN): {mlp_test_rmse / delan_test_rmse:.2f}x" if delan_test_rmse > 0 else "")
    print(f"\n  Per-joint test RMSE (rad/s²):")
    print(f"  {'Joint':<6}  {'DeLaN':>14}  {'MLP':>14}")
    for j in range(dof):
        print(f"  {j+1:<6}  {delan_joint_rmse[j]:>14.4e}  {mlp_joint_rmse[j]:>14.4e}")
    print(f"\n  DeLaN M(q) min eigenvalue: {eig_min.min():.4e} (must be > 0)")
    print(f"  DeLaN passes PD check: {'YES' if eig_min.min() > 0 else 'NO'}")

    # ── 6. Save figures ───────────────────────────────────────────────────────
    print("\n[5/5] Saving figures …")
    _plot_loss_curves(delan_hist, mlp_hist, args.system, args.n_train,
                      fig_dir / "pinn_loss_curves.png")
    _plot_accel_scatter(delan, mlp, test_data, fig_dir / "pinn_accel_scatter.png")
    _plot_energy(delan, q0.to(dtype), qd0.to(dtype), energy_steps, dt_sim,
                 fig_dir / "pinn_energy.png")

    # ── 7. Save JSON results ──────────────────────────────────────────────────
    results = {
        "system": args.system,
        "dof": dof,
        "n_train": args.n_train,
        "n_test": args.n_test,
        "seed": args.seed,
        "epochs": args.epochs,
        "delan": {
            "params": delan_params,
            "hidden_sizes": [128, 128],
            "activation": "softplus",
            "loss_type": "forward",
            "best_test_accel_rmse": delan_test_rmse,
            "final_test_accel_rmse": delan_hist["final_test_accel_rmse"],
            "per_joint_rmse": delan_joint_rmse,
            "mass_matrix_min_eigenvalue": float(eig_min.min()),
            "wall_time_s": delan_time,
            "epoch_history": {
                "epoch": delan_hist["epoch"],
                "train_loss": delan_hist["train_loss"],
                "test_accel_rmse": delan_hist["test_accel_rmse"],
            },
        },
        "mlp": {
            "params": mlp_params,
            "hidden_sizes": [256, 256, 256],
            "best_test_accel_rmse": mlp_test_rmse,
            "final_test_accel_rmse": mlp_hist["final_test_accel_rmse"],
            "per_joint_rmse": mlp_joint_rmse,
            "wall_time_s": mlp_time,
            "epoch_history": {
                "epoch": mlp_hist["epoch"],
                "train_loss": mlp_hist["train_loss"],
                "test_accel_rmse": mlp_hist["test_accel_rmse"],
            },
        },
        "improvement_ratio": mlp_test_rmse / delan_test_rmse if delan_test_rmse > 0 else None,
        "delan_wins": delan_test_rmse < mlp_test_rmse,
        "figures": [
            "figures/pinn_loss_curves.png",
            "figures/pinn_accel_scatter.png",
            "figures/pinn_energy.png",
        ],
    }

    results_path = out_dir / "pinn_results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"  saved: {results_path}")

    print(f"\n{'='*65}")
    verdict = "PASS ✓" if results["delan_wins"] else "FAIL ✗"
    print(f"  DeLaN < MLP on test RMSE: {verdict}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
