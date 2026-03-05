"""
PINN λ Ablation Study — automated sweep & visualization.

Sweeps over different values of the conservation loss weight λ,
collects Test-set MAE and WMAPE from each run, and produces
a publication-quality trade-off curve.

Usage:
    python scripts/run_pinn_ablation.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ─── Configuration ────────────────────────────────────────────────────────────
BASE_CFG = "configs/GatedGCN/network-pairs-topology.yaml"
LAMBDA_VALUES = [0.0, 0.01, 0.1, 1.0, 10.0]
RESULTS_ROOT = Path("results")
ABLATION_TAG = "pinn-lambda-ablation"
FIGURE_DIR = Path("results") / ABLATION_TAG
PYTHON = sys.executable


def run_single_experiment(lam: float) -> dict:
    """Launch main.py with a specific lambda_cons and return test metrics."""
    lam_str = f"{lam:.4g}".replace("+", "")
    name_tag = f"{ABLATION_TAG}-lambda{lam_str}"

    cmd = [
        PYTHON, "main.py",
        "--cfg", BASE_CFG,
        "--repeat", "1",
        "--opts",
        f"model.lambda_cons", str(lam),
        f"name_tag", name_tag,
    ]

    print(f"\n{'='*70}")
    print(f"  Running λ = {lam}  |  name_tag = {name_tag}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"[WARNING] Experiment λ={lam} exited with code {result.returncode}")

    cfg_basename = Path(BASE_CFG).stem
    run_dir_name = f"{cfg_basename}-{name_tag}"
    out_dir = RESULTS_ROOT / run_dir_name

    metrics = _parse_test_metrics(out_dir)
    metrics["lambda"] = lam
    return metrics


def _parse_test_metrics(out_dir: Path) -> dict:
    """
    Parse best test MAE and WMAPE from the run directory.

    The logger writes per-epoch stats to {run_dir}/{seed}/test/stats.json.
    Each line is a JSON dict with keys: epoch, loss, mae, wmape, ...
    We pick the epoch with the lowest val loss (mirroring the training loop's
    best-epoch selection) and report the corresponding test metrics.
    """
    mae, wmape = float("nan"), float("nan")

    seed_dirs = sorted(out_dir.iterdir()) if out_dir.exists() else []
    for seed_dir in seed_dirs:
        if not seed_dir.is_dir():
            continue

        test_stats_file = seed_dir / "test" / "stats.json"
        val_stats_file = seed_dir / "val" / "stats.json"

        if not test_stats_file.exists():
            print(f"[WARNING] stats.json not found: {test_stats_file}")
            continue

        test_records = _read_jsonl(test_stats_file)
        val_records = _read_jsonl(val_stats_file) if val_stats_file.exists() else test_records

        if not val_records or not test_records:
            continue

        best_epoch_idx = int(np.argmin([r.get("loss", float("inf")) for r in val_records]))
        best_epoch_idx = min(best_epoch_idx, len(test_records) - 1)

        best_test = test_records[best_epoch_idx]
        mae = best_test.get("mae", float("nan"))
        wmape = best_test.get("wmape", float("nan"))
        break

    return {"mae": mae, "wmape": wmape}


def _read_jsonl(path: Path) -> list:
    """Read a file where each line is a JSON object."""
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def plot_tradeoff(results: list[dict], save_dir: Path):
    """Draw a publication-quality λ trade-off curve."""
    save_dir.mkdir(parents=True, exist_ok=True)

    lambdas = [r["lambda"] for r in results]
    maes = [r["mae"] for r in results]
    wmapes = [r["wmape"] for r in results]

    x_labels = [f"{l:.4g}" if l > 0 else "0" for l in lambdas]
    x_pos = np.arange(len(lambdas))

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.dpi": 150,
    })

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color_mae = "#2563EB"
    color_wmape = "#DC2626"

    ln1 = ax1.plot(x_pos, maes, "o-", color=color_mae, linewidth=2,
                   markersize=8, label="MAE", zorder=3)
    ax1.set_xlabel(r"Conservation Loss Weight $\lambda$")
    ax1.set_ylabel("MAE (normalized)", color=color_mae)
    ax1.tick_params(axis="y", labelcolor=color_mae)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels)

    for i, (xp, m) in enumerate(zip(x_pos, maes)):
        ax1.annotate(f"{m:.4f}", (xp, m), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9, color=color_mae)

    ax2 = ax1.twinx()
    ln2 = ax2.plot(x_pos, wmapes, "s--", color=color_wmape, linewidth=2,
                   markersize=8, label="WMAPE", zorder=3)
    ax2.set_ylabel("WMAPE", color=color_wmape)
    ax2.tick_params(axis="y", labelcolor=color_wmape)

    for i, (xp, w) in enumerate(zip(x_pos, wmapes)):
        ax2.annotate(f"{w:.4f}", (xp, w), textcoords="offset points",
                     xytext=(0, -15), ha="center", fontsize=9, color=color_wmape)

    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="upper center", framealpha=0.9)

    ax1.set_title(r"PINN Conservation Loss Weight $\lambda$ — Trade-off Curve")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()

    png_path = save_dir / "pinn_lambda_tradeoff.png"
    pdf_path = save_dir / "pinn_lambda_tradeoff.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[✓] Figures saved to:\n    {png_path}\n    {pdf_path}")


def main():
    all_results = []

    for lam in LAMBDA_VALUES:
        metrics = run_single_experiment(lam)
        all_results.append(metrics)
        print(f"  λ={lam:>6}  →  MAE={metrics['mae']:.5f}  WMAPE={metrics['wmape']:.5f}")

    print(f"\n{'='*70}")
    print("  Summary")
    print(f"{'='*70}")
    print(f"  {'λ':>10}  {'MAE':>10}  {'WMAPE':>10}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*10}")
    for r in all_results:
        print(f"  {r['lambda']:>10.4g}  {r['mae']:>10.5f}  {r['wmape']:>10.5f}")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = FIGURE_DIR / "ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[✓] Summary JSON saved to: {summary_path}")

    plot_tradeoff(all_results, FIGURE_DIR)


if __name__ == "__main__":
    main()
