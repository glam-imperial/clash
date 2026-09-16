from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CONDITION_COLORS = {"O": "#2F4858", "L": "#3A7D44", "P": "#D17B0F", "F": "#8C8C8C"}
EFFECT_COLORS = {"L_minus_P": "#2B6CB0", "O_minus_L": "#B83280"}


def _pyplot():
    try:
        import matplotlib

        # Paper figures are written to files; a GUI backend is unnecessary and
        # can fail on headless servers or Windows installations without Qt.
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("visualization requires matplotlib") from exc
    return plt


def plot_condition_aurocs(table: pd.DataFrame, destination: Path) -> None:
    plt = _pyplot()
    labels = table.apply(
        lambda row: f"{row['model']} | {row['setting']} | {row['dataset']}", axis=1
    )
    y = np.arange(len(table))
    figure, axis = plt.subplots(figsize=(10, max(4, len(table) * 0.34)))
    for condition in ("O", "L", "P", "F"):
        axis.scatter(
            table[f"auc_{condition}"], y, label=condition, color=CONDITION_COLORS[condition]
        )
    axis.axvline(0.5, color="#555555", linestyle="--", linewidth=1)
    axis.set_yticks(y, labels)
    axis.set_xlabel("AUROC")
    axis.set_title("Condition-wise discrimination")
    axis.legend(ncol=4, frameon=False)
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=200)
    plt.close(figure)


def plot_primary_effects(table: pd.DataFrame, destination: Path) -> None:
    plt = _pyplot()
    labels = table.apply(
        lambda row: f"{row['model']} | {row['setting']} | {row['dataset']}", axis=1
    )
    y = np.arange(len(table), dtype=float)
    figure, axis = plt.subplots(figsize=(10, max(4, len(table) * 0.38)))
    for offset, effect in ((-0.12, "L_minus_P"), (0.12, "O_minus_L")):
        values = table[effect].to_numpy(dtype=float)
        lower = table[f"{effect}_ci_lower"].to_numpy(dtype=float)
        upper = table[f"{effect}_ci_upper"].to_numpy(dtype=float)
        axis.errorbar(
            values,
            y + offset,
            xerr=np.vstack([values - lower, upper - values]),
            fmt="o",
            capsize=2,
            color=EFFECT_COLORS[effect],
            label=effect,
        )
    axis.axvline(0.0, color="#555555", linestyle="--", linewidth=1)
    axis.set_yticks(y, labels)
    axis.set_xlabel("AUROC difference (95% cluster-bootstrap CI)")
    axis.set_title("Primary effects independent of the F baseline")
    axis.legend(frameon=False)
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=200)
    plt.close(figure)


def plot_score_comparison(table: pd.DataFrame, destination: Path) -> None:
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(7, 5))
    for condition, group in table.groupby("condition", sort=True):
        axis.scatter(
            group["auc_token_set"],
            group["auc_single_token"],
            label=condition,
            color=CONDITION_COLORS.get(str(condition), "#333333"),
        )
    limits = [
        min(table["auc_token_set"].min(), table["auc_single_token"].min()) - 0.02,
        max(table["auc_token_set"].max(), table["auc_single_token"].max()) + 0.02,
    ]
    axis.plot(limits, limits, color="#555555", linestyle="--", linewidth=1)
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_xlabel("Token-set logsumexp AUROC")
    axis.set_ylabel("Single-token AUROC")
    axis.set_title("Sensitivity to score extraction")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=200)
    plt.close(figure)


def generate_figures(output_dir: Path) -> list[Path]:
    token_effects = pd.read_csv(output_dir / "full_token_set_auc_effects.csv")
    comparison = pd.read_csv(output_dir / "score_method_comparison.csv")
    seven_model = pd.read_csv(output_dir / "seven_model_auc_effects.csv")
    destinations = [
        output_dir / "full_condition_aurocs.png",
        output_dir / "seven_model_primary_effects.png",
        output_dir / "score_method_comparison.png",
    ]
    plot_condition_aurocs(token_effects, destinations[0])
    plot_primary_effects(seven_model, destinations[1])
    plot_score_comparison(comparison, destinations[2])
    return destinations


def main() -> None:
    parser = argparse.ArgumentParser(description="Render CLASH analysis figures.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in generate_figures(args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
