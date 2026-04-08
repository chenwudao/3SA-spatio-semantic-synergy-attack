from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def aggregate_results(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"image_id", "method", "target_model", "attack_success_rate", "ssim_score"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")
    grouped = (
        df.groupby(["method", "target_model"], as_index=False)
        .agg(mean_asr=("attack_success_rate", "mean"), mean_ssim=("ssim_score", "mean"))
        .sort_values(["target_model", "mean_ssim", "mean_asr"], ascending=[True, False, False])
    )
    return grouped


def compute_pareto_frontier(df: pd.DataFrame) -> pd.DataFrame:
    ordered = df.sort_values(["mean_ssim", "mean_asr"], ascending=[False, False]).reset_index(drop=True)
    frontier_rows = []
    best_asr = -1.0
    for _, row in ordered.iterrows():
        if row["mean_asr"] >= best_asr:
            frontier_rows.append(row)
            best_asr = row["mean_asr"]
    return pd.DataFrame(frontier_rows)


def plot_pareto_frontier(df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    frontier = compute_pareto_frontier(df)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 6), dpi=180)

    for method, method_df in df.groupby("method"):
        ax.scatter(
            method_df["mean_ssim"],
            method_df["mean_asr"],
            s=80,
            alpha=0.9,
            label=method,
        )

    if not frontier.empty:
        ax.plot(
            frontier["mean_ssim"],
            frontier["mean_asr"],
            linestyle="--",
            linewidth=1.5,
            color="black",
            label="Pareto Frontier",
        )

    ax.set_xlabel("Mean SSIM")
    ax.set_ylabel("Mean ASR")
    ax.set_title("3SA Pareto Frontier")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
