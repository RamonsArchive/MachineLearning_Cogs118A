"""
EDA for the ModalAI part-week demand panel (REGRESSION).

Mirrors eda_parkinsons.py: a text report (with naive-baseline floor) plus figures.
Focus is the two things that dominate this target: zero-inflation (~94% zeros) and a
heavy recount-driven tail. Outputs -> plots/forecast_demand_plots/eda/.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

from utils.clean.clean_forecast_demand import TARGET_COL


def eda_forecast_demand(clean_df):
    output_dir = "plots/forecast_demand_plots/eda"
    os.makedirs(output_dir, exist_ok=True)

    df = clean_df.copy()
    y = df[TARGET_COL].astype(float)

    print("\n[EDA] Generating ModalAI demand EDA...")
    print(f"[EDA] Output directory: {output_dir}")

    nonzero = y[y > 0]
    zero_frac = float((y == 0).mean())

    report = []
    report.append("=" * 70)
    report.append("MODALAI PART/COMPONENT DEMAND - EXPLORATORY DATA ANALYSIS")
    report.append("=" * 70)
    report.append("\nTask: REGRESSION (predict weekly demand_qty per part)")
    report.append(f"Target: {TARGET_COL} = max(sh_outflow_qty, build_consumed_qty)")

    report.append("\n" + "-" * 50)
    report.append("DATASET OVERVIEW")
    report.append("-" * 50)
    report.append(f"Part-weeks (rows):     {len(df)}")
    report.append(f"Distinct weeks:        {df['week_start'].nunique()}")
    report.append(f"Week range:            {df['week_start'].min().date()} -> "
                  f"{df['week_start'].max().date()}")

    report.append("\n" + "-" * 50)
    report.append("TARGET DISTRIBUTION (demand_qty) - zero-inflated & heavy-tailed")
    report.append("-" * 50)
    report.append(f"Zero part-weeks:       {zero_frac*100:.2f}% (intermittent demand)")
    report.append(f"Mean (all):            {y.mean():.4f}")
    report.append(f"Std  (all):            {y.std():.4f}")
    report.append(f"Max  (all):            {y.max():.1f}   <- recount/correction spikes")
    for p in (0.5, 0.9, 0.99, 0.999):
        report.append(f"P{p*100:<5.1f} (all):         {y.quantile(p):.4f}")
    report.append(f"\nAmong NON-ZERO part-weeks (n={len(nonzero)}):")
    report.append(f"  Mean: {nonzero.mean():.4f}  Median: {nonzero.median():.4f}  "
                  f"Max: {nonzero.max():.1f}")

    # --- Naive-baseline floor (mean predictor on the raw target) ---
    report.append("\n" + "-" * 50)
    report.append("NAIVE BASELINE FLOOR (raw target)")
    report.append("-" * 50)
    y_pred_mean = np.full_like(y, y.mean())
    rmse_mean = np.sqrt(mean_squared_error(y, y_pred_mean))
    mae_mean = mean_absolute_error(y, y_pred_mean)
    report.append("Mean predictor (always predict global mean):")
    report.append(f"  RMSE: {rmse_mean:.4f}")
    report.append(f"  MAE:  {mae_mean:.4f}")
    report.append(f"  R2:   0.0000 (by definition)")
    report.append("\nNote: the experiment ALSO reports last-value, 4-week moving-average,")
    report.append("and Croston baselines on the held-out weeks (a fairer intermittent floor")
    report.append("than the global mean), and evaluates models against them.")

    report.append("\n" + "-" * 50)
    report.append("MODELING NOTES (applied in the experiment)")
    report.append("-" * 50)
    report.append("- Winsorize target upper tail (recount spikes) + log1p; invert for metrics.")
    report.append("- Temporal split (test weeks strictly later than train) - NOT shuffled.")
    report.append("- XGB keeps NaN natively; RF/NN median-impute + missingness indicator.")
    report.append("- Static part attributes are CURRENT snapshots (backtesting caveat).")

    report_path = os.path.join(output_dir, "eda_forecast_demand_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"[EDA] Saved report to {report_path}")

    # ---- Figure 1: target distribution (raw nonzero + log1p) ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].hist(np.log1p(nonzero), bins=60, color="steelblue",
                 edgecolor="black", alpha=0.75)
    axes[0].set_title("log1p(demand) | non-zero part-weeks")
    axes[0].set_xlabel("log1p(demand_qty)")
    axes[0].set_ylabel("Frequency")

    labels = ["zero", "non-zero"]
    axes[1].bar(labels, [(y == 0).sum(), (y > 0).sum()],
                color=["lightgray", "steelblue"], edgecolor="black")
    axes[1].set_title(f"Zero-inflation: {zero_frac*100:.1f}% zeros")
    axes[1].set_ylabel("Part-weeks")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "target_distribution.png"), dpi=150)
    plt.close()
    print("[EDA] Saved target_distribution.png")

    # ---- Figure 2: total demand over time (weekly) ----
    weekly = df.groupby("week_start")[TARGET_COL].sum()
    plt.figure(figsize=(11, 4.5))
    plt.plot(weekly.index, weekly.values, marker="o", markersize=3, color="steelblue")
    plt.title("Total weekly demand across all parts (recount spikes visible)")
    plt.xlabel("Week")
    plt.ylabel("Sum demand_qty")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "weekly_total_demand.png"), dpi=150)
    plt.close()
    print("[EDA] Saved weekly_total_demand.png")

    print(f"\n[EDA] Complete! Outputs saved to {output_dir}/")
    return df
