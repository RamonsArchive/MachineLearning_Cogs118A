import os
from io import StringIO

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def eda_thyroid(data: pd.DataFrame) -> pd.DataFrame:
    """
    Exploratory Data Analysis for the Thyroid Cancer dataset (classification).

    - Target: 'Recurred' ('Yes' / 'No')
    - Outputs:
      - Text report with dataset overview and naive baselines
      - Target distribution plot
      - Simple bar plots for a few key categorical features
    """
    output_dir = "plots/thyroid_plots/eda"
    os.makedirs(output_dir, exist_ok=True)

    df = data.copy()
    target_col = "Recurred"

    print(f"\n[EDA] Generating Thyroid Cancer EDA...")
    print(f"[EDA] Output directory: {output_dir}")

    # ==========================================
    # 1. TEXT REPORT
    # ==========================================
    report_path = os.path.join(output_dir, "eda_thyroid_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("THYROID CANCER DATASET – EXPLORATORY DATA ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        f.write("DATA INFO:\n")
        f.write("-" * 80 + "\n")
        buf = StringIO()
        df.info(buf=buf)
        f.write(buf.getvalue())
        f.write("\n\n")

        f.write("DATA DESCRIBE (NUMERIC FEATURES):\n")
        f.write("-" * 80 + "\n")
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        if len(numeric_cols) > 0:
            f.write(str(df[numeric_cols].describe()))
        else:
            f.write("No purely numeric columns.\n")
        f.write("\n\n")

        # ===== NAIVE BASELINES =====
        f.write("=" * 80 + "\n")
        f.write("NAIVE BASELINES (What to beat!)\n")
        f.write("=" * 80 + "\n\n")

        counts = df[target_col].value_counts()
        n_total = len(df)
        majority_label = counts.idxmax()
        majority_acc = counts.max() / n_total

        f.write("Target Distribution (Recurred):\n")
        f.write("-" * 40 + "\n")
        for label, cnt in counts.items():
            f.write(f"  {label}: {cnt:3d} ({cnt / n_total:.2%})\n")
        f.write(f"  Total: {n_total}\n\n")

        f.write("Baseline 1: Majority Class Classifier (always predict most frequent label)\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Majority label: {majority_label}\n")
        f.write(f"  Accuracy:       {majority_acc:.4f} ({majority_acc:.2%})\n\n")

        f.write("Your models (XGBoost, Random Forest, Neural Net) should beat this accuracy.\n")

    print(f"[EDA] Saved report to {report_path}")

    # ==========================================
    # 2. PLOTS
    # ==========================================
    # --- Target Distribution ---
    plt.figure(figsize=(6, 4))
    sns.countplot(x=target_col, data=df)
    plt.title("Thyroid Cancer – Recurrence Distribution")
    plt.xlabel("Recurred")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "thyroid_recurred_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("[EDA] Saved thyroid_recurred_distribution.png")

    # --- Example categorical feature plots (up to 3) ---
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if target_col in cat_cols:
        cat_cols.remove(target_col)

    for col in cat_cols[:3]:
        plt.figure(figsize=(8, 4))
        sns.countplot(x=col, hue=target_col, data=df)
        plt.title(f"{col} by Recurrence")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"thyroid_{col}_by_recurred.png"), dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[EDA] Saved thyroid_{col}_by_recurred.png")

    print(f"\n[EDA] Complete! All outputs saved to {output_dir}/")
    return df


