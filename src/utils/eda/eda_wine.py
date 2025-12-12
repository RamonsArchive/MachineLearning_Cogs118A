import os
from io import StringIO

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def eda_wine(data: pd.DataFrame) -> pd.DataFrame:
    """
    Exploratory Data Analysis for the Wine dataset (classification).

    - Target: 'Class' (1, 2, 3)
    - Outputs:
      - Text report with dataset overview and naive baselines
      - Class distribution plot
      - Correlation heatmap of numeric features
    """
    output_dir = "plots/wine_plots/eda"
    os.makedirs(output_dir, exist_ok=True)

    df = data.copy()
    target_col = "Class"

    print(f"\n[EDA] Generating Wine EDA...")
    print(f"[EDA] Output directory: {output_dir}")

    # ==========================================
    # 1. TEXT REPORT
    # ==========================================
    report_path = os.path.join(output_dir, "eda_wine_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("WINE DATASET – EXPLORATORY DATA ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        f.write("DATA INFO:\n")
        f.write("-" * 80 + "\n")
        buf = StringIO()
        df.info(buf=buf)
        f.write(buf.getvalue())
        f.write("\n\n")

        f.write("DATA DESCRIBE (NUMERIC FEATURES):\n")
        f.write("-" * 80 + "\n")
        f.write(str(df.describe()))
        f.write("\n\n")

        # ===== NAIVE BASELINES =====
        f.write("=" * 80 + "\n")
        f.write("NAIVE BASELINES (What to beat!)\n")
        f.write("=" * 80 + "\n\n")

        class_counts = df[target_col].value_counts().sort_index()
        n_total = len(df)
        majority_class = class_counts.idxmax()
        majority_acc = class_counts.max() / n_total

        f.write("Class Distribution:\n")
        f.write("-" * 40 + "\n")
        for cls, cnt in class_counts.items():
            f.write(f"  Class {cls}: {cnt:3d} ({cnt / n_total:.2%})\n")
        f.write(f"  Total: {n_total}\n\n")

        f.write("Baseline 1: Majority Class Classifier (always predict most frequent class)\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Majority class: {majority_class}\n")
        f.write(f"  Accuracy:       {majority_acc:.4f} ({majority_acc:.2%})\n\n")

        f.write("Your models (XGBoost, Random Forest, Neural Net) should beat this accuracy.\n")

    print(f"[EDA] Saved report to {report_path}")

    # ==========================================
    # 2. PLOTS
    # ==========================================
    # --- Class Distribution ---
    plt.figure(figsize=(6, 4))
    sns.countplot(x=target_col, data=df)
    plt.title("Wine Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "wine_class_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("[EDA] Saved wine_class_distribution.png")

    # --- Correlation Heatmap ---
    feature_cols = [c for c in df.columns if c != target_col]
    corr = df[feature_cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Wine Features Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "wine_correlation_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("[EDA] Saved wine_correlation_heatmap.png")

    print(f"\n[EDA] Complete! All outputs saved to {output_dir}/")
    return df


