import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from io import StringIO

def eda_bank(data):
    # Create eda_plots directory if it doesn't exist
    output_dir = "plots/eda_plots"
    os.makedirs(output_dir, exist_ok=True)

    df = data.copy()
    
    # ----- Save text report with info and describe -----
    report_path = os.path.join(output_dir, "eda_bank_plots.txt")
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("BANK DATA EDA REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("DATA INFO:\n")
        f.write("-" * 80 + "\n")
        buffer = StringIO()
        df.info(buf=buffer)
        f.write(buffer.getvalue())
        f.write("\n\n") 
        
        f.write("DATA DESCRIBE:\n")
        f.write("-" * 80 + "\n")
        f.write(str(df.describe()))
        f.write("\n\n")
        
        # Encode y for correlation
        numeric_cols = df.select_dtypes(include=['int64','float64']).columns
        df["_y"] = (df["y"] == "yes").astype(int)
        corr_with_target = df[numeric_cols.tolist() + ["_y"]].corr()["_y"].sort_values(ascending=False)
        f.write("CORRELATION OF NUMERIC FEATURES WITH TARGET:\n")
        f.write("-" * 80 + "\n")
        f.write(str(corr_with_target))
        f.write("\n\n")
        
        # ===== NAIVE BASELINE =====
        f.write("=" * 80 + "\n")
        f.write("NAIVE BASELINE (What to beat!)\n")
        f.write("=" * 80 + "\n\n")
        
        # Class distribution
        n_total = len(df)
        n_no = (df["y"] == "no").sum()
        n_yes = (df["y"] == "yes").sum()
        pct_no = n_no / n_total
        pct_yes = n_yes / n_total
        
        f.write("Class Distribution:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  No (negative):  {n_no:,} ({pct_no:.2%})\n")
        f.write(f"  Yes (positive): {n_yes:,} ({pct_yes:.2%})\n")
        f.write(f"  Total:          {n_total:,}\n\n")
        
        # Majority class baseline (always predict "no")
        f.write("Baseline 1: Majority Class Classifier (always predict 'no')\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Accuracy:  {pct_no:.4f} ({pct_no:.2%})\n")
        f.write(f"  Precision: 0.0000 (never predicts 'yes')\n")
        f.write(f"  Recall:    0.0000 (misses all 'yes' cases)\n")
        f.write(f"  F1-Score:  0.0000\n")
        f.write(f"  ROC-AUC:   0.5000 (random chance)\n\n")
        
        # Random baseline (predict based on class proportions)
        f.write("Baseline 2: Random Classifier (predict by class proportion)\n")
        f.write("-" * 40 + "\n")
        # Expected accuracy = P(no)^2 + P(yes)^2
        random_acc = pct_no**2 + pct_yes**2
        # Expected precision = P(yes) (when you guess yes, P(yes) of them are actually yes)
        random_prec = pct_yes
        # Expected recall = P(yes) (you guess yes P(yes) of the time, so you catch P(yes) of actual yes)
        random_recall = pct_yes
        random_f1 = 2 * (random_prec * random_recall) / (random_prec + random_recall) if (random_prec + random_recall) > 0 else 0
        f.write(f"  Accuracy:  {random_acc:.4f} ({random_acc:.2%})\n")
        f.write(f"  Precision: {random_prec:.4f}\n")
        f.write(f"  Recall:    {random_recall:.4f}\n")
        f.write(f"  F1-Score:  {random_f1:.4f}\n")
        f.write(f"  ROC-AUC:   0.5000 (random chance)\n\n")
        
        f.write("YOUR MODELS SHOULD BEAT THESE BASELINES!\n")
        f.write(f"  - Accuracy > {pct_no:.2%} (majority class)\n")
        f.write(f"  - ROC-AUC > 0.50 (random chance)\n")
        f.write(f"  - Recall > 0% (actually find some 'yes' cases!)\n")
    
    print(f"Text report saved to {report_path}")

    # ----- class inbalance -----
    plt.figure(figsize=(6, 4))
    sns.countplot(x="y", data=df)
    plt.title("Class Distribution (Subscription)")
    plt.savefig(os.path.join(output_dir, "eda_bank_plots_1.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot 1 saved: eda_bank_plots_1.png")

    # ----- Correlation Heatmap (numeric only) -----
    numeric_cols = df.select_dtypes(include=['int64','float64']).columns
    plt.figure(figsize=(10,8))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap (Numeric Features)")
    plt.savefig(os.path.join(output_dir, "eda_bank_plots_2.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot 2 saved: eda_bank_plots_2.png")

    # ----- Boxplot for age by outcome -----
    plt.figure(figsize=(8,6))
    sns.boxplot(x="y", y="age", data=df)
    plt.title("Age Distribution by Subscription Outcome")
    plt.savefig(os.path.join(output_dir, "eda_bank_plots_3.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot 3 saved: eda_bank_plots_3.png")

    # ----- Category Subscription Rates -----
    plt.figure(figsize=(10,6))
    (df.groupby("job")["y"]
         .value_counts(normalize=True)
         .unstack()["yes"]
         .sort_values()
         .plot(kind="barh"))
    plt.title("Subscription Rate by Job Type")
    plt.xlabel("Rate of Subscription (yes)")
    plt.savefig(os.path.join(output_dir, "eda_bank_plots_4.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot 4 saved: eda_bank_plots_4.png")
    
    print(f"\nAll plots and report saved to {output_dir}/ directory")

